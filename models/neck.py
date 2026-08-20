"""
Custom Neck Module - Multi-Scale Feature Fusion (Built From Scratch)
========================================================================
The backbone gives us 4 feature maps at different scales:
  Stage 1: 160x160, 64 ch  (fine detail - very small/distant drones)
  Stage 2: 80x80,  128 ch
  Stage 3: 40x40,  256 ch
  Stage 4: 20x20,  512 ch  (deep semantic features - large/close drones)

Problem this solves:
Deep features (Stage 4) know "what" an object is but lose precise
location info due to heavy downsampling. Shallow features (Stage 1)
have precise location info but weak semantics. A detector that only
uses deep features struggles to LOCALIZE small objects precisely -
this is exactly the mAP50 vs mAP50-95 gap issue we identified earlier.

Solution (inspired by the general idea of feature pyramids, but our
own implementation from scratch): propagate deep semantic info DOWN
to shallow layers via upsampling + fusion, so every scale gets both
"what" and "where" information.
"""

import torch
import torch.nn as nn
from models.backbone import ConvBlock


class FusionBlock(nn.Module):
    """
    Fuses a deeper (semantically strong) feature map with a shallower
    (spatially precise) feature map from the backbone.

    Steps:
      1. Upsample the deep feature map to match the shallow map's size
      2. Project both to the same channel count (1x1 conv)
      3. Add them together
      4. Refine with a conv block
    """

    def __init__(self, deep_channels: int, shallow_channels: int,
                 out_channels: int, activation: str = "silu"):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # 1x1 convs project both inputs to the same channel dimension
        self.deep_proj = nn.Conv2d(deep_channels, out_channels, kernel_size=1)
        self.shallow_proj = nn.Conv2d(shallow_channels, out_channels, kernel_size=1)

        self.refine = ConvBlock(out_channels, out_channels, kernel_size=3,
                                 stride=1, activation=activation)

    def forward(self, deep_feat: torch.Tensor, shallow_feat: torch.Tensor) -> torch.Tensor:
        deep_up = self.upsample(deep_feat)
        deep_up = self.deep_proj(deep_up)

        shallow_proj = self.shallow_proj(shallow_feat)

        fused = deep_up + shallow_proj
        return self.refine(fused)


class CustomNeck(nn.Module):
    """
    Takes all 4 backbone feature maps and produces `num_scales`
    fused feature maps (configurable) - each carrying both fine
    spatial detail AND strong semantic information.

    Output feature maps are what the Detection Head will use to
    predict boxes at multiple scales.
    """

    def __init__(self, config: dict):
        super().__init__()

        backbone_filters = config["model"]["backbone"]["filters"]  # e.g. [64,128,256,512]
        neck_cfg = config["model"]["neck"]
        self.num_scales = neck_cfg["num_scales"]
        fusion_ch = neck_cfg["fusion_channels"]
        activation = config["model"]["backbone"]["activation"]

        assert self.num_scales <= len(backbone_filters), \
            "num_scales cannot exceed number of backbone stages"

        # We fuse top-down: deepest stage -> progressively shallower stages.
        # Build one FusionBlock per adjacent pair of stages we're using.
        # e.g. for num_scales=3 using stages [2,3,4] (0-indexed: 1,2,3):
        #   fuse(stage4, stage3) -> P3
        #   fuse(P3_upsampled_equivalent, stage2) -> P2
        self.fusion_blocks = nn.ModuleList()

        # indices of backbone stages we'll use, from deepest to shallowest
        used_indices = list(range(len(backbone_filters) - self.num_scales,
                                   len(backbone_filters)))[::-1]
        # e.g. num_scales=3, 4 stages total -> used_indices = [3, 2, 1] (deep to shallow)

        for i in range(len(used_indices) - 1):
            shallow_idx = used_indices[i + 1]
            # NOTE: the "deep" input to every fusion block is always `current`,
            # which already has fusion_ch channels because it passed through
            # self.deep_proj (first stage) or a previous FusionBlock's `refine`
            # conv (subsequent stages). Only the shallow input comes straight
            # from the backbone, so only its channel count varies.
            deep_ch = fusion_ch
            shallow_ch = backbone_filters[shallow_idx]

            self.fusion_blocks.append(
                FusionBlock(deep_ch, shallow_ch, fusion_ch, activation)
            )

        # Also project the deepest feature map on its own (no shallower map to fuse with it)
        self.deep_proj = nn.Conv2d(backbone_filters[used_indices[0]], fusion_ch, kernel_size=1)

        self.used_indices = used_indices

    def forward(self, backbone_features: list) -> list:
        """
        backbone_features: list of feature maps from CustomBackbone,
                            ordered shallow -> deep (stage1, stage2, ...)

        Returns: list of fused feature maps, ordered deep -> shallow
                 (i.e. large-object scale first, small-object scale last)
        """
        # Select only the stages we're using, deep to shallow
        selected = [backbone_features[i] for i in self.used_indices]

        outputs = []
        current = self.deep_proj(selected[0])
        outputs.append(current)

        for i, fusion_block in enumerate(self.fusion_blocks):
            current = fusion_block(current, selected[i + 1])
            outputs.append(current)

        return outputs  # [deepest/large-object scale, ..., shallowest/small-object scale]


if __name__ == "__main__":
    import yaml
    from models.backbone import CustomBackbone

    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    backbone = CustomBackbone(config)
    neck = CustomNeck(config)

    dummy_input = torch.randn(2, 3, 640, 640)
    backbone_feats = backbone(dummy_input)
    neck_feats = neck(backbone_feats)

    print(f"Backbone produced {len(backbone_feats)} feature maps:")
    for i, f in enumerate(backbone_feats):
        print(f"  Stage {i+1}: {f.shape}")

    print(f"\nNeck fused them into {len(neck_feats)} scales (deep -> shallow):")
    for i, f in enumerate(neck_feats):
        print(f"  Fused scale {i+1}: {f.shape}")

    total_params = sum(p.numel() for p in neck.parameters())
    print(f"\nTotal neck parameters: {total_params:,}")
    print("Neck sanity check passed.")
