"""
Custom CNN Backbone - Built From Scratch
==========================================
This is the feature-extraction part of the model. It takes a raw
image and progressively extracts higher-level features while
reducing spatial resolution and increasing channel depth.

Fully configurable via config.yaml:
- num_stages: how many downsampling stages
- filters: channel count per stage
- kernel_size, activation: per-conv settings

Design note:
The backbone returns features from MULTIPLE stages (not just the
last one). This is required later by the Neck module, which fuses
multi-scale features to help detect both small/distant drones and
large/close drones.
"""

import torch
import torch.nn as nn
from models.attention import CustomAttentionBlock


def get_activation(name: str) -> nn.Module:
    """Small factory so activation type is configurable from YAML."""
    name = name.lower()
    if name == "silu":
        return nn.SiLU(inplace=True)
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    else:
        raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    """
    Basic reusable building block: Conv -> BatchNorm -> Activation.
    This is the fundamental unit the whole backbone is built from.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, activation: str = "silu"):
        super().__init__()
        padding = kernel_size // 2  # keeps spatial size predictable ("same" padding)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BackboneStage(nn.Module):
    """
    One stage of the backbone = 2 conv blocks (first one downsamples
    via stride=2) + an optional attention block at the end of the stage.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, activation: str,
                 use_attention: bool = True,
                 attn_channel: bool = True, attn_spatial: bool = True,
                 attn_reduction: int = 16):
        super().__init__()

        self.downsample_conv = ConvBlock(
            in_channels, out_channels, kernel_size, stride=2, activation=activation
        )
        self.refine_conv = ConvBlock(
            out_channels, out_channels, kernel_size, stride=1, activation=activation
        )

        self.use_attention = use_attention
        if use_attention:
            self.attention = CustomAttentionBlock(
                in_channels=out_channels,
                use_channel=attn_channel,
                use_spatial=attn_spatial,
                reduction_ratio=attn_reduction,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_conv(x)
        x = self.refine_conv(x)
        if self.use_attention:
            x = self.attention(x)
        return x


class CustomBackbone(nn.Module):
    """
    Full backbone: a stack of BackboneStages.
    Returns a LIST of feature maps (one per stage) so the Neck
    module can later fuse multiple scales together.
    """

    def __init__(self, config: dict):
        super().__init__()

        backbone_cfg = config["model"]["backbone"]
        attn_cfg = config["model"]["attention"]

        num_stages = backbone_cfg["num_stages"]
        filters = backbone_cfg["filters"]
        kernel_size = backbone_cfg["kernel_size"]
        activation = backbone_cfg["activation"]

        assert len(filters) == num_stages, \
            "Number of filter values must match num_stages in config"

        # Stem: first conv that takes raw RGB input (3 channels)
        self.stem = ConvBlock(3, filters[0], kernel_size=3, stride=2, activation=activation)

        # Build stages sequentially, each stage downsamples further
        self.stages = nn.ModuleList()
        in_ch = filters[0]
        for i in range(num_stages):
            out_ch = filters[i]
            self.stages.append(
                BackboneStage(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    activation=activation,
                    use_attention=True,
                    attn_channel=attn_cfg["use_channel"],
                    attn_spatial=attn_cfg["use_spatial"],
                    attn_reduction=attn_cfg["reduction_ratio"],
                )
            )
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> list:
        """
        Returns a list of feature maps from each stage, e.g.:
        [stage1_out, stage2_out, stage3_out, stage4_out]
        Later stages = smaller spatial size, more channels (deeper features).
        """
        x = self.stem(x)

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features


if __name__ == "__main__":
    import yaml

    # Quick sanity check using the actual config file
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    backbone = CustomBackbone(config)
    dummy_input = torch.randn(2, 3, 640, 640)  # (batch, RGB, H, W)

    features = backbone(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Number of feature maps returned: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Stage {i+1} output shape: {feat.shape}")

    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"\nTotal backbone parameters: {total_params:,}")
    print("Backbone sanity check passed.")
