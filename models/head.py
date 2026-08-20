"""
Custom Detection Head - Built From Scratch (Anchor-Free)
============================================================
Takes the Neck's fused multi-scale feature maps and produces the
final predictions at every spatial location (grid cell) of each scale:

  1. Bounding box regression: (tx, ty, tw, th) - offsets to compute
     the actual box position/size at that grid cell
  2. Objectness score: how confident the model is that THIS grid
     cell contains a drone (vs background)
  3. Class probability: what class the object is (for us: just
     "drone", but kept general/configurable for future reuse per
     the roadmap's "reusable across projects" requirement)

Anchor-free design (no predefined anchor boxes):
Instead of predicting offsets relative to fixed anchor shapes (like
classic YOLOv3/v4), we directly predict box size/position per grid
cell. This is simpler, avoids anchor-tuning as a hyperparameter, and
is the direction modern detectors (YOLOv8, FCOS) have moved toward.

Each scale from the Neck gets its OWN head instance (weights are not
shared across scales) so the head can specialize slightly per
object-size range, while still following the same architecture.
"""

import torch
import torch.nn as nn
from models.backbone import ConvBlock


class DetectionHeadSingleScale(nn.Module):
    """
    Detection head for ONE feature scale (e.g. just the 80x80 map).
    Splits into two lightweight branches (common in modern detectors):
      - Regression branch -> bounding box coordinates
      - Classification branch -> objectness + class scores
    Splitting branches (instead of one shared conv stack) lets each
    branch specialize: box regression cares about precise edges,
    classification cares about semantic "what is this".
    """

    def __init__(self, in_channels: int, num_classes: int, activation: str = "silu"):
        super().__init__()
        self.num_classes = num_classes

        # --- Regression branch: predicts (tx, ty, tw, th) per cell ---
        self.reg_branch = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, activation=activation),
            ConvBlock(in_channels, in_channels, kernel_size=3, activation=activation),
        )
        self.reg_output = nn.Conv2d(in_channels, 4, kernel_size=1)  # tx, ty, tw, th

        # --- Classification branch: predicts objectness + class scores ---
        self.cls_branch = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, activation=activation),
            ConvBlock(in_channels, in_channels, kernel_size=3, activation=activation),
        )
        self.objectness_output = nn.Conv2d(in_channels, 1, kernel_size=1)         # is there a drone here?
        self.class_output = nn.Conv2d(in_channels, num_classes, kernel_size=1)    # what class?

    def forward(self, x: torch.Tensor) -> dict:
        # Regression branch
        reg_feat = self.reg_branch(x)
        bbox_pred = self.reg_output(reg_feat)  # (B, 4, H, W)

        # Classification branch
        cls_feat = self.cls_branch(x)
        objectness_pred = self.objectness_output(cls_feat)  # (B, 1, H, W)
        class_pred = self.class_output(cls_feat)             # (B, num_classes, H, W)

        return {
            "bbox": bbox_pred,
            "objectness": objectness_pred,
            "class": class_pred,
        }


class CustomDetectionHead(nn.Module):
    """
    Wraps one DetectionHeadSingleScale per scale coming from the Neck.
    Input: list of feature maps from CustomNeck (deep -> shallow order)
    Output: list of prediction dicts, one per scale, same order.
    """

    def __init__(self, config: dict):
        super().__init__()

        neck_cfg = config["model"]["neck"]
        head_cfg = config["model"]["head"]
        backbone_cfg = config["model"]["backbone"]

        in_channels = neck_cfg["fusion_channels"]   # all neck outputs share this channel count
        num_scales = neck_cfg["num_scales"]
        num_classes = head_cfg["num_classes"]
        activation = backbone_cfg["activation"]

        self.heads = nn.ModuleList([
            DetectionHeadSingleScale(in_channels, num_classes, activation)
            for _ in range(num_scales)
        ])

    def forward(self, neck_features: list) -> list:
        assert len(neck_features) == len(self.heads), \
            "Number of neck feature maps must match number of detection heads"

        predictions = []
        for feat, head in zip(neck_features, self.heads):
            predictions.append(head(feat))
        return predictions


if __name__ == "__main__":
    import yaml
    from models.backbone import CustomBackbone
    from models.neck import CustomNeck

    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    backbone = CustomBackbone(config)
    neck = CustomNeck(config)
    head = CustomDetectionHead(config)

    dummy_input = torch.randn(2, 3, 640, 640)
    backbone_feats = backbone(dummy_input)
    neck_feats = neck(backbone_feats)
    predictions = head(neck_feats)

    print(f"Number of detection scales: {len(predictions)}\n")
    for i, pred in enumerate(predictions):
        print(f"Scale {i+1}:")
        print(f"  bbox shape:       {pred['bbox'].shape}       (B, 4, H, W)")
        print(f"  objectness shape: {pred['objectness'].shape}       (B, 1, H, W)")
        print(f"  class shape:      {pred['class'].shape}       (B, num_classes, H, W)")
        print()

    total_params = sum(p.numel() for p in head.parameters())
    print(f"Total detection head parameters: {total_params:,}")

    full_model_params = (
        sum(p.numel() for p in backbone.parameters())
        + sum(p.numel() for p in neck.parameters())
        + sum(p.numel() for p in head.parameters())
    )
    print(f"Total FULL MODEL parameters so far: {full_model_params:,}")
    print("Detection head sanity check passed.")