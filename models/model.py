"""
AttentiveDroneNet - Full Model Assembly
==========================================
Combines Backbone -> Neck -> Detection Head into a single model class.
This is the "reusable" entry point mentioned throughout the roadmap:
one object, one forward() call, fully driven by the config dict.

To reuse this model for a DIFFERENT project later (per Section 22 of
the roadmap - "What does Reusable mean?"), you would only need to:
  - Point `data/dataset.py` at a new dataset
  - Adjust `configs/base_config.yaml` (num_classes, input_size, etc.)
  - The architecture code here does not need to change.
"""

import torch
import torch.nn as nn
from models.backbone import CustomBackbone
from models.neck import CustomNeck
from models.head import CustomDetectionHead


class AttentiveDroneNet(nn.Module):
    """
    Full detection model: Backbone -> Neck -> Detection Head.

    Forward pass:
      image (B, 3, H, W)
        -> backbone -> list of 4 multi-scale feature maps
        -> neck     -> list of N fused feature maps (N = config num_scales)
        -> head     -> list of N prediction dicts {bbox, objectness, class}
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.backbone = CustomBackbone(config)
        self.neck = CustomNeck(config)
        self.head = CustomDetectionHead(config)

    def forward(self, x: torch.Tensor) -> list:
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        predictions = self.head(neck_features)
        return predictions

    def count_parameters(self) -> dict:
        """Returns a breakdown of parameter counts per sub-module.
        Useful for logging/documentation (roadmap Section 17 - experiment tracking)."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        neck_params = sum(p.numel() for p in self.neck.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        total = backbone_params + neck_params + head_params

        return {
            "backbone": backbone_params,
            "neck": neck_params,
            "head": head_params,
            "total": total,
        }


def build_model(config: dict) -> AttentiveDroneNet:
    """
    Factory function - the single place that constructs the model
    from a config dict. Keeps model creation consistent across
    train.py, evaluate.py, and predict.py (roadmap Section 9).
    """
    model = AttentiveDroneNet(config)
    return model


if __name__ == "__main__":
    import yaml

    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = build_model(config)

    dummy_input = torch.randn(2, 3, 640, 640)
    predictions = model(dummy_input)

    print("=" * 50)
    print("AttentiveDroneNet - Full Model Test")
    print("=" * 50)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Number of output scales: {len(predictions)}\n")

    for i, pred in enumerate(predictions):
        print(f"Scale {i+1}:")
        print(f"  bbox:       {pred['bbox'].shape}")
        print(f"  objectness: {pred['objectness'].shape}")
        print(f"  class:      {pred['class'].shape}")

    param_counts = model.count_parameters()
    print("\n" + "=" * 50)
    print("Parameter Breakdown:")
    print("=" * 50)
    for module_name, count in param_counts.items():
        print(f"  {module_name:12s}: {count:,}")

    print("\nFull model sanity check passed.")