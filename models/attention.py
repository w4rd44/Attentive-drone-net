"""
Custom Attention Modules - Built From Scratch
================================================
These are NOT imported from any library (no CBAM/SE-block package).
Every operation is implemented manually using basic PyTorch ops
(Conv2d, Linear, pooling, activation) so the architecture is fully
our own design.

Why this exists (project motivation):
Previous YOLO experiments on the drone dataset showed a large gap
between mAP50 (98.1%) and mAP50-95 (~51%). This indicates the model
finds drones but localizes them imprecisely - especially small or
distant ones. Attention helps the network weight the informative
regions/channels more strongly, which should tighten bounding boxes.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Learns "which feature channels matter most" for detecting drones.
    Built from scratch using global pooling + a small MLP (no SE-block import).
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced_channels = max(in_channels // reduction_ratio, 8)

        # Global average pooling branch (captures overall channel importance)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Global max pooling branch (captures the most salient activation)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared small MLP applied to both pooled vectors
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))

        # Combine both pooled signals, then squash to [0, 1] weights
        channel_weights = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

        return x * channel_weights  # reweight each channel


class SpatialAttention(nn.Module):
    """
    Learns "which spatial regions matter most" for detecting drones.
    Useful for small/distant drones that occupy only a few pixels.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2

        # Takes a 2-channel map (avg + max across channels) -> 1-channel attention map
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)   # avg across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # max across channels

        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        spatial_weights = self.sigmoid(self.conv(combined))  # (B, 1, H, W)

        return x * spatial_weights  # reweight each spatial location


class CustomAttentionBlock(nn.Module):
    """
    Combines channel + spatial attention. Both are optional/configurable
    via config, per the "modular components" requirement in the roadmap.
    """

    def __init__(self, in_channels: int, use_channel: bool = True,
                 use_spatial: bool = True, reduction_ratio: int = 16):
        super().__init__()
        self.use_channel = use_channel
        self.use_spatial = use_spatial

        if use_channel:
            self.channel_attn = ChannelAttention(in_channels, reduction_ratio)
        if use_spatial:
            self.spatial_attn = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_channel:
            x = self.channel_attn(x)
        if self.use_spatial:
            x = self.spatial_attn(x)
        return x


if __name__ == "__main__":
    # Quick sanity check
    dummy_input = torch.randn(2, 128, 40, 40)  # (batch, channels, H, W)
    attn_block = CustomAttentionBlock(in_channels=128)
    output = attn_block(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape
    print("Attention block sanity check passed.")
