"""Module D: sonar CNN feature encoder (design doc §3.4)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SonarFeatureEncoder(nn.Module):
    """Lightweight CNN: grayscale sonar image -> feature vector."""

    def __init__(
        self,
        height: int = 64,
        width: int = 128,
        feature_dim: int = 256,
        history_len: int = 1,
    ):
        super().__init__()
        self.history_len = max(1, int(history_len))
        in_channels = self.history_len

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, feature_dim),
            nn.ELU(),
        )
        self.angle_proj = nn.Linear(1, feature_dim)

    def forward(
        self,
        sonar_frames: torch.Tensor,
        sonar_angle: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode sonar history.

        Args:
            sonar_frames: [B, H, W] or [B, N, H, W]
            sonar_angle: optional [B, 1] heading (rad), fused at the end
        """
        if sonar_frames.dim() == 3:
            x = sonar_frames.unsqueeze(1)
        elif sonar_frames.dim() == 4:
            x = sonar_frames.reshape(sonar_frames.shape[0], -1, sonar_frames.shape[-2], sonar_frames.shape[-1])
        else:
            raise ValueError(f"Expected [B,H,W] or [B,N,H,W], got {tuple(sonar_frames.shape)}")

        feat = self.head(self.backbone(x))
        if sonar_angle is not None:
            angle = sonar_angle.reshape(feat.shape[0], 1)
            feat = feat + self.angle_proj(angle)
        return feat
