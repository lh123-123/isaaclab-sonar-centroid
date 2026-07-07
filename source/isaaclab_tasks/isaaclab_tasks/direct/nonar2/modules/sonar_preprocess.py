"""Module A: sonar image preprocessing (design doc §3.1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def adaptive_histogram_equalization(image: torch.Tensor) -> torch.Tensor:
    """Per-frame contrast stretch (2% / 98% quantiles)."""
    flat = image.reshape(image.shape[0], -1)
    lo = flat.quantile(0.02, dim=1, keepdim=True)
    hi = flat.quantile(0.98, dim=1, keepdim=True)
    span = torch.clamp(hi - lo, min=1.0e-6)
    norm = (flat - lo) / span
    return torch.clamp(norm.reshape_as(image), 0.0, 1.0)


def preprocess_sonar(image: torch.Tensor) -> torch.Tensor:
    """Normalize and lightly denoise sonar images [N, H, W]."""
    if image.ndim != 3:
        raise ValueError(f"Expected [N,H,W], got shape {tuple(image.shape)}")

    x = adaptive_histogram_equalization(image)
    x = x.unsqueeze(1)
    pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    blurred = (
        pad[:, :, :-2, :-2]
        + pad[:, :, :-2, 1:-1]
        + pad[:, :, :-2, 2:]
        + pad[:, :, 1:-1, :-2]
        + pad[:, :, 1:-1, 1:-1]
        + pad[:, :, 1:-1, 2:]
        + pad[:, :, 2:, :-2]
        + pad[:, :, 2:, 1:-1]
        + pad[:, :, 2:, 2:]
    ) / 9.0
    return torch.clamp(blurred.squeeze(1), 0.0, 1.0)
