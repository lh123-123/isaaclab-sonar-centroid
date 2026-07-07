"""Sonar column profile: subsampled per-beam peak intensity (raw image columns)."""

from __future__ import annotations

import torch


def column_profile_dim(beam_count: int, stride: int = 3) -> int:
    """Number of profile bins when subsampling ``beam_count`` beams by ``stride``."""
    if beam_count < 1 or stride < 1:
        return 0
    return (beam_count - 1) // stride + 1


def beam_intensity_column_profile(
    beam_data: torch.Tensor,
    *,
    stride: int = 3,
    col_contrast: bool = True,
) -> torch.Tensor:
    """Build column profile from beam triplets [N, B, 3] (angle, range, intensity).

    Subsamples every ``stride`` beams (61 beams / stride 3 → 21 bins).
    Beam 0 → profile[0] (−60°), center → profile[10] (0°), last → profile[20] (+60°).
    """
    if beam_data.ndim != 3 or beam_data.shape[-1] != 3:
        raise ValueError(f"Expected [N,B,3], got {tuple(beam_data.shape)}")

    profile = beam_data[:, ::stride, 2]
    if col_contrast:
        profile = profile - profile.median(dim=1, keepdim=True).values
    return profile
