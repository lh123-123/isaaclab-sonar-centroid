# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Polar sonar preprocessing (log + median). Re-exports sonar_flow_hover implementation."""

from __future__ import annotations

from isaaclab_tasks.direct.sonar_flow_hover.sonar_preprocess import (
    FIXED_LOG_MAX,
    float_to_u8_fixed,
    log_compress,
    median_filter_3x3,
    preprocess_polar_batch,
    u8_to_float01,
)

__all__ = [
    "FIXED_LOG_MAX",
    "float_to_u8_fixed",
    "log_compress",
    "median_filter_3x3",
    "preprocess_polar_batch",
    "preprocess_polar_single",
    "u8_to_float01",
]


def preprocess_polar_single(raw: "torch.Tensor") -> "torch.Tensor":
    """Preprocess one polar frame (H, W) → (H, W)."""
    import torch

    if raw.ndim == 2:
        return preprocess_polar_batch(raw.unsqueeze(0))[0]
    if raw.ndim == 3 and raw.shape[0] == 1:
        return preprocess_polar_batch(raw)[0]
    raise ValueError(f"Expected (H,W) or (1,H,W), got {tuple(raw.shape)}")
