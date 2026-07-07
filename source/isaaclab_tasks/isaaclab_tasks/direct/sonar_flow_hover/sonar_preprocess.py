# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sonar image preprocessing: log compression, median filter, polar → Cartesian."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

# Fixed log scale for stable frame-to-frame intensity (LK needs consistent u8 mapping).
FIXED_LOG_MAX = float(np.log1p(50.0))


def float_to_u8_fixed(arr: np.ndarray, log_max: float = FIXED_LOG_MAX) -> np.ndarray:
    """Map log-compressed sonar floats to uint8 with a fixed ceiling."""
    return (arr / max(log_max, 1.0e-6) * 255.0).clip(0, 255).astype(np.uint8)


def u8_to_float01(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 255.0


def log_compress(image: torch.Tensor) -> torch.Tensor:
    """Apply ``log(1 + I)`` compression."""
    return torch.log1p(image.clamp(min=0.0))


def median_filter_3x3(image: torch.Tensor) -> torch.Tensor:
    """3×3 median filter on batched grayscale images (N, H, W)."""
    if image.ndim != 3:
        raise ValueError(f"Expected (N, H, W), got {tuple(image.shape)}")
    out = torch.empty_like(image)
    for i in range(image.shape[0]):
        arr = image[i].detach().cpu().numpy()
        u8 = float_to_u8_fixed(arr)
        filtered = cv2.medianBlur(u8, 3)
        out[i] = torch.from_numpy(u8_to_float01(filtered)).to(image.device)
    return out


def polar_to_cartesian_batch(
    polar: torch.Tensor,
    *,
    hori_fov_deg: float,
    max_range: float,
    out_size: int = 128,
) -> torch.Tensor:
    """Map polar sonar intensity (N, H_range, W_az) to Cartesian (N, out_size, out_size).

    Cartesian frame: x = lateral (right positive), y = forward range.
    """
    if polar.ndim != 3:
        raise ValueError(f"Expected (N, H, W), got {tuple(polar.shape)}")

    n, h_range, w_az = polar.shape
    fov_rad = math.radians(hori_fov_deg)
    az_min = -0.5 * fov_rad
    az_max = 0.5 * fov_rad

    yy, xx = torch.meshgrid(
        torch.linspace(-max_range, max_range, out_size, device=polar.device),
        torch.linspace(-max_range, max_range, out_size, device=polar.device),
        indexing="ij",
    )
    range_m = torch.sqrt(xx * xx + yy * yy)
    azimuth = torch.atan2(xx, yy)

    range_idx = (range_m / max(max_range, 1.0e-6) * (h_range - 1)).clamp(0, h_range - 1)
    az_idx = ((azimuth - az_min) / max(az_max - az_min, 1.0e-6) * (w_az - 1)).clamp(0, w_az - 1)

    in_bounds = (range_m >= 0.0) & (range_m <= max_range) & (azimuth >= az_min) & (azimuth <= az_max)

    r0 = range_idx.floor().long()
    r1 = (r0 + 1).clamp(max=h_range - 1)
    a0 = az_idx.floor().long()
    a1 = (a0 + 1).clamp(max=w_az - 1)
    wr = (range_idx - r0.float()).unsqueeze(0)
    wa = (az_idx - a0.float()).unsqueeze(0)

    out = torch.zeros(n, out_size, out_size, device=polar.device, dtype=polar.dtype)
    for b in range(n):
        p = polar[b]
        v00 = p[r0, a0]
        v01 = p[r0, a1]
        v10 = p[r1, a0]
        v11 = p[r1, a1]
        cart = (1 - wr) * (1 - wa) * v00 + (1 - wr) * wa * v01 + wr * (1 - wa) * v10 + wr * wa * v11
        out[b] = torch.where(in_bounds, cart, torch.zeros_like(cart))
    return out


def preprocess_polar_batch(raw: torch.Tensor, *, skip_median: bool = False) -> torch.Tensor:
    """Log compress + optional median filter on raw polar sonar (N, H_range, W_az)."""
    logged = log_compress(raw)
    if skip_median:
        return logged
    return median_filter_3x3(logged)


def preprocess_sonar_batch(
    raw: torch.Tensor,
    *,
    hori_fov_deg: float,
    max_range: float,
    cart_size: int = 128,
) -> torch.Tensor:
    """Full pipeline: log compress → median filter → polar to Cartesian."""
    filtered = preprocess_polar_batch(raw)
    return polar_to_cartesian_batch(
        filtered,
        hori_fov_deg=hori_fov_deg,
        max_range=max_range,
        out_size=cart_size,
    )


def cartesian_diagonal(size: int) -> float:
    """Image diagonal length for normalizing pixel errors."""
    return float(size) * math.sqrt(2.0)
