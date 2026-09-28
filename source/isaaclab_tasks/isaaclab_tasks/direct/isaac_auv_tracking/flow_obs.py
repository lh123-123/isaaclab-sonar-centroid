# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Stacked polar residual + centroid displacement for Direct-v2 (no Isaac Sim)."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

# Same ceiling as sonar_flow_hover.sonar_preprocess.FIXED_LOG_MAX (keep this file Isaac-free).
FIXED_LOG_MAX = float(np.log1p(50.0))

FLOW_REF_RANGE_M = 2.0
FLOW_REF_BEARING_RAD = 0.0
FLOW_STATE_DIM = 8


def wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def normalize_polar_fixed(polar: torch.Tensor, log_max: float = FIXED_LOG_MAX) -> torch.Tensor:
    """Log-compressed polar → [0, 1] with a fixed ceiling (stable across frames)."""
    return (polar / max(float(log_max), 1.0e-6)).clamp(0.0, 1.0)


def photometric_residual(curr_01: torch.Tensor, prev_01: torch.Tensor, ready: torch.Tensor) -> torch.Tensor:
    """``I_t − I_{t-1}``; zeros where the previous frame is not valid."""
    residual = curr_01 - prev_01
    mask = ready.to(dtype=residual.dtype).view(-1, 1, 1)
    return residual * mask


def pack_flow_image(
    curr_01: torch.Tensor,
    residual: torch.Tensor,
    *,
    size: tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """``[N,H,W]`` current + residual → uint8 ``[N,3,h,w]``.

    ch0 = current polar, ch1 = residual mapped so 128 = no change, ch2 = |residual|.
    """
    ch0 = curr_01.clamp(0.0, 1.0)
    ch1 = ((residual + 1.0) * 0.5).clamp(0.0, 1.0)
    ch2 = residual.abs().clamp(0.0, 1.0)
    stacked = torch.stack([ch0, ch1, ch2], dim=1)
    h, w = int(size[0]), int(size[1])
    if stacked.shape[-2] != h or stacked.shape[-1] != w:
        stacked = F.interpolate(stacked, size=(h, w), mode="bilinear", align_corners=False)
    return (stacked * 255.0).clamp(0.0, 255.0).to(torch.uint8)


def centroid_displacement(
    r: torch.Tensor,
    alpha: torch.Tensor,
    valid: torch.Tensor,
    r_prev: torch.Tensor,
    alpha_prev: torch.Tensor,
    valid_prev: torch.Tensor,
    ready: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inter-frame ``(Δr, Δα)`` and a flow-valid mask.

    Displacement is over one control step (0.2 s). Image-plane speed is ``Δ / dt``.
    """
    valid_b = valid.bool()
    prev_b = valid_prev.bool()
    ready_b = ready.bool()
    flow_valid = ready_b & valid_b & prev_b
    dr = torch.where(flow_valid, r - r_prev, torch.zeros_like(r))
    dalpha = torch.where(flow_valid, wrap_pi(alpha - alpha_prev), torch.zeros_like(alpha))
    return dr, dalpha, flow_valid


def build_flow_state(
    r: torch.Tensor,
    alpha: torch.Tensor,
    valid: torch.Tensor,
    r_star: torch.Tensor,
    alpha_star: torch.Tensor,
    dr: torch.Tensor,
    dalpha: torch.Tensor,
    flow_valid: torch.Tensor,
) -> torch.Tensor:
    """``[r, α, r*, α*, Δr, Δα, valid_detect, valid_flow]``."""
    obs = torch.stack(
        [
            r,
            alpha,
            r_star.expand_as(r) if r_star.ndim == 0 else r_star,
            alpha_star.expand_as(alpha) if alpha_star.ndim == 0 else alpha_star,
            dr,
            dalpha,
            valid.float(),
            flow_valid.float(),
        ],
        dim=-1,
    )
    return torch.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0)


def intensity_centroid_polar(
    polar: torch.Tensor,
    *,
    min_range: float,
    max_range: float,
    hori_fov_deg: float,
    min_peak: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched intensity-weighted centroid on polar ``[N,H,W]`` → ``(r, α, valid)``.

    Used by unit tests and as a GPU fallback. Env measurement still goes through
    ``detect_centroid_polar`` for KF; flow obs can use this or the same detector.
    """
    if polar.ndim != 3:
        raise ValueError(f"expected (N,H,W), got {tuple(polar.shape)}")
    n, h, w = polar.shape
    device = polar.device
    peak = polar.amax(dim=(-2, -1))
    valid = peak > max(float(min_peak), 1.0e-8)
    weights = polar.clamp(min=0.0)
    mass = weights.flatten(1).sum(dim=-1).clamp(min=1.0e-8)
    rows = torch.arange(h, device=device, dtype=polar.dtype).view(1, h, 1)
    cols = torch.arange(w, device=device, dtype=polar.dtype).view(1, 1, w)
    row_c = (weights * rows).flatten(1).sum(dim=-1) / mass
    col_c = (weights * cols).flatten(1).sum(dim=-1) / mass
    h_max = max(float(h) - 1.0, 1.0)
    span = max(float(max_range) - float(min_range), 1.0e-6)
    r = float(min_range) + row_c / h_max * span
    # Matches sonar_meas.col_to_azimuth_rad: +az → lower col.
    fov_deg = max(float(hori_fov_deg), 1.0e-6)
    az_deg = (float(w) / 2.0 - col_c) * fov_deg / max(float(w), 1.0)
    alpha = torch.deg2rad(az_deg)
    r = torch.where(valid, r, torch.zeros_like(r))
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    return r, alpha, valid


def make_gaussian_blob(
    *,
    height: int,
    width: int,
    row: float,
    col: float,
    sigma: float = 1.6,
    amplitude: float = 2.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Single-env synthetic polar bright spot ``[H,W]`` (log-ish intensity)."""
    yy = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
    blob = amplitude * torch.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2.0 * sigma * sigma))
    return blob


def reference_polar_pixel(
    *,
    r_star: float,
    min_range: float,
    max_range: float,
    polar_height: int,
    polar_width: int,
) -> tuple[float, float]:
    """``(row*, col*)`` of the fixed tracking reference on a polar image."""
    h_max = max(float(polar_height) - 1.0, 1.0)
    span = max(float(max_range) - float(min_range), 1.0e-6)
    row = (float(r_star) - float(min_range)) / span * h_max
    col = 0.5 * (float(polar_width) - 1.0)
    return float(row), float(col)
