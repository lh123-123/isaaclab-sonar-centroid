# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2: discrete P controller from fixed-point tracking error."""

from __future__ import annotations

import torch

from .constants import ACTION_LEFT, ACTION_RIGHT, ACTION_STAY


def tracking_error_to_lateral_velocity(
    u_norm: float | torch.Tensor,
    *,
    kp: float = 0.15,
    max_vy: float = 0.15,
    deadband: float = 0.008,
) -> float | torch.Tensor:
    """Continuous P: vy = clip(kp * u_norm, ±max_vy). Same sign convention as discrete."""
    if isinstance(u_norm, torch.Tensor):
        u = u_norm
        vy = kp * u
        vy = torch.where(u.abs() <= deadband, torch.zeros_like(vy), vy)
        return torch.clamp(vy, -max_vy, max_vy)

    u = float(u_norm)
    if abs(u) <= deadband:
        return 0.0
    return float(max(-max_vy, min(max_vy, kp * u)))


def sonar_col_delta_norm(
    sonar_col: float | torch.Tensor,
    fixed_col: float | torch.Tensor,
    *,
    polar_width: float,
) -> float | torch.Tensor:
    """Normalized azimuth gap: green_col − yellow_col on polar image (col = x / azimuth)."""
    delta = sonar_col - fixed_col
    w = max(float(polar_width), 1.0)
    if isinstance(delta, torch.Tensor):
        return delta / w
    return float(delta) / w


def image_delta_lateral_velocity(
    delta_col_norm: float | torch.Tensor,
    target_vy: float | torch.Tensor = 0.0,
    *,
    kp: float = 1.0,
    kff: float = 1.0,
    max_vy: float = 0.32,
    deadband: float = 0.003,
) -> float | torch.Tensor:
    """vy = kff·target_vy + kp·(green_col − yellow_col)/W; drive green onto yellow."""
    if isinstance(delta_col_norm, torch.Tensor):
        vy = kff * target_vy
        vy = vy + torch.where(
            delta_col_norm.abs() > deadband, kp * delta_col_norm, torch.zeros_like(vy)
        )
        return torch.clamp(vy, -max_vy, max_vy)

    vy = float(kff) * float(target_vy)
    d = float(delta_col_norm)
    if abs(d) > deadband:
        vy += float(kp) * d
    return float(max(-max_vy, min(max_vy, vy)))


def moving_target_lateral_velocity(
    u_norm: float | torch.Tensor,
    target_vy: float | torch.Tensor,
    sway_err: float | torch.Tensor,
    *,
    kff: float = 1.0,
    kp_img: float = 0.55,
    kp_sway: float = 0.65,
    max_vy: float = 0.32,
    deadband_img: float = 0.004,
    deadband_sway: float = 0.015,
) -> float | torch.Tensor:
    """FF on target speed + image P + sway position P (sway_err = target_dy − robot_sway)."""
    if isinstance(u_norm, torch.Tensor):
        vy = kff * target_vy
        vy = vy + torch.where(u_norm.abs() > deadband_img, kp_img * u_norm, torch.zeros_like(vy))
        vy = vy + torch.where(sway_err.abs() > deadband_sway, kp_sway * sway_err, torch.zeros_like(vy))
        return torch.clamp(vy, -max_vy, max_vy)

    vy = float(kff) * float(target_vy)
    u = float(u_norm)
    if abs(u) > deadband_img:
        vy += float(kp_img) * u
    se = float(sway_err)
    if abs(se) > deadband_sway:
        vy += float(kp_sway) * se
    return float(max(-max_vy, min(max_vy, vy)))


def tracking_error_to_discrete_action(
    u_norm: float | torch.Tensor,
    *,
    kp: float = 1.0,
    deadband: float = 0.01,
) -> int | torch.Tensor:
    """Map normalized lateral error u/W to {left, stay, right}.

    u > 0: target appears right of fixed point → move robot right (ACTION_RIGHT).
    u < 0: move left (ACTION_LEFT).
    """
    if isinstance(u_norm, torch.Tensor):
        u = u_norm
        out = torch.full_like(u, ACTION_STAY, dtype=torch.long)
        out = torch.where(u > deadband, torch.full_like(out, ACTION_RIGHT), out)
        out = torch.where(u < -deadband, torch.full_like(out, ACTION_LEFT), out)
        return out

    u = float(u_norm)
    if abs(u) <= deadband:
        return ACTION_STAY
    return ACTION_RIGHT if u > 0.0 else ACTION_LEFT
