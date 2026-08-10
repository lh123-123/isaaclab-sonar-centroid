# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase A2 base law: GT position + lead + vy loop (frozen Phase A1 PD baseline)."""

from __future__ import annotations

import torch

from .auv_image_range_base import desired_surge_from_range


def compute_position_lead_base_setpoints(
    *,
    target_dy: torch.Tensor,
    target_vy: torch.Tensor,
    robot_sway_m: torch.Tensor,
    robot_sway_vy: torch.Tensor,
    surge_robot_m: torch.Tensor,
    kp_sway: float,
    lead_s: float,
    kp_vy: float,
    sway_amp_m: float,
    kp_range: float = 0.0,
    kp_surge_hold: float = 2.0,
    max_surge: float = 1.0,
    range_meas: torch.Tensor | None = None,
    range_ref_m: torch.Tensor | None = None,
    range_valid: torch.Tensor | None = None,
    range_deadband_m: float = 0.0,
    kp_range_fwd: float | None = None,
    kp_range_back: float | None = None,
    range_close_margin_m: float = 0.25,
    inhibit_surge_forward: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Oracle sine setpoints + optional surge hold (kp_range=0 by default)."""
    tgt_dy = target_dy.view(-1).float()
    tgt_vy = target_vy.view(-1).float()
    sway = robot_sway_m.view(-1).float()
    sway_vy = robot_sway_vy.view(-1).float()
    surge = surge_robot_m.view(-1).float()

    lead_m = float(lead_s) * tgt_vy
    pos_m = float(kp_sway) * (tgt_dy - sway)
    vel_m = float(kp_vy) * (tgt_vy - sway_vy)
    base_sway = tgt_dy + lead_m + pos_m + vel_m

    amp = max(float(sway_amp_m), 1.0e-3)
    base_sway = torch.clamp(base_sway, -amp * 1.05, amp * 1.05)

    n = base_sway.shape[0]
    device = base_sway.device
    if range_meas is None:
        range_meas = torch.zeros(n, device=device)
    if range_ref_m is None:
        range_ref_m = torch.zeros(n, device=device)
    if range_valid is None:
        range_valid = torch.ones(n, device=device)

    base_surge = desired_surge_from_range(
        range_meas=range_meas,
        range_ref_gt_m=range_ref_m,
        valid=range_valid,
        surge_robot_m=surge,
        kp_range=float(kp_range),
        kp_surge_hold=float(kp_surge_hold),
        max_surge=float(max_surge),
        range_deadband_m=float(range_deadband_m),
        kp_range_fwd=kp_range_fwd,
        kp_range_back=kp_range_back,
        range_close_margin_m=float(range_close_margin_m),
        inhibit_surge_forward=inhibit_surge_forward,
    )
    return base_sway, base_surge
