# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase A2 base law: pure image P (no GT vy FF) + beam peak range hold on surge.

Range target = hold_meas reference at reset.
Sonar estimate = gated + calibrated + EMA beam range each step.
"""

from __future__ import annotations

import torch

from .ibvs_p_controller import image_delta_lateral_velocity


def desired_surge_from_range(
    *,
    range_meas: torch.Tensor,
    range_ref_gt_m: torch.Tensor,
    valid: torch.Tensor,
    surge_robot_m: torch.Tensor,
    kp_range: float,
    kp_surge_hold: float,
    max_surge: float,
    range_deadband_m: float = 0.0,
    kp_range_fwd: float | None = None,
    kp_range_back: float | None = None,
    range_close_margin_m: float = 0.25,
    inhibit_surge_forward: bool = False,
) -> torch.Tensor:
    """Range-hold surge setpoint (m) in mission forward axis.

    Beam-centric law (``inhibit_surge_forward=False``, deploy step A):
    - ``err < -close_margin``: strong ``kp_back`` retreat
    - ``err < -deadband``: mild retreat (never command further forward)
    - ``err > deadband``: weak ``kp_fwd`` approach toward hold_meas ref
    - in band / invalid: freeze surge setpoint (PD D-term damps velocity only)

    Legacy ``inhibit_surge_forward=True``: task-origin surge=0 except retreat when
    very close (no beam approach — PPO / inhibit path).
    """
    n = range_meas.shape[0]
    device = range_meas.device
    cap = max(float(max_surge), 0.0)
    kp_b = float(kp_range_back if kp_range_back is not None else kp_range)
    kp_f = float(kp_range_fwd if kp_range_fwd is not None else max(float(kp_range) * 0.35, 0.0))
    db = max(float(range_deadband_m), 0.0)
    close_m = max(float(range_close_margin_m), 0.0)

    err = (range_meas - range_ref_gt_m) * valid
    surge = surge_robot_m.view(n).float()
    too_close = err < -db
    too_far = err > db
    very_close = err < -close_m
    in_band = (err >= -db) & (err <= db)

    # Legacy: hard task-origin station; beam only used for emergency retreat.
    if inhibit_surge_forward:
        target = torch.zeros_like(surge)
        if kp_b > 0.0:
            retreat_sp = torch.clamp(surge + kp_b * err, min=-cap, max=0.0)
            target = torch.where(very_close, retreat_sp, target)
        return torch.clamp(target, -cap, cap)

    # Beam hold_meas: corrections relative to current surge; in-band freezes.
    target = surge.clone()
    corr = torch.zeros(n, device=device)
    if kp_b > 0.0 or kp_f > 0.0:
        corr = torch.where(too_close, kp_b * err, corr)
        corr = torch.where(too_far, kp_f * err, corr)
        target = surge + corr
        target = torch.where(too_close, torch.minimum(target, surge), target)
        if kp_b > 0.0:
            retreat_sp = torch.minimum(torch.clamp(surge + kp_b * err, min=-cap), surge)
            target = torch.where(very_close, retreat_sp, target)

    # Invalid meas → err=0 → in_band; freeze. Optional soft pull to task 0 unused here
    # (kp_surge_hold retained for API / callers that still pass it).
    _ = kp_surge_hold
    target = torch.where(in_band, surge, target)

    return torch.clamp(target, -cap, cap)


def compute_image_range_base_setpoints(
    *,
    tracker,
    robot_sway_m: torch.Tensor,
    surge_robot_m: torch.Tensor,
    range_ref_gt_m: torch.Tensor,
    polar_width: int,
    polar_height: int,
    step_dt: float,
    kp_img: float,
    deadband: float,
    kp_range: float,
    kp_surge_hold: float,
    max_surge: float,
    sway_amp_m: float,
    range_meas: torch.Tensor,
    range_valid: torch.Tensor,
    range_deadband_m: float = 0.0,
    kp_range_fwd: float | None = None,
    kp_range_back: float | None = None,
    range_close_margin_m: float = 0.25,
    inhibit_surge_forward: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env base sway/surge setpoints from sonar image P + beam range hold (kff=0)."""
    n = robot_sway_m.shape[0]
    device = robot_sway_m.device
    base_sway = torch.zeros(n, device=device)

    w = max(float(polar_width), 1.0)
    amp = max(float(sway_amp_m), 1.0e-3)

    for i in range(n):
        p_rel = tracker.detected_rel(env_id=i)
        delta_norm = 0.0
        if p_rel is not None:
            delta_norm = float(p_rel.col) / w
            if abs(delta_norm) <= deadband:
                delta_norm = 0.0

        rvy = image_delta_lateral_velocity(
            delta_norm,
            target_vy=0.0,
            kp=float(kp_img),
            kff=0.0,
            max_vy=max(amp / max(step_dt, 1.0e-3), 0.12),
            deadband=deadband,
        )
        base_sway[i] = robot_sway_m[i] + float(rvy) * float(step_dt)
        base_sway[i] = torch.clamp(base_sway[i], -amp * 1.05, amp * 1.05)

    base_surge = desired_surge_from_range(
        range_meas=range_meas,
        range_ref_gt_m=range_ref_gt_m,
        valid=range_valid,
        surge_robot_m=surge_robot_m,
        kp_range=kp_range,
        kp_surge_hold=kp_surge_hold,
        max_surge=max_surge,
        range_deadband_m=range_deadband_m,
        kp_range_fwd=kp_range_fwd,
        kp_range_back=kp_range_back,
        range_close_margin_m=range_close_margin_m,
        inhibit_surge_forward=inhibit_surge_forward,
    )
    return base_sway, base_surge
