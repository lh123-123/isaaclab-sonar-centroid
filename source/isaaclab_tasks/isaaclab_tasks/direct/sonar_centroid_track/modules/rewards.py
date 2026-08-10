# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Module M8: reward from fixed-point tracking error."""

from __future__ import annotations

import torch

from .constants import (
    MIN_VALID_RATIO,
    REWARD_W_ACTION_SWITCH,
    REWARD_W_RANGE,
    REWARD_W_RESIDUAL,
    REWARD_W_SAFETY,
    REWARD_W_TOO_CLOSE,
    REWARD_W_U,
    REWARD_W_V,
    RANGE_ACCEPT_M,
)
from .sonar_centroid_tracker import SonarCentroidTracker


def _range_tracking_penalty(
    range_sonar_m: float,
    range_ref_m: float,
    *,
    accept_m: float,
    w_range: float,
    w_range_excess: float,
) -> float:
    """Piecewise symmetric range penalty; returns 0 if ref is NaN."""
    if range_ref_m != range_ref_m:
        return 0.0
    err = range_sonar_m - range_ref_m
    accept = max(float(accept_m), 1.0e-6)
    abs_err = abs(err)
    if abs_err > accept:
        excess = abs_err - accept
        return w_range * accept + w_range_excess * excess
    return w_range * abs_err


def compute_reward(
    tracker: SonarCentroidTracker,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    *,
    min_depth: float,
    max_depth: float,
    depth: torch.Tensor,
    w_u: float = REWARD_W_U,
    w_v: float = REWARD_W_V,
    w_switch: float = REWARD_W_ACTION_SWITCH,
    w_safety: float = REWARD_W_SAFETY,
    min_valid_ratio: float = MIN_VALID_RATIO,
    continuous_actions: bool = False,
) -> torch.Tensor:
    """Minimize ||p_sonar|| in fixed-point frame; optional action-switch penalty."""
    num_envs = actions.shape[0]
    device = actions.device
    reward = torch.zeros(num_envs, device=device)
    a = actions.view(-1)
    pa = prev_actions.view(-1)

    for i in range(num_envs):
        p = tracker.detected_rel(env_id=i)
        if p is not None:
            reward[i] -= w_u * abs(p.col) / max(float(tracker.polar_width), 1.0)
            reward[i] -= w_v * abs(p.row) / max(float(tracker.polar_height), 1.0)

        if continuous_actions:
            reward[i] -= w_switch * abs(a[i].item() - pa[i].item())
        elif int(a[i].item()) != int(pa[i].item()):
            reward[i] -= w_switch

        unsafe = tracker.valid_ratio(env_id=i) < min_valid_ratio
        unsafe = unsafe or bool((depth[i] < min_depth or depth[i] > max_depth).item())
        if unsafe:
            reward[i] -= w_safety

    return reward


def compute_auv_residual_reward(
    tracker: SonarCentroidTracker,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    *,
    range_ref_gt_m: torch.Tensor,
    min_depth: float,
    max_depth: float,
    depth: torch.Tensor,
    w_u: float = REWARD_W_U,
    w_v: float = REWARD_W_V,
    w_range: float = REWARD_W_RANGE,
    w_too_close: float = REWARD_W_TOO_CLOSE,
    w_residual: float = REWARD_W_RESIDUAL,
    w_lateral_action: float = 0.0,
    w_sway: float = 0.0,
    sway_err_m: torch.Tensor | None = None,
    sway_amp_m: float = 1.0,
    w_safety: float = REWARD_W_SAFETY,
    min_valid_ratio: float = MIN_VALID_RATIO,
    range_accept_m: float = RANGE_ACCEPT_M,
    sonar_range_meas_m: torch.Tensor | None = None,
    sonar_range_valid: torch.Tensor | None = None,
    range_actual_m: torch.Tensor | None = None,
    lost_track_steps: torch.Tensor | None = None,
    lost_track_safety_steps: int = 1,
    immediate_lost_safety: bool = False,
) -> torch.Tensor:
    """Image tracking + range hold vs setpoint.

    Range reward uses ``range_actual_m`` (live GT horizontal range) when provided,
    else falls back to ``sonar_range_meas_m``. Setpoint is ``range_ref_gt_m`` (e.g. hold_meas).

    Optional ``sway_err_m`` (train-only GT target−robot lateral, m) with ``w_sway``
    pulls the residual toward lateral tracking when image blend α is low.

    Range reward (piecewise, symmetric; independent of image centroid detection):
    - |err| <= accept: ``-w_range * |err|`` (encourage toward ref inside band)
    - |err| > accept: ``-w_range * accept - w_too_close * (|err| - accept)`` (large close/far penalty)

    Lost-track safety penalty aligns with termination grace:
    - ``immediate_lost_safety``: penalize every lost frame
    - else: penalize only when ``lost_track_steps >= lost_track_safety_steps``
    Depth out-of-range always incurs ``w_safety`` immediately.
    """

    num_envs = actions.shape[0]
    device = actions.device
    reward = torch.zeros(num_envs, device=device)
    a = actions.view(num_envs, -1)
    pa = prev_actions.view(num_envs, -1)
    amp = max(float(sway_amp_m), 1.0e-3)

    for i in range(num_envs):
        p = tracker.detected_rel(env_id=i)
        if p is not None:
            reward[i] -= w_u * abs(p.col) / max(float(tracker.polar_width), 1.0)
            reward[i] -= w_v * abs(p.row) / max(float(tracker.polar_height), 1.0)

        if w_sway > 0.0 and sway_err_m is not None:
            se = float(sway_err_m[i].item())
            if se == se:
                reward[i] -= w_sway * abs(se) / amp

        ref_m = float(range_ref_gt_m[i].item())
        if ref_m != ref_m:
            pass
        else:
            dist_m = float("nan")
            if range_actual_m is not None:
                dist_m = float(range_actual_m[i].item())
            elif (
                sonar_range_meas_m is not None
                and sonar_range_valid is not None
                and float(sonar_range_valid[i].item()) >= 0.5
            ):
                dist_m = float(sonar_range_meas_m[i].item())
            if dist_m == dist_m:
                reward[i] -= _range_tracking_penalty(
                    dist_m,
                    ref_m,
                    accept_m=range_accept_m,
                    w_range=w_range,
                    w_range_excess=w_too_close,
                )

        if a.shape[1] >= 2:
            reward[i] -= w_residual * torch.sum((a[i] - pa[i]) ** 2).item()
        elif a.shape[1] == 1:
            reward[i] -= w_residual * ((a[i, 0] - pa[i, 0]) ** 2).item()
        if a.shape[1] >= 2 and w_lateral_action > 0.0:
            reward[i] -= w_lateral_action * (a[i, 0].item() ** 2)

        lost = tracker.valid_ratio(env_id=i) < min_valid_ratio
        depth_bad = bool((depth[i] < min_depth or depth[i] > max_depth).item())
        if depth_bad:
            reward[i] -= w_safety
        elif lost:
            if immediate_lost_safety:
                reward[i] -= w_safety
            elif lost_track_steps is not None and int(lost_track_steps[i].item()) >= int(
                lost_track_safety_steps
            ):
                reward[i] -= w_safety

    return reward
