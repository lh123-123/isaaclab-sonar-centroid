# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify PPO 4D obs matches Phase-2 green/yellow overlay convention."""

from __future__ import annotations

from typing import Any

from .constants import ACTION_LEFT, ACTION_RIGHT, ACTION_STAY
from .coordinates import normalize_uv
from .sonar_centroid_tracker import SonarCentroidTracker


def prev_action_to_obs(prev_action: int | float, *, continuous: bool = False) -> float:
    if continuous:
        return float(max(-1.0, min(1.0, float(prev_action))))
    if int(prev_action) == ACTION_LEFT:
        return -1.0
    if int(prev_action) == ACTION_RIGHT:
        return 1.0
    return 0.0


def obs_action_name(prev_obs: float, *, continuous: bool = False) -> str:
    if continuous:
        return f"vy_norm({prev_obs:+.3f})"
    if prev_obs < -0.5:
        return "left(0)"
    if prev_obs > 0.5:
        return "right(2)"
    return "stay(1)"


def correspondence_report(
    tracker: SonarCentroidTracker,
    *,
    env_id: int = 0,
    prev_action: int | float = ACTION_STAY,
    polar_width: int | None = None,
    polar_height: int | None = None,
    u_tol: float = 1.0e-5,
    continuous_actions: bool = False,
) -> dict[str, Any]:
    """Cross-check policy obs against Phase-2 ``d_col = green_col − yellow_col``.

    Mapping (image-only 4D):
        u_norm       = (green_col − yellow_col) / W
        v_norm       = (green_row − yellow_row) / H
        valid_ratio  = 1 if green detected else 0
        prev_act     = {−1,0,+1} for discrete {left,stay,right}; float in [−1,1] for continuous vy_norm
    """
    w = int(polar_width if polar_width is not None else tracker.polar_width)
    h = int(polar_height if polar_height is not None else tracker.polar_height)

    p_green = tracker.detected_abs(env_id)
    p_yellow = tracker.fixed_point.p_fixed_abs_env(env_id)
    p_rel = tracker.detected_rel(env_id)
    valid = float(tracker.valid_ratio(env_id))

    green_col = float(p_green.col) if p_green is not None else None
    green_row = float(p_green.row) if p_green is not None else None
    yellow_col = float(p_yellow.col) if p_yellow is not None else None
    yellow_row = float(p_yellow.row) if p_yellow is not None else None

    d_col_px = None
    d_row_px = None
    if p_green is not None and p_yellow is not None:
        d_col_px = green_col - yellow_col
        d_row_px = green_row - yellow_row

    u_norm = v_norm = 0.0
    if p_rel is not None:
        u_norm, v_norm = normalize_uv(p_rel, w, h)

    prev_obs = prev_action_to_obs(prev_action, continuous=continuous_actions)
    u_from_dcol = (d_col_px / w) if d_col_px is not None else None
    v_from_drow = (d_row_px / h) if d_row_px is not None else None

    if valid < 0.5:
        u_match = abs(u_norm) <= u_tol
        v_match = abs(v_norm) <= u_tol
    else:
        u_match = u_from_dcol is not None and abs(u_norm - u_from_dcol) <= u_tol
        v_match = v_from_drow is not None and abs(v_norm - v_from_drow) <= u_tol

    if valid < 0.5:
        control_hint = "lost — terminate / no col signal"
    elif continuous_actions:
        vy_hint = prev_obs * 0.30
        if u_norm > 0.0:
            control_hint = f"u>0 → increase vy (P suggests +); prev_vy≈{vy_hint:+.3f} m/s"
        elif u_norm < 0.0:
            control_hint = f"u<0 → decrease vy (P suggests −); prev_vy≈{vy_hint:+.3f} m/s"
        else:
            control_hint = f"u≈0 aligned; prev_vy≈{vy_hint:+.3f} m/s"
    elif u_norm > 0.0:
        control_hint = "u>0 green RIGHT of yellow → action=right(2)"
    elif u_norm < 0.0:
        control_hint = "u<0 green LEFT of yellow → action=left(0)"
    else:
        control_hint = "u≈0 aligned → action=stay(1)"

    return {
        "green_col": green_col,
        "green_row": green_row,
        "yellow_col": yellow_col,
        "yellow_row": yellow_row,
        "d_col_px": d_col_px,
        "d_row_px": d_row_px,
        "W": w,
        "H": h,
        "u_norm": u_norm,
        "v_norm": v_norm,
        "valid_ratio": valid,
        "prev_act": prev_obs,
        "prev_action": obs_action_name(prev_obs, continuous=continuous_actions),
        "u_norm_from_d_col": u_from_dcol,
        "v_norm_from_d_row": v_from_drow,
        "u_match": u_match,
        "v_match": v_match,
        "control_hint": control_hint,
        "policy_obs_4d": [u_norm, v_norm, valid, prev_obs],
    }


def format_correspondence_line(rep: dict[str, Any], *, step: int | None = None) -> str:
    prefix = f"step={step:4d} " if step is not None else ""
    dc = rep["d_col_px"]
    dr = rep["d_row_px"]
    dc_s = f"{dc:+.2f}" if dc is not None else "  nan"
    dr_s = f"{dr:+.2f}" if dr is not None else "  nan"
    ok = "OK" if rep["u_match"] and rep["v_match"] else "MISMATCH"
    return (
        f"{prefix}"
        f"d_col={dc_s}px d_row={dr_s}px  "
        f"u={rep['u_norm']:+.6f} v={rep['v_norm']:+.6f} valid={rep['valid_ratio']:.0f} "
        f"prev={rep['prev_action']:>8s}  [{ok}]  {rep['control_hint']}"
    )
