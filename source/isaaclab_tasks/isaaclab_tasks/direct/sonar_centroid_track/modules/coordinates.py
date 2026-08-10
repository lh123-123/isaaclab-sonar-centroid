# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Polar pixel coordinates: absolute frame and fixed-point origin frame."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PolarPoint:
    """Polar image pixel (col=azimuth, row=range)."""

    col: float
    row: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.col, self.row)

    def norm(self) -> float:
        return math.hypot(self.col, self.row)

    def __sub__(self, other: PolarPoint) -> PolarPoint:
        return PolarPoint(self.col - other.col, self.row - other.row)


def polar_point(col: float, row: float) -> PolarPoint:
    return PolarPoint(float(col), float(row))


def mirror_azimuth_col(col: float, width: int) -> float:
    """Map OceanSim column index to geometry-aligned azimuth column."""
    return float(width - 1) - float(col)


def to_fixed_frame(p_abs: PolarPoint | Sequence[float], p_fixed_abs: PolarPoint | Sequence[float]) -> PolarPoint:
    """Absolute polar pixel → fixed-point frame (origin = teach alignment)."""
    if not isinstance(p_abs, PolarPoint):
        p_abs = polar_point(p_abs[0], p_abs[1])
    if not isinstance(p_fixed_abs, PolarPoint):
        p_fixed_abs = polar_point(p_fixed_abs[0], p_fixed_abs[1])
    return p_abs - p_fixed_abs


def to_absolute_frame(p_rel: PolarPoint | Sequence[float], p_fixed_abs: PolarPoint | Sequence[float]) -> PolarPoint:
    if not isinstance(p_rel, PolarPoint):
        p_rel = polar_point(p_rel[0], p_rel[1])
    if not isinstance(p_fixed_abs, PolarPoint):
        p_fixed_abs = polar_point(p_fixed_abs[0], p_fixed_abs[1])
    return PolarPoint(p_rel.col + p_fixed_abs.col, p_rel.row + p_fixed_abs.row)


def polar_row_to_range_m(
    row: float,
    *,
    min_range: float,
    max_range: float,
    polar_height: int,
) -> float:
    """Polar row index → horizontal range (m), matching ``beacon_polar_rois``."""
    h_max = max(float(polar_height) - 1, 1.0)
    span = max(float(max_range) - float(min_range), 1.0e-6)
    return float(min_range) + float(row) / h_max * span


def polar_range_m_to_row(
    range_m: float,
    *,
    min_range: float,
    max_range: float,
    polar_height: int,
) -> float:
    h_max = max(float(polar_height) - 1, 1.0)
    span = max(float(max_range) - float(min_range), 1.0e-6)
    row = (float(range_m) - float(min_range)) / span * h_max
    return max(0.0, min(h_max, row))


def polar_col_to_azimuth_rad(col: float, width: int, hori_fov_rad: float) -> float:
    """Map polar image column index to beam azimuth (rad), matching OceanSim linspace."""
    w_max = max(float(width) - 1, 1.0)
    return -float(hori_fov_rad) / 2.0 + float(col) / w_max * float(hori_fov_rad)


def gt_sonar_beacon_horiz_range_m(
    beacon_pos_w: torch.Tensor,
    robot_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    sonar_mount_offset_b: torch.Tensor,
    env_id: int = 0,
) -> float:
    """GT horizontal range (m) from sonar mount to beacon in robot body XY."""
    return float(
        gt_sonar_beacon_horiz_range_batch(
            beacon_pos_w, robot_pos_w, robot_quat_w, sonar_mount_offset_b
        )[int(env_id)].item()
    )


def gt_sonar_beacon_horiz_range_batch(
    beacon_pos_w: torch.Tensor,
    robot_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    sonar_mount_offset_b: torch.Tensor,
) -> torch.Tensor:
    """Per-env GT horizontal range (m), sonar mount → beacon in body XY."""
    from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import beacon_bearing_range, sonar_pos_w

    sonar_pw = sonar_pos_w(robot_pos_w, robot_quat_w, sonar_mount_offset_b)
    _, horiz_range = beacon_bearing_range(beacon_pos_w, sonar_pw, robot_quat_w)
    return horiz_range


def gt_sonar_beacon_slant_range_m(
    beacon_pos_w: torch.Tensor,
    robot_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    sonar_mount_offset_b: torch.Tensor,
    env_id: int = 0,
) -> float:
    """GT 3D slant range (m) from sonar mount to beacon."""
    import torch
    from isaaclab.utils.math import quat_apply_inverse
    from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import sonar_pos_w

    sonar_pw = sonar_pos_w(robot_pos_w, robot_quat_w, sonar_mount_offset_b)
    rel_w = beacon_pos_w - sonar_pw
    rel_b = quat_apply_inverse(robot_quat_w, rel_w)
    return float(torch.norm(rel_b[int(env_id)]).item())


def green_sonar_range_m(
    green_row: float,
    *,
    min_range: float,
    max_range: float,
    polar_height: int,
) -> float:
    """Sonar-image green point row → horizontal range (m)."""
    return polar_row_to_range_m(
        green_row, min_range=min_range, max_range=max_range, polar_height=polar_height
    )


def range_err_norm(
    range_sonar_m: float,
    range_gt_m: float,
    *,
    accept_m: float = 0.5,
) -> float:
    """Normalized range error: ±accept_m → ±1; inside ±accept_m is acceptable."""
    scale = max(float(accept_m), 1.0e-6)
    return float(max(-1.0, min(1.0, (float(range_sonar_m) - float(range_gt_m)) / scale)))


def normalize_uv(p_rel: PolarPoint, width: int, height: int) -> tuple[float, float]:
    """Fixed-point coords normalized for RL obs."""
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    return p_rel.col / w, p_rel.row / h
