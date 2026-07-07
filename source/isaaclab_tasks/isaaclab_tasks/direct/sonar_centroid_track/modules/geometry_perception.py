# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Module M5: geometry-projected centroid (p_perc) — requires sim TF at runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import beacon_bearing_range, beacon_polar_rois, sonar_pos_w

from .coordinates import PolarPoint, polar_point
from .fixed_point import FixedPointStore

if TYPE_CHECKING:
    pass


class GeometryPerception:
    """Project beacon world position into polar ROI center (p_perc_abs)."""

    def __init__(self, fixed_point: FixedPointStore):
        self.fixed_point = fixed_point

    def project_abs(
        self,
        *,
        beacon_pos_w: torch.Tensor,
        robot_pos_w: torch.Tensor,
        robot_quat_w: torch.Tensor,
        sonar_mount_offset_b: torch.Tensor,
        env_ids: torch.Tensor,
        min_range: float,
        max_range: float,
        polar_height: int,
        polar_width: int,
        hori_fov_deg: float,
    ) -> list[PolarPoint | None]:
        sonar_pw = sonar_pos_w(robot_pos_w, robot_quat_w, sonar_mount_offset_b)
        bearing, horiz_range = beacon_bearing_range(beacon_pos_w, sonar_pw, robot_quat_w)
        rois = beacon_polar_rois(
            bearing,
            horiz_range,
            env_ids,
            min_range=min_range,
            max_range=max_range,
            polar_height=polar_height,
            polar_width=polar_width,
            hori_fov_deg=hori_fov_deg,
        )
        out: list[PolarPoint | None] = []
        for roi in rois:
            if roi is None:
                out.append(None)
            else:
                col, row = roi
                out.append(polar_point(float(col), float(row)))
        return out

    def project_rel(self, p_perc_abs: PolarPoint | None, env_id: int = 0) -> PolarPoint | None:
        if p_perc_abs is None:
            return None
        return self.fixed_point.to_relative(p_perc_abs, env_id)
