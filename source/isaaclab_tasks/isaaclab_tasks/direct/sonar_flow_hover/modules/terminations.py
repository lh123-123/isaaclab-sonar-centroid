# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Episode termination logic."""

from __future__ import annotations

import torch

from .constants import LOST_FEATURE_TERMINATE_STEPS
from .geometry import euler_xyz_from_quat, robot_lateral_sway_abs, station_pos_error_norm
from ..sparse_feature_tracker import SparseFeatureTracker


class TerminationChecker:
    """Track consecutive feature-loss steps and evaluate episode termination."""

    def __init__(self, num_envs: int, device: str | torch.device):
        self.num_envs = num_envs
        self.lost_feature_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.lost_feature_steps[env_ids] = 0

    def evaluate(
        self,
        *,
        cfg,
        feature_tracker: SparseFeatureTracker,
        robot,
        scene,
        station_origin_w: torch.Tensor,
        episode_length_buf: torch.Tensor,
        max_episode_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = episode_length_buf >= max_episode_length - 1

        if bool(getattr(cfg, "use_boundaries", True)):
            rel = robot.data.root_pos_w - scene.env_origins
            out_of_bounds = (
                (torch.abs(rel[:, 0]) > float(cfg.max_robot_x))
                | (torch.abs(rel[:, 1]) > float(cfg.max_robot_y))
                | (torch.abs(rel[:, 2] - float(cfg.workspace_height)) > float(cfg.max_robot_z))
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=robot.device)

        lost_features = feature_tracker.valid_count() < int(cfg.min_valid_features)
        self.lost_feature_steps[lost_features] += 1
        self.lost_feature_steps[~lost_features] = 0
        lost_terminate = self.lost_feature_steps >= int(
            getattr(cfg, "lost_feature_terminate_steps", LOST_FEATURE_TERMINATE_STEPS)
        )
        if bool(getattr(cfg, "debug_disable_lost_terminate", False)):
            lost_terminate = torch.zeros_like(lost_terminate)

        drift_limit = float(max(cfg.spawn_radius * 2.5, 3.0))
        if bool(getattr(cfg, "lateral_1d_mode", False)):
            _, _, yaw_w = euler_xyz_from_quat(robot.data.root_quat_w)
            drift_terminate = (
                robot_lateral_sway_abs(station_origin_w, robot.data.root_pos_w, yaw_w) > drift_limit
            )
        else:
            drift_terminate = station_pos_error_norm(station_origin_w, robot.data.root_pos_w) > drift_limit

        depth = robot.data.root_pos_w[:, 2]
        depth_bad = (depth < float(cfg.min_depth)) | (depth > float(cfg.max_depth))

        terminated = out_of_bounds | lost_terminate | drift_terminate | depth_bad
        return terminated, time_out
