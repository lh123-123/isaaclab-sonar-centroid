# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Episode termination for centroid tracking."""

from __future__ import annotations

import torch

from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import euler_xyz_from_quat, robot_lateral_sway_abs

from .sonar_centroid_tracker import SonarCentroidTracker


class CentroidTerminationChecker:
    def __init__(self, num_envs: int, device: str | torch.device):
        self.num_envs = num_envs
        self.lost_track_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.lost_track_steps[env_ids] = 0

    def evaluate(
        self,
        *,
        cfg,
        tracker: SonarCentroidTracker,
        robot,
        scene,
        station_origin_w: torch.Tensor,
        episode_length_buf: torch.Tensor,
        max_episode_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = robot.device
        time_out = episode_length_buf >= max_episode_length - 1

        if bool(getattr(cfg, "use_boundaries", True)):
            rel = robot.data.root_pos_w - scene.env_origins
            out_of_bounds = (
                (torch.abs(rel[:, 0]) > float(cfg.max_robot_x))
                | (torch.abs(rel[:, 1]) > float(cfg.max_robot_y))
                | (torch.abs(rel[:, 2] - float(cfg.workspace_height)) > float(cfg.max_robot_z))
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        min_valid = float(getattr(cfg, "min_valid_ratio", 0.2))
        lost = torch.tensor(
            [tracker.valid_ratio(i) < min_valid for i in range(self.num_envs)],
            device=device,
            dtype=torch.bool,
        )
        immediate_lost = bool(getattr(cfg, "immediate_lost_terminate", False))
        lost_steps_limit = int(getattr(cfg, "lost_track_terminate_steps", 80))
        if immediate_lost:
            lost_terminate = lost
        else:
            self.lost_track_steps[lost] += 1
            self.lost_track_steps[~lost] = 0
            lost_terminate = self.lost_track_steps >= lost_steps_limit
        if bool(getattr(cfg, "debug_disable_lost_terminate", False)):
            lost_terminate = torch.zeros_like(lost_terminate)

        drift_limit = float(max(getattr(cfg, "spawn_radius", 0.15) * 2.5, 3.0))
        _, _, yaw_w = euler_xyz_from_quat(robot.data.root_quat_w)
        drift_terminate = robot_lateral_sway_abs(station_origin_w, robot.data.root_pos_w, yaw_w) > drift_limit

        depth = robot.data.root_pos_w[:, 2]
        depth_bad = (depth < float(cfg.min_depth)) | (depth > float(cfg.max_depth))

        terminated = out_of_bounds | lost_terminate | drift_terminate | depth_bad
        return terminated, time_out
