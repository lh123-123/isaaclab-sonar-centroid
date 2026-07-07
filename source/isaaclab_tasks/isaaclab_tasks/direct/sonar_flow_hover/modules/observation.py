# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Policy observation assembly."""

from __future__ import annotations

import torch

from ..sparse_feature_tracker import SparseFeatureTracker
from .geometry import euler_xyz_from_quat


class ObservationBuilder:
    """Build 72-d policy observation and track depth for rate estimation."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        ang_vel_scale: float,
        depth_scale: float,
        depth_rate_scale: float,
        step_dt: float,
    ):
        self.num_envs = num_envs
        self.device = device
        self.ang_vel_scale = ang_vel_scale
        self.depth_scale = depth_scale
        self.depth_rate_scale = depth_rate_scale
        self.step_dt = max(step_dt, 1.0e-6)
        self._prev_depth = torch.zeros(num_envs, device=device)

    def reset_depth(self, env_ids: torch.Tensor, depth: torch.Tensor) -> None:
        self._prev_depth[env_ids] = depth

    def body_state(self, robot) -> torch.Tensor:
        quat = robot.data.root_quat_w
        roll, pitch, yaw = euler_xyz_from_quat(quat)
        ang_vel = robot.data.root_ang_vel_b
        depth = robot.data.root_pos_w[:, 2]
        depth_rate = (depth - self._prev_depth) / self.step_dt
        return torch.stack(
            [
                roll,
                pitch,
                yaw,
                ang_vel[:, 0] / self.ang_vel_scale,
                ang_vel[:, 1] / self.ang_vel_scale,
                ang_vel[:, 2] / self.ang_vel_scale,
                depth / self.depth_scale,
                depth_rate / self.depth_rate_scale,
            ],
            dim=-1,
        )

    def build(
        self,
        feature_tracker: SparseFeatureTracker,
        robot,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        feat_err, valid_flags = feature_tracker.observation_features()
        body = self.body_state(robot)
        return torch.cat([feat_err, valid_flags, body, actions], dim=-1)

    def commit_depth(self, robot) -> None:
        self._prev_depth = robot.data.root_pos_w[:, 2].clone()


class Lateral1DObservationBuilder:
    """Centroid offset obs: (track - teach) in sonar image pixels + valid + prev sway."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        max_track_points: int,
    ):
        self.num_envs = num_envs
        self.device = device
        self.max_track_points = max(int(max_track_points), 1)

    def build(
        self,
        feature_tracker: SparseFeatureTracker,
        robot,
        actions: torch.Tensor,
        *,
        centroid_err: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_ratio = feature_tracker.valid_count().float() / float(self.max_track_points)
        if centroid_err is None:
            centroid_err = feature_tracker.centroid_errors()

        prev_sway = actions[:, 0] if actions.shape[-1] == 1 else actions[:, 1]
        return torch.stack(
            [
                centroid_err[:, 0],
                centroid_err[:, 1],
                valid_ratio,
                prev_sway,
            ],
            dim=-1,
        )
