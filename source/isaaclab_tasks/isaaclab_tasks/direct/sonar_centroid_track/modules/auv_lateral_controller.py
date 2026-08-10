# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""WarpAUV lateral position tracking via BaseController + thruster allocation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_tasks.direct.residual_rl.base_controller import BaseController

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from ..sonar_centroid_track_auv_env_cfg import SonarCentroidTrackAUVEnvCfg


class AUVLateralController:
    """Track a world-frame lateral position (sway) with 6 PWM + depth/yaw hold."""

    def __init__(self, cfg: SonarCentroidTrackAUVEnvCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device
        self._base = BaseController(
            num_envs,
            device,
            imperfect=False,
            kp=float(getattr(cfg, "auv_pos_kp", 2.5)),
            kd=float(getattr(cfg, "auv_pos_kd", 1.2)),
            kp_ang=float(getattr(cfg, "auv_kp_yaw", 45.0)),
            kd_ang=float(getattr(cfg, "auv_kd_yaw", 8.0)),
            thrust_bias=0.0,
            max_wrench=float(getattr(cfg, "auv_max_wrench", 80.0)),
        )

    def compute_pwm(
        self,
        desired_sway_m: torch.Tensor,
        robot: RigidObject,
        *,
        station_origin_w: torch.Tensor,
        fixed_yaw_w: torch.Tensor,
        target_vy: torch.Tensor | None = None,
        desired_surge_m: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``desired_sway_m`` / ``desired_surge_m`` (body frame, m) -> bounded PWM ``(N, 6)``."""
        n = self.num_envs
        sway = desired_sway_m.view(n).float()
        if desired_surge_m is None:
            surge = torch.zeros(n, device=self.device)
        else:
            surge = desired_surge_m.view(n).float()
        yaw = fixed_yaw_w.view(n).float()
        zeros = torch.zeros(n, device=self.device)

        forward_w = torch.stack([torch.cos(yaw), torch.sin(yaw), zeros], dim=-1)
        lateral_w = torch.stack([-torch.sin(yaw), torch.cos(yaw), zeros], dim=-1)
        target_pos_w = station_origin_w.clone()
        target_pos_w += forward_w * surge.unsqueeze(-1)
        target_pos_w += lateral_w * sway.unsqueeze(-1)

        target_quat = quat_from_euler_xyz(zeros, zeros, yaw)
        wrench = self._base.compute_wrench(
            target_pos_w,
            robot.data.root_pos_w,
            robot.data.root_lin_vel_b,
            target_quat,
            robot.data.root_quat_w,
            robot.data.root_ang_vel_b,
        )
        fmax = float(getattr(self.cfg, "auv_max_force", 250.0))
        mmax = float(getattr(self.cfg, "auv_max_torque", 50.0))
        wrench[:, 0:3] = torch.clamp(wrench[:, 0:3], -fmax, fmax)
        wrench[:, 3:6] = torch.clamp(wrench[:, 3:6], -mmax, mmax)
        if target_vy is not None:
            vff = float(getattr(self.cfg, "auv_vy_ff", 35.0))
            mass = float(getattr(self.cfg, "auv_mass", 22.701))
            vy_b = robot.data.root_lin_vel_b[:, 1]
            tgt_v = target_vy.view(n).float()
            wrench[:, 1] = wrench[:, 1] + vff * mass * (tgt_v - vy_b)
        return self._base.allocate_thrusters_from_wrench(wrench)
