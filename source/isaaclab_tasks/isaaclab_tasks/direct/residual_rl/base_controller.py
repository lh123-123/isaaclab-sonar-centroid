# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Imperfect PD base controller for residual RL.

Computes body-frame wrench from pose error, then maps to 6 thruster PWM via the
WarpAUV thruster allocation matrix (pseudoinverse).
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply, quat_conjugate

from .thruster_allocation import allocate_thrusters, build_thruster_allocation_matrix


class BaseController:
    """Body-frame PD + thruster allocation (6-DOF wrench -> 6 PWM)."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        imperfect: bool = True,
        kp: float | None = None,
        kd: float | None = None,
        kp_ang: float | None = None,
        kd_ang: float | None = None,
        thrust_bias: float | None = None,
        max_wrench: float | None = None,
    ):
        self.num_envs = num_envs
        self.device = device

        if imperfect:
            self.kp = kp if kp is not None else 0.3
            self.kd = kd if kd is not None else 0.01
            self.kp_ang = kp_ang if kp_ang is not None else 0.2
            self.kd_ang = kd_ang if kd_ang is not None else 0.01
            self.thrust_bias = thrust_bias if thrust_bias is not None else 0.05
            self.max_wrench = max_wrench if max_wrench is not None else 2.0
        else:
            self.kp = kp if kp is not None else 1.0
            self.kd = kd if kd is not None else 0.5
            self.kp_ang = kp_ang if kp_ang is not None else 0.8
            self.kd_ang = kd_ang if kd_ang is not None else 0.3
            self.thrust_bias = thrust_bias if thrust_bias is not None else 0.0
            self.max_wrench = max_wrench if max_wrench is not None else 5.0

        self._b_mat, self._b_pinv = build_thruster_allocation_matrix(device)

    def compute_wrench(
        self,
        target_pos_w: torch.Tensor,
        current_pos_w: torch.Tensor,
        lin_vel_b: torch.Tensor,
        target_quat_w: torch.Tensor,
        current_quat_w: torch.Tensor,
        ang_vel_b: torch.Tensor,
    ) -> torch.Tensor:
        """Body-frame PD wrench [Fx, Fy, Fz, Mx, My, Mz] before allocation."""
        pos_error_w = target_pos_w - current_pos_w
        pos_error_b = quat_apply(quat_conjugate(current_quat_w), pos_error_w)

        quat_err = math_utils.quat_mul(target_quat_w, math_utils.quat_conjugate(current_quat_w))
        quat_err = math_utils.quat_unique(quat_err)
        ang_error_b = quat_err[:, 1:4] * torch.sign(quat_err[:, 0:1]).clamp(min=0.0, max=1.0) * 2.0

        wrench = torch.zeros(self.num_envs, 6, device=self.device)
        wrench[:, 0:3] = self.kp * pos_error_b - self.kd * lin_vel_b
        wrench[:, 3:6] = self.kp_ang * ang_error_b - self.kd_ang * ang_vel_b

        if self.thrust_bias != 0.0:
            wrench[:, 0] = wrench[:, 0] + self.thrust_bias

        return torch.clamp(wrench, -self.max_wrench, self.max_wrench)

    def allocate_thrusters_from_wrench(
        self,
        wrench: torch.Tensor,
        thruster_gain: torch.Tensor | None = None,
        thruster_bias: torch.Tensor | None = None,
        *,
        priority: bool = True,
        priority_attitude: bool = True,
    ) -> torch.Tensor:
        """Map body wrench to bounded thruster PWM (heave prioritized; attitude optional)."""
        u = allocate_thrusters(
            wrench, self._b_pinv, priority=priority, priority_attitude=priority_attitude
        )
        if thruster_gain is not None:
            u = u * thruster_gain
        if thruster_bias is not None:
            u = u + thruster_bias
        return torch.clamp(u, -1.0, 1.0)

    def compute(
        self,
        target_pos_w: torch.Tensor,
        current_pos_w: torch.Tensor,
        lin_vel_b: torch.Tensor,
        target_quat_w: torch.Tensor,
        current_quat_w: torch.Tensor,
        ang_vel_b: torch.Tensor,
        thruster_gain: torch.Tensor | None = None,
        thruster_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute base thruster PWM from pose tracking error."""
        wrench = self.compute_wrench(
            target_pos_w, current_pos_w, lin_vel_b, target_quat_w, current_quat_w, ang_vel_b
        )
        return self.allocate_thrusters_from_wrench(wrench, thruster_gain, thruster_bias)
