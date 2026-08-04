"""Batch PID velocity controller for WarpAUV (HoloOcean v0-style).

RL action (after scale): cmd_vel = [v_surge, omega_yaw]
→ surge (+ light yaw damping) → main drives (0,1)
→ depth/attitude → angled thrusters (2..5)

Empirically: equal drive PWM tracks body +vx; heave must not share drive channels.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate

from isaaclab_tasks.direct.residual_rl.thruster_allocation import (
    allocate_thrusters,
    build_thruster_allocation_matrix,
)


class BatchPIDController:
    """PID [v, ω] + depth hold → 6 thruster PWM, batched on GPU."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dt: float = 1.0 / 120.0,
        # Calibrated so cmd≈1 m/s settles near ~0.8–1.0 body vx (see debug_auv_pid_surge).
        kp_lin: float = 2.8,
        ki_lin: float = 0.4,
        kd_lin: float = 0.08,
        yaw_damp: float = 0.2,
        kp_ang: float = 0.1,
        # Depth / attitude (angled thrusters only)
        kp_depth: float = 5.0,
        kd_depth: float = 3.0,
        kp_att: float = 1.5,
        kd_att: float = 0.4,
        max_heave_wrench: float = 3.0,
        depth_deadband: float = 0.06,
    ):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.kp_lin = kp_lin
        self.ki_lin = ki_lin
        self.kd_lin = kd_lin
        self.yaw_damp = yaw_damp
        self.kp_ang = kp_ang
        self.kp_depth = kp_depth
        self.kd_depth = kd_depth
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.max_heave_wrench = max_heave_wrench
        self.depth_deadband = depth_deadband

        _, self._b_pinv = build_thruster_allocation_matrix(device)

        self.lin_error_sum = torch.zeros(num_envs, device=device)
        self.lin_last_error = torch.zeros(num_envs, device=device)
        self.depth_target = torch.zeros(num_envs, device=device)
        self.lin_int_limit = 2.0

    def reset(self, env_ids: torch.Tensor, depth_target: torch.Tensor | float | None = None):
        self.lin_error_sum[env_ids] = 0.0
        self.lin_last_error[env_ids] = 0.0
        if depth_target is not None:
            if isinstance(depth_target, torch.Tensor):
                self.depth_target[env_ids] = depth_target
            else:
                self.depth_target[env_ids] = float(depth_target)

    def compute(
        self,
        cmd_vel: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Map cmd_vel (N,2)=[v,ω] and robot state to PWM (N,6)."""
        dt = self.dt if dt is None else dt
        quat_b = quat_conjugate(root_quat_w)
        lin_vel_b = quat_apply(quat_b, root_lin_vel_w)
        ang_vel_b = quat_apply(quat_b, root_ang_vel_w)

        target_vx = cmd_vel[:, 0]
        target_omega = cmd_vel[:, 1]
        curr_vx = lin_vel_b[:, 0]
        curr_omega = ang_vel_b[:, 2]

        # Surge PID (output is drive PWM command)
        lin_error = target_vx - curr_vx
        self.lin_error_sum = torch.clip(
            self.lin_error_sum + lin_error * dt, -self.lin_int_limit, self.lin_int_limit
        )
        lin_deriv = (lin_error - self.lin_last_error) / max(dt, 1e-6)
        self.lin_last_error = lin_error
        u_surge = self.kp_lin * lin_error + self.ki_lin * self.lin_error_sum + self.kd_lin * lin_deriv

        # Mild yaw: track ω cmd + damp rate (clamped so surge stays dominant)
        u_yaw = self.kp_ang * (target_omega - curr_omega) - self.yaw_damp * curr_omega
        u_yaw = torch.clamp(u_yaw, -0.25, 0.25)

        pwm = torch.zeros(self.num_envs, 6, device=self.device)
        # Empirically both drives same-sign produces body surge; differential for yaw.
        pwm[:, 0] = torch.clamp(u_surge - u_yaw, -1.0, 1.0)
        pwm[:, 1] = torch.clamp(u_surge + u_yaw, -1.0, 1.0)

        # Depth / attitude → angled thrusters only
        depth_error = self.depth_target - root_pos_w[:, 2]
        depth_error = torch.where(
            depth_error.abs() < self.depth_deadband, torch.zeros_like(depth_error), depth_error
        )
        fz = self.kp_depth * depth_error - self.kd_depth * lin_vel_b[:, 2]
        mx = -self.kp_att * root_quat_w[:, 1] - self.kd_att * ang_vel_b[:, 0]
        my = -self.kp_att * root_quat_w[:, 2] - self.kd_att * ang_vel_b[:, 1]

        wrench = torch.zeros(self.num_envs, 6, device=self.device)
        wrench[:, 2] = torch.clamp(fz, -self.max_heave_wrench, self.max_heave_wrench)
        wrench[:, 3] = torch.clamp(mx, -self.max_heave_wrench, self.max_heave_wrench)
        wrench[:, 4] = torch.clamp(my, -self.max_heave_wrench, self.max_heave_wrench)
        u_vert = allocate_thrusters(wrench, self._b_pinv, clamp=1.0, priority=False)
        pwm[:, 2:6] = u_vert[:, 2:6]
        return torch.clamp(pwm, -1.0, 1.0)
