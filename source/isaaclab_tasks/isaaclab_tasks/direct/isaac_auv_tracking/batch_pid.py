"""Batch PID velocity controller for WarpAUV (HoloOcean v0-style).

RL action (after scale): cmd_vel = [v_surge, omega_yaw]
→ surge (+ light yaw damping) → main drives (0,1)
→ depth/attitude → angled thrusters (2..5)

Empirically: equal drive PWM tracks body +vx; heave must not share drive channels.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate

from isaaclab_tasks.direct.residual_rl.thruster_allocation import (
    allocate_thrusters,
    build_allocation_from_geometry,
    build_thruster_allocation_matrix,
)


class BatchPIDController:
    """PID [v, ω] + depth hold → N thruster commands, batched on GPU.

    Two allocation layouts are supported:

    * ``"warpauv6"`` (default): the calibrated WarpAUV mapping -- surge/yaw go to the two
      main drives by hand, depth/attitude are allocated onto the four angled thrusters.
    * ``"gsrov8"``: the GS-ROV hull. Surge and yaw become a body wrench allocated across
      the four horizontal thrusters, depth/attitude across the four vertical ones. The two
      groups are physically decoupled, so a plain pseudoinverse suffices.
    """

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
        ki_depth: float = 0.0,
        kd_depth: float = 3.0,
        depth_int_limit: float = 4.0,
        kp_att: float = 1.5,
        kd_att: float = 0.4,
        max_heave_wrench: float = 3.0,
        depth_deadband: float = 0.06,
        thruster_layout: str = "warpauv6",
        thruster_offsets: torch.Tensor | None = None,
        thruster_quats: torch.Tensor | None = None,
        yaw_pwm_cap: float = 0.25,
        trim_roll_rad: float = 0.0,
        trim_pitch_rad: float = 0.0,
        att_ki: float = 0.0,
        att_int_limit: float = 8.0,
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
        self.ki_depth = ki_depth
        self.kd_depth = kd_depth
        self.depth_int_limit = depth_int_limit
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.max_heave_wrench = max_heave_wrench
        self.depth_deadband = depth_deadband
        self.yaw_pwm_cap = yaw_pwm_cap
        # Target trim attitude (rad). A free-floating hull settles at a nose-up pitch set by
        # its COM-COB offset (~20 deg for the GS-ROV); the attitude loop should command that
        # trim instead of level, otherwise it fights buoyancy forever and burns thrust.
        self.trim_roll_rad = float(trim_roll_rad)
        self.trim_pitch_rad = float(trim_pitch_rad)
        # Attitude integral (off by default — legacy tasks keep the P+D law). Holding an
        # attitude away from the free-floating trim needs a CONSTANT moment (buoyancy
        # couple ~5 N·m for the GS-ROV at level); a P-only loop has no way to supply it
        # without standing error, so add integral action for that case. The env pre-charges
        # att_error_sum at reset so frame 0 already carries the leveling moment.
        self.att_ki = float(att_ki)
        self.att_int_limit = float(att_int_limit)
        self.thruster_layout = str(thruster_layout).lower()

        if self.thruster_layout == "gsrov8":
            if thruster_offsets is None or thruster_quats is None:
                raise ValueError("gsrov8 layout needs thruster_offsets and thruster_quats")
            self.num_thrusters = int(thruster_offsets.shape[0])
            b_mat, self._b_pinv = build_allocation_from_geometry(thruster_offsets, thruster_quats)
            # Scales so |u|==1 saturates the horizontal (surge/sway/yaw) group.
            horizontal = (3, 4, 5, 6)
            self._surge_scale = float(b_mat[0, list(horizontal)].abs().sum().item())
            self._sway_scale = float(b_mat[1, list(horizontal)].abs().sum().item())
            self._yaw_scale = float(b_mat[5, list(horizontal)].abs().sum().item())
        else:
            self.num_thrusters = 6
            self._b_pinv = build_thruster_allocation_matrix(device)[1]
            self._surge_scale = 1.0
            self._sway_scale = 1.0
            self._yaw_scale = 1.0

        self.lin_error_sum = torch.zeros(num_envs, device=device)
        self.lin_last_error = torch.zeros(num_envs, device=device)
        self.sway_error_sum = torch.zeros(num_envs, device=device)
        self.sway_last_error = torch.zeros(num_envs, device=device)
        self.depth_target = torch.zeros(num_envs, device=device)
        self.depth_error_sum = torch.zeros(num_envs, device=device)
        self.att_error_sum = torch.zeros(num_envs, 2, device=device)
        self.lin_int_limit = 2.0

    def reset(self, env_ids: torch.Tensor, depth_target: torch.Tensor | float | None = None):
        self.lin_error_sum[env_ids] = 0.0
        self.lin_last_error[env_ids] = 0.0
        self.sway_error_sum[env_ids] = 0.0
        self.sway_last_error[env_ids] = 0.0
        self.depth_error_sum[env_ids] = 0.0
        self.att_error_sum[env_ids] = 0.0
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
        if self.thruster_layout == "gsrov8" and cmd_vel.shape[-1] >= 3:
            target_vy = cmd_vel[:, 1]
            target_omega = cmd_vel[:, 2]
        else:
            target_vy = torch.zeros(self.num_envs, device=self.device)
            target_omega = cmd_vel[:, 1]
        curr_vx = lin_vel_b[:, 0]
        curr_vy = lin_vel_b[:, 1]
        curr_omega = ang_vel_b[:, 2]

        # Surge PID (output is drive PWM command)
        lin_error = target_vx - curr_vx
        self.lin_error_sum = torch.clip(
            self.lin_error_sum + lin_error * dt, -self.lin_int_limit, self.lin_int_limit
        )
        lin_deriv = (lin_error - self.lin_last_error) / max(dt, 1e-6)
        self.lin_last_error = lin_error
        u_surge = self.kp_lin * lin_error + self.ki_lin * self.lin_error_sum + self.kd_lin * lin_deriv

        # Sway PID (only meaningful on gsrov8; warpauv6 has no lateral channel)
        sway_error = target_vy - curr_vy
        self.sway_error_sum = torch.clip(
            self.sway_error_sum + sway_error * dt, -self.lin_int_limit, self.lin_int_limit
        )
        sway_deriv = (sway_error - self.sway_last_error) / max(dt, 1e-6)
        self.sway_last_error = sway_error
        u_sway = self.kp_lin * sway_error + self.ki_lin * self.sway_error_sum + self.kd_lin * sway_deriv

        # Mild yaw: track ω cmd + damp rate (clamped so surge stays dominant)
        u_yaw = self.kp_ang * (target_omega - curr_omega) - self.yaw_damp * curr_omega
        u_yaw = torch.clamp(u_yaw, -self.yaw_pwm_cap, self.yaw_pwm_cap)

        # Depth / attitude demand, shared by both layouts.
        # Integrate the raw error so a constant negative-buoyancy bias (~2 N on the
        # GS-ROV) is rejected. The deadband applies to the P term only; ki_depth
        # defaults to 0, which leaves every existing task on the old P+D law (and
        # skips the integration entirely).
        depth_error_raw = self.depth_target - root_pos_w[:, 2]
        if self.ki_depth != 0.0:
            self.depth_error_sum = torch.clip(
                self.depth_error_sum + depth_error_raw * dt,
                -self.depth_int_limit,
                self.depth_int_limit,
            )
        depth_error = torch.where(
            depth_error_raw.abs() < self.depth_deadband,
            torch.zeros_like(depth_error_raw),
            depth_error_raw,
        )
        fz = (
            self.kp_depth * depth_error
            + self.ki_depth * self.depth_error_sum
            - self.kd_depth * lin_vel_b[:, 2]
        )
        # Attitude error in the same (half-angle) domain the loop was tuned in, but measured
        # against the trim attitude. For a XYZ rotation, quat x/y are sin(roll/2), sin(pitch/2),
        # so the target component is sin(trim/2). Keeps the existing gains' scale.
        target_qx = math.sin(0.5 * self.trim_roll_rad)
        target_qy = math.sin(0.5 * self.trim_pitch_rad)
        roll_error = root_quat_w[:, 1] - target_qx
        pitch_error = root_quat_w[:, 2] - target_qy
        if self.att_ki != 0.0:
            att_err = torch.stack([roll_error, pitch_error], dim=-1)
            self.att_error_sum = torch.clip(
                self.att_error_sum + att_err * dt, -self.att_int_limit, self.att_int_limit
            )
        mx = -self.kp_att * roll_error - self.kd_att * ang_vel_b[:, 0]
        my = -self.kp_att * pitch_error - self.kd_att * ang_vel_b[:, 1]
        if self.att_ki != 0.0:
            mx = mx - self.att_ki * self.att_error_sum[:, 0]
            my = my - self.att_ki * self.att_error_sum[:, 1]

        if self.thruster_layout == "gsrov8":
            # Horizontals carry Fx/Mz, verticals carry Fz/Mx/My; allocate as one wrench.
            wrench = torch.zeros(self.num_envs, 6, device=self.device)
            wrench[:, 0] = torch.clamp(u_surge, -1.0, 1.0) * self._surge_scale
            # Sway cancellation: lateral drift is the dominant failure mode for hover.
            wrench[:, 1] = torch.clamp(u_sway, -1.0, 1.0) * self._sway_scale
            wrench[:, 2] = torch.clamp(fz, -self.max_heave_wrench, self.max_heave_wrench)
            wrench[:, 3] = torch.clamp(mx, -self.max_heave_wrench, self.max_heave_wrench)
            wrench[:, 4] = torch.clamp(my, -self.max_heave_wrench, self.max_heave_wrench)
            wrench[:, 5] = u_yaw * self._yaw_scale
            u = allocate_thrusters(wrench, self._b_pinv, clamp=1.0, priority=False)
            return torch.clamp(u, -1.0, 1.0)

        pwm = torch.zeros(self.num_envs, 6, device=self.device)
        # Empirically both drives same-sign produces body surge; differential for yaw.
        pwm[:, 0] = torch.clamp(u_surge - u_yaw, -1.0, 1.0)
        pwm[:, 1] = torch.clamp(u_surge + u_yaw, -1.0, 1.0)

        # Depth / attitude → angled thrusters only
        wrench = torch.zeros(self.num_envs, 6, device=self.device)
        wrench[:, 2] = torch.clamp(fz, -self.max_heave_wrench, self.max_heave_wrench)
        wrench[:, 3] = torch.clamp(mx, -self.max_heave_wrench, self.max_heave_wrench)
        wrench[:, 4] = torch.clamp(my, -self.max_heave_wrench, self.max_heave_wrench)
        u_vert = allocate_thrusters(wrench, self._b_pinv, clamp=1.0, priority=False)
        pwm[:, 2:6] = u_vert[:, 2:6]
        return torch.clamp(pwm, -1.0, 1.0)
