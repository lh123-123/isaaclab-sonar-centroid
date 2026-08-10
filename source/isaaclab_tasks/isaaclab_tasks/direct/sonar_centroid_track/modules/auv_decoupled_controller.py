# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Mission-frame decoupled PD: sway / surge / heave + roll / pitch / yaw hold.

Station layer (depth + attitude) is prioritized so the tracking task approximates
the existing 1-DoF lateral follow: when off depth/attitude, horizontal Fx/Fy is
gated down and thruster allocation keeps Fz/Mx/My/Mz first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_conjugate, wrap_to_pi

from isaaclab_tasks.direct.residual_rl.base_controller import BaseController
from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import robot_lateral_sway_signed

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from ..sonar_centroid_track_auv_env_cfg import SonarCentroidTrackAUVEnvCfg


class AUVDecoupledController:
    """Body-frame wrench from scalar mission errors (no coupled 3D pos PD)."""

    def __init__(self, cfg: SonarCentroidTrackAUVEnvCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device
        self._alloc = BaseController(
            num_envs,
            device,
            imperfect=False,
            kp=1.0,
            kd=0.0,
            kp_ang=1.0,
            kd_ang=0.0,
            thrust_bias=0.0,
            max_wrench=float(getattr(cfg, "auv_max_wrench", 200.0)),
        )
        self._heave_i = torch.zeros(self.num_envs, device=self.device)
        # Delayed station obs for sim→real noise: [delay+1, N, 8]
        # (heave_e, roll, pitch, yaw_err, vz, wx, wy, wz)
        self._station_obs_ring: torch.Tensor | None = None
        self._station_obs_ring_i: int = 0

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._heave_i.zero_()
            if self._station_obs_ring is not None:
                self._station_obs_ring.zero_()
            self._station_obs_ring_i = 0
        else:
            self._heave_i[env_ids] = 0.0
            if self._station_obs_ring is not None:
                self._station_obs_ring[:, env_ids] = 0.0

    def _station_obs_for_control(
        self,
        heave_e: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        yaw_err: torch.Tensor,
        vz: torch.Tensor,
        ang_vel_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optional Gaussian noise + ring delay on station-loop measurements only."""
        cfg = self.cfg
        if not bool(getattr(cfg, "auv_station_obs_noise", False)):
            return heave_e, roll, pitch, yaw_err, vz, ang_vel_b

        n = heave_e.shape[0]
        sig_d = float(getattr(cfg, "auv_noise_depth_m", 0.02))
        sig_a = float(getattr(cfg, "auv_noise_att_rad", 0.02))
        sig_vz = float(getattr(cfg, "auv_noise_vz", 0.02))
        sig_w = float(getattr(cfg, "auv_noise_ang_vel", 0.02))
        delay = max(int(getattr(cfg, "auv_noise_delay_steps", 0)), 0)

        heave_n = heave_e + sig_d * torch.randn(n, device=self.device)
        roll_n = roll + sig_a * torch.randn(n, device=self.device)
        pitch_n = pitch + sig_a * torch.randn(n, device=self.device)
        yaw_n = yaw_err + sig_a * torch.randn(n, device=self.device)
        vz_n = vz + sig_vz * torch.randn(n, device=self.device)
        ang_n = ang_vel_b + sig_w * torch.randn_like(ang_vel_b)

        packed = torch.stack(
            [heave_n, roll_n, pitch_n, yaw_n, vz_n, ang_n[:, 0], ang_n[:, 1], ang_n[:, 2]],
            dim=-1,
        )
        if delay <= 0:
            out = packed
        else:
            ring_len = delay + 1
            if self._station_obs_ring is None or self._station_obs_ring.shape[0] != ring_len:
                self._station_obs_ring = packed.unsqueeze(0).repeat(ring_len, 1, 1)
                self._station_obs_ring_i = 0
            self._station_obs_ring[self._station_obs_ring_i] = packed
            read_i = (self._station_obs_ring_i - delay) % ring_len
            out = self._station_obs_ring[read_i]
            self._station_obs_ring_i = (self._station_obs_ring_i + 1) % ring_len

        heave_o = out[:, 0]
        roll_o = out[:, 1]
        pitch_o = out[:, 2]
        yaw_o = out[:, 3]
        vz_o = out[:, 4]
        ang_o = out[:, 5:8]
        return heave_o, roll_o, pitch_o, yaw_o, vz_o, ang_o

    def compute_pwm(
        self,
        desired_sway_m: torch.Tensor,
        robot: RigidObject,
        *,
        station_origin_w: torch.Tensor,
        fixed_yaw_w: torch.Tensor,
        target_vy: torch.Tensor | None = None,
        desired_surge_m: torch.Tensor | None = None,
        desired_heave_m: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mission setpoints (m) -> bounded 6 PWM. Depth loop is body-frame."""
        n = self.num_envs
        sway_sp = desired_sway_m.view(n).float()
        yaw_fixed = fixed_yaw_w.view(n).float()
        sway = robot_lateral_sway_signed(station_origin_w, robot.data.root_pos_w, yaw_fixed)

        zeros = torch.zeros(n, device=self.device)
        forward_w = torch.stack([torch.cos(yaw_fixed), torch.sin(yaw_fixed), zeros], dim=-1)
        delta_w = robot.data.root_pos_w - station_origin_w
        surge = torch.sum(delta_w * forward_w, dim=-1)

        if desired_surge_m is None:
            surge_sp = surge
        else:
            surge_sp = desired_surge_m.view(n).float()

        if desired_heave_m is None:
            heave_sp = station_origin_w[:, 2]
        else:
            heave_sp = desired_heave_m.view(n).float()

        roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        yaw_err = wrap_to_pi(yaw_fixed - yaw)

        lin_vel_b = robot.data.root_lin_vel_b
        ang_vel_b = robot.data.root_ang_vel_b
        # Depth: station world-Z setpoint → body-z error + body vz.
        target_pos_w = robot.data.root_pos_w.clone()
        target_pos_w[:, 2] = heave_sp
        heave_err_b = quat_apply(quat_conjugate(robot.data.root_quat_w), target_pos_w - robot.data.root_pos_w)
        heave_e = heave_err_b[:, 2]

        # Step B: dirty station measurements for depth/attitude loops only.
        heave_e, roll, pitch, yaw_err, vz_b, ang_vel_b = self._station_obs_for_control(
            heave_e, roll, pitch, yaw_err, lin_vel_b[:, 2], ang_vel_b
        )

        cfg = self.cfg
        kp_sway = float(getattr(cfg, "auv_kp_sway", getattr(cfg, "auv_pos_kp", 12.0)))
        kd_sway = float(getattr(cfg, "auv_kd_sway", getattr(cfg, "auv_pos_kd", 3.0)))
        kp_surge = float(getattr(cfg, "auv_kp_surge", kp_sway))
        kd_surge = float(getattr(cfg, "auv_kd_surge", kd_sway))
        kp_depth = float(getattr(cfg, "auv_kp_depth", kp_sway))
        kd_depth = float(getattr(cfg, "auv_kd_depth", kd_sway))
        ki_depth = float(getattr(cfg, "auv_ki_depth", 0.0))
        ki_lim = float(getattr(cfg, "auv_ki_depth_limit", 40.0))
        kp_roll = float(getattr(cfg, "auv_kp_roll", getattr(cfg, "auv_kp_yaw", 80.0)))
        kd_roll = float(getattr(cfg, "auv_kd_roll", getattr(cfg, "auv_kd_yaw", 14.0)))
        kp_pitch = float(getattr(cfg, "auv_kp_pitch", kp_roll))
        kd_pitch = float(getattr(cfg, "auv_kd_pitch", kd_roll))
        kp_yaw = float(getattr(cfg, "auv_kp_yaw", 80.0))
        kd_yaw = float(getattr(cfg, "auv_kd_yaw", 14.0))

        dt = float(cfg.sim.dt) * float(getattr(cfg, "decimation", 1))
        if ki_depth > 0.0:
            self._heave_i = (self._heave_i + heave_e * dt).clamp(-ki_lim, ki_lim)
        else:
            self._heave_i.zero_()

        wrench = torch.zeros(n, 6, device=self.device)
        # Sway/surge use clean horizontal velocity; depth/attitude use noisy station obs.
        wrench[:, 0] = kp_surge * (surge_sp - surge) - kd_surge * lin_vel_b[:, 0]
        wrench[:, 1] = kp_sway * (sway_sp - sway) - kd_sway * lin_vel_b[:, 1]
        wrench[:, 2] = kp_depth * heave_e + ki_depth * self._heave_i - kd_depth * vz_b
        wrench[:, 3] = kp_roll * (-roll) - kd_roll * ang_vel_b[:, 0]
        wrench[:, 4] = kp_pitch * (-pitch) - kd_pitch * ang_vel_b[:, 1]
        wrench[:, 5] = kp_yaw * yaw_err - kd_yaw * ang_vel_b[:, 2]

        if target_vy is not None:
            vff = float(getattr(cfg, "auv_vy_ff", 55.0))
            mass = float(getattr(cfg, "auv_mass", 22.701))
            vy_b = lin_vel_b[:, 1]
            tgt_v = target_vy.view(n).float()
            wrench[:, 1] = wrench[:, 1] + vff * mass * (tgt_v - vy_b)

        # Station-first gate: when depth/attitude bad, shrink mission Fy (lateral)
        # so station Z/attitude can recover. Never gate Fx — surge hold must keep
        # braking or the vehicle creeps forward under priority PWM starvation.
        if bool(getattr(cfg, "auv_station_gate", True)):
            d_gate = max(float(getattr(cfg, "auv_station_gate_depth_m", 0.12)), 1.0e-3)
            a_gate = max(float(getattr(cfg, "auv_station_gate_att_rad", 0.10)), 1.0e-3)
            att_e = roll.abs() + pitch.abs()
            d_excess = (heave_e.abs() - d_gate).clamp(min=0.0) / d_gate
            a_excess = (att_e - a_gate).clamp(min=0.0) / a_gate
            scale = (1.0 - d_excess - a_excess).clamp(
                float(getattr(cfg, "auv_station_gate_min_scale", 0.05)), 1.0
            )
            gate_surge = bool(getattr(cfg, "auv_station_gate_surge", False))
            if gate_surge:
                wrench[:, 0] = wrench[:, 0] * scale
            wrench[:, 1] = wrench[:, 1] * scale

        fmax_h = float(getattr(cfg, "auv_max_force_horiz", getattr(cfg, "auv_max_force", 280.0)))
        fmax_z = float(getattr(cfg, "auv_max_force_heave", getattr(cfg, "auv_max_force", 280.0)))
        mmax = float(getattr(cfg, "auv_max_torque", 55.0))
        wrench[:, 0] = torch.clamp(wrench[:, 0], -fmax_h, fmax_h)
        wrench[:, 1] = torch.clamp(wrench[:, 1], -fmax_h, fmax_h)
        wrench[:, 2] = torch.clamp(wrench[:, 2], -fmax_z, fmax_z)
        wrench[:, 3:6] = torch.clamp(wrench[:, 3:6], -mmax, mmax)

        priority = bool(getattr(cfg, "auv_alloc_priority", True))
        pri_att = bool(getattr(cfg, "auv_alloc_priority_attitude", True))
        return self._alloc.allocate_thrusters_from_wrench(
            wrench, priority=priority, priority_attitude=pri_att
        )
