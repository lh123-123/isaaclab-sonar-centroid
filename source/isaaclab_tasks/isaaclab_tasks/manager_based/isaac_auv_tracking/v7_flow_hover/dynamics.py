# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared GS-ROV plant (PID + thrusters + buoyancy/viscosity) for the manager-based v7 task.

This is a faithful extraction of the physics that used to live inside
``AUVTrackingEnv`` (Direct) so that both the Direct and Manager-Based workflows drive
the *identical* low-level model. The numbers/formulas here mirror
``auv_tracking_env.py::_compute_dynamics`` / ``_apply_action`` exactly.

The plant is a plain object (no env dependency). The manager-based ``VelocityPIDAction``
term owns one instance and calls :meth:`step` once per physics sub-step.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.direct.isaac_auv_tracking.batch_pid import BatchPIDController
from isaaclab_tasks.direct.isaac_auv_tracking.ocean_current import (
    clip_current_velocity,
    create_ocean_current_field,
)
from isaaclab_tasks.direct.isaac_auv_tracking.rigid_body_hydrodynamics import HydrodynamicForceModels
from isaaclab_tasks.direct.isaac_auv_tracking.thruster_dynamics import (
    ConversionFunctionBasic,
    DynamicsFirstOrder,
    get_gsrov_thruster_geometry,
    get_thruster_com_and_orientations,
)


class GSROVPlant:
    """Batched hydrodynamics + thruster/PID model for the GS-ROV 8-thruster hull.

    Mirrors ``AUVTrackingEnv`` physics. ``cfg`` must expose the same fields the Direct
    ``AUVTrackingEnvCfg`` does (water_rho, water_beta, rotor_constant, dyn_time_constant,
    thruster_first_order_lag, inertia_diag, com_to_cob_offset, volume, mass, thruster_* ...).
    """

    def __init__(self, num_envs: int, device: str | torch.device, cfg, sim_dt: float, gravity_mag: float):
        self.num_envs = int(num_envs)
        self.device = device
        self.cfg = cfg
        self._sim_dt = float(sim_dt)

        # --- thruster geometry (same branch logic as the Direct base class) ---
        self.thruster_layout = str(getattr(cfg, "thruster_layout", "warpauv6")).lower()
        if self.thruster_layout == "gsrov8":
            geometry = get_gsrov_thruster_geometry(device)
        else:
            self.thruster_layout = "warpauv6"
            geometry = get_thruster_com_and_orientations(device)
        base_offsets, base_quats = geometry
        self.num_thrusters = int(base_offsets.shape[0])
        self.thruster_com_offsets = base_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = base_quats.repeat(self.num_envs, 1)

        self._pwm = torch.zeros(self.num_envs, self.num_thrusters, device=device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=device)

        # --- rigid-body / fluid properties ---
        self._gravity_mag = float(gravity_mag)
        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=device, dtype=torch.float)
        inr = tuple(getattr(cfg, "inertia_diag", (0.37, 0.97, 1.19)))
        self.inertia_tensors[:, 0] = float(inr[0])
        self.inertia_tensors[:, 1] = float(inr[1])
        self.inertia_tensors[:, 2] = float(inr[2])
        self.masses = torch.full((self.num_envs, 1), cfg.mass, device=device)
        self.com_to_cob_offsets = torch.tensor(cfg.com_to_cob_offset, device=device).repeat(self.num_envs, 1)
        self.volumes = torch.full((self.num_envs, 1), cfg.volume, device=device)

        self.force_calc = HydrodynamicForceModels(self.num_envs, device, False)
        self.thruster_dyn = DynamicsFirstOrder(
            self.num_envs,
            self.num_thrusters,
            float(cfg.dyn_time_constant),
            device,
            enable_lag=bool(getattr(cfg, "thruster_first_order_lag", False)),
        )
        self.thruster_conv = ConversionFunctionBasic(cfg.rotor_constant)

        # Rated thrust scaled by (V/V_rated)^2.
        self._rated_voltage = float(getattr(cfg, "thruster_rated_voltage_v", 24.0))
        self._battery_voltage = float(getattr(cfg, "thruster_battery_voltage_v", self._rated_voltage))
        voltage_scale = (self._battery_voltage / max(self._rated_voltage, 1e-6)) ** 2
        self._thrust_fwd_n = float(getattr(cfg, "thruster_max_thrust_fwd_n", 19.613)) * voltage_scale
        self._thrust_rev_n = float(getattr(cfg, "thruster_max_thrust_rev_n", 17.652)) * voltage_scale
        self._deadband = float(getattr(cfg, "thruster_deadband", 0.05))

        self.pid = BatchPIDController(
            num_envs=self.num_envs,
            device=device,
            dt=self._sim_dt,
            thruster_layout=self.thruster_layout,
            thruster_offsets=base_offsets if self.thruster_layout == "gsrov8" else None,
            thruster_quats=base_quats if self.thruster_layout == "gsrov8" else None,
            trim_roll_rad=float(getattr(cfg, "trim_roll_rad", 0.0)),
            trim_pitch_rad=float(getattr(cfg, "trim_pitch_rad", 0.0)),
        )

        # --- optional ocean current field (calm water by default) ---
        self.ocean_current_field = create_ocean_current_field(getattr(cfg, "ocean_current", None))
        self._ocean_current_vel = torch.zeros((self.num_envs, 3), device=device)

    # ------------------------------------------------------------------ helpers
    def _compute_dynamics(
        self, motor: torch.Tensor, robot, episode_time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Motor command ``(N, n_thrusters)`` -> world force ``(N, 1, 3)`` and torque ``(N, 1, 3)``.

        Verbatim port of ``AUVTrackingEnv._compute_dynamics`` (gsrov8 + warpauv6 branches).
        ``episode_time`` is the per-env clock ``t`` (shape ``(N,)``) the first-order thruster
        lag integrates against.
        """
        n_thrusters = self.num_thrusters
        thruster_forces = torch.zeros((self.num_envs, n_thrusters, 3), device=self.device, dtype=torch.float)
        motor = torch.clone(motor)
        motor = torch.nan_to_num(motor, nan=0.0, posinf=0.0, neginf=0.0)
        t = episode_time  # shape (N,)

        if self.thruster_layout == "gsrov8":
            motor = self.thruster_dyn.update(motor, t)
            motor = torch.where(motor.abs() < self._deadband, torch.zeros_like(motor), motor)
            thrust = torch.where(
                motor >= 0.0,
                torch.full_like(motor, self._thrust_fwd_n),
                torch.full_like(motor, self._thrust_rev_n),
            )
            thrust = thrust * motor * motor.abs()
        else:
            motor[torch.abs(motor) < 0.08] = 0
            mp, mn = motor >= 0.08, motor <= -0.08
            motor[mp] = -139.0 * (motor[mp] ** 2) + 500 * motor[mp] + 8.28
            motor[mn] = 161.0 * (motor[mn] ** 2) + 517.86 * motor[mn] - 5.72
            motor = self.thruster_dyn.update(motor, t)
            thrust = self.thruster_conv.convert(motor)

        thruster_forces[..., 0] = 1.0
        thruster_forces = quat_apply(self.thruster_quats, thruster_forces)
        thruster_forces = thruster_forces * thrust.unsqueeze(-1)
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)
        thruster_forces = torch.sum(thruster_forces, dim=-2)
        thruster_torques = torch.sum(thruster_torques, dim=-2)

        bf, bt = self.force_calc.calculate_buoyancy_forces(
            robot.data.root_quat_w,
            self.cfg.water_rho,
            self.volumes,
            abs(self._gravity_mag),
            self.com_to_cob_offsets,
        )
        lin_vel = robot.data.root_lin_vel_w
        if self.ocean_current_field is not None or torch.any(self._ocean_current_vel != 0):
            lin_vel = lin_vel - self._ocean_current_vel
        df, dt_, vf, vt_ = self.force_calc.calculate_density_and_viscosity_forces(
            robot.data.root_quat_w,
            lin_vel,
            robot.data.root_ang_vel_w,
            self.inertia_tensors,
            self.inertia_tensors.mean(dim=1, keepdim=True),
            self.cfg.water_beta,
            self.cfg.water_rho,
            self.masses,
        )
        return df + bf + vf + thruster_forces, dt_ + bt + vt_ + thruster_torques

    def set_ocean_currents(self, env_ids, velocity) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        v = torch.as_tensor(velocity, device=self.device, dtype=torch.float32)
        if v.ndim == 1:
            v = v.view(1, 3).expand(env_ids.numel(), 3).clone()
        v = clip_current_velocity(v, float(getattr(self.cfg.ocean_current, "clip_speed", 5.0)))
        self._ocean_current_vel[env_ids] = v

    # ------------------------------------------------------------------ step / reset
    def step(self, cmd_vel: torch.Tensor, robot, episode_time: torch.Tensor) -> None:
        """Run PID at physics rate and push (force, torque) onto the robot's external wrench."""
        if self.ocean_current_field is not None:
            v = self.ocean_current_field(robot.data.root_pos_w, episode_time)
            self.set_ocean_currents(None, v)

        self._pwm[:] = self.pid.compute(
            cmd_vel,
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            robot.data.root_lin_vel_w,
            robot.data.root_ang_vel_w,
            dt=self._sim_dt,
        )
        self._thrust[:, 0, :], self._moment[:, 0, :] = self._compute_dynamics(self._pwm, robot, episode_time)
        robot.set_external_force_and_torque(self._thrust, self._moment)

    def reset(self, env_ids: torch.Tensor, depth_target: float) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        self._pwm[env_ids] = 0.0
        self.pid.reset(env_ids, depth_target=float(depth_target))
        if getattr(self.cfg, "thruster_first_order_lag", False):
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[env_ids] = True
            self.thruster_dyn.reset(reset_mask)
        self._ocean_current_vel[env_ids] = 0.0
