# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""WarpAUV thruster + hydrodynamics helper (from isaac-auv-env)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import importlib
import torch
from isaaclab.utils.math import quat_apply

_thruster = importlib.import_module("isaaclab_tasks.direct.isaac-auv-env.thruster_dynamics")
_hydro = importlib.import_module("isaaclab_tasks.direct.isaac-auv-env.rigid_body_hydrodynamics")

DynamicsFirstOrder = _thruster.DynamicsFirstOrder
ConversionFunctionBasic = _thruster.ConversionFunctionBasic
get_thruster_com_and_orientations = _thruster.get_thruster_com_and_orientations
HydrodynamicForceModels = _hydro.HydrodynamicForceModels

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from ..sonar_centroid_track_auv_env_cfg import SonarCentroidTrackAUVEnvCfg


class WarpAUVDynamicsHelper:
    """Apply 6-PWM thruster model + analytic hydrodynamics to a WarpAUV ``RigidObject``."""

    def __init__(self, cfg: SonarCentroidTrackAUVEnvCfg, num_envs: int, device: str | torch.device, robot: RigidObject):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device
        self.robot = robot

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._pwm = torch.zeros(self.num_envs, 6, device=self.device)

        offsets, quats = get_thruster_com_and_orientations(self.device)
        self.thruster_com_offsets = offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = quats.repeat(self.num_envs, 1, 1)

        self._gravity_magnitude = torch.tensor(cfg.sim.gravity, device=self.device).norm()

        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device)
        self.inertia_tensors[:, 0] = 0.37
        self.inertia_tensors[:, 1] = 0.97
        self.inertia_tensors[:, 2] = 1.19
        self.inertia_tensors_mean = self.inertia_tensors.mean(dim=1, keepdim=True)

        self.masses = torch.full((self.num_envs, 1), float(cfg.auv_mass), device=self.device)
        self.com_to_cob_offsets = torch.tensor(cfg.auv_com_to_cob_offset, device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self.volumes = torch.full((self.num_envs, 1), float(cfg.auv_volume), device=self.device)

        self.force_calculation_functions = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_dynamics = DynamicsFirstOrder(
            self.num_envs, 6, float(cfg.auv_dyn_time_constant), self.device
        )
        self.thruster_conversion = ConversionFunctionBasic(float(cfg.auv_rotor_constant))

    def apply_pwm(self, pwm: torch.Tensor, *, sim_dt: float, step_count: torch.Tensor) -> None:
        """Compute wrench from PWM + hydro and write to the robot."""
        self._pwm[:] = torch.clamp(pwm.view(self.num_envs, 6), -1.0, 1.0)
        forces_b, torques_b = self._compute_wrench(self._pwm, sim_dt=sim_dt, step_count=step_count)
        self._thrust[:, 0, :] = forces_b
        self._moment[:, 0, :] = torques_b
        self.robot.set_external_force_and_torque(self._thrust, self._moment)

    def apply_hover(self, *, sim_dt: float, step_count: torch.Tensor) -> None:
        """Zero PWM — buoyancy + drag only (Phase A0 settle)."""
        self.apply_pwm(torch.zeros(self.num_envs, 6, device=self.device), sim_dt=sim_dt, step_count=step_count)

    def _compute_wrench(
        self, actions: torch.Tensor, *, sim_dt: float, step_count: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        thruster_forces = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        motor_values = torch.clone(actions)

        motor_values[torch.abs(motor_values) < 0.08] = 0.0
        motor_values[motor_values >= 0.08] = (
            -139.0 * (motor_values[motor_values >= 0.08] ** 2.0)
            + 500.0 * motor_values[motor_values >= 0.08]
            + 8.28
        )
        motor_values[motor_values <= -0.08] = (
            161.0 * (motor_values[motor_values <= -0.08] ** 2.0)
            + 517.86 * motor_values[motor_values <= -0.08]
            - 5.72
        )

        motor_values = self.thruster_dynamics.update(motor_values, step_count * sim_dt)
        motor_values = self.thruster_conversion.convert(motor_values)

        thruster_forces[..., 0] = 1.0
        thruster_forces = quat_apply(self.thruster_quats, thruster_forces)
        thruster_forces = thruster_forces * motor_values.unsqueeze(-1)
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)

        thruster_forces_sum = torch.sum(thruster_forces, dim=-2)
        thruster_torques_sum = torch.sum(thruster_torques, dim=-2)

        buoyancy_forces, buoyancy_torques = self.force_calculation_functions.calculate_buoyancy_forces(
            self.robot.data.root_quat_w,
            float(self.cfg.auv_water_rho),
            self.volumes,
            abs(float(self._gravity_magnitude.item())),
            self.com_to_cob_offsets,
        )
        density_forces, density_torques, viscosity_forces, viscosity_torques = (
            self.force_calculation_functions.calculate_density_and_viscosity_forces(
                self.robot.data.root_quat_w,
                self.robot.data.root_lin_vel_w,
                self.robot.data.root_ang_vel_w,
                self.inertia_tensors,
                self.inertia_tensors_mean,
                float(self.cfg.auv_water_beta),
                float(self.cfg.auv_water_rho),
                self.masses,
            )
        )

        forces = density_forces + buoyancy_forces + viscosity_forces + thruster_forces_sum
        torques = density_torques + buoyancy_torques + viscosity_torques + thruster_torques_sum
        return forces, torques
