# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Action term: raw 3-D stick -> cmd_vel -> PID -> thruster wrench (GS-ROV plant).

Replaces the Direct env's ``_pre_physics_step`` + ``_apply_action`` pair. ``process_actions``
runs once per policy step (raw action -> bounded ``cmd_vel``); ``apply_actions`` runs once per
physics sub-step (PID + hydrodynamics pushed onto the robot's external wrench), matching the
Direct workflow's per-sim-tick PID loop.
"""

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.assets import RigidObject
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..dynamics import GSROVPlant


class VelocityPIDAction(ActionTerm):
    """3-D stick ``[surge, sway, yaw]`` -> cmd_vel -> GS-ROV PID -> external wrench."""

    cfg: "VelocityPIDActionCfg"
    _asset: RigidObject

    def __init__(self, cfg: "VelocityPIDActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_cmd_vel = torch.zeros(self.num_envs, 3, device=self.device)

        env_cfg = self._env.cfg
        gravity_mag = float(torch.tensor(env_cfg.sim.gravity, device=self.device).norm())
        self._plant = GSROVPlant(
            num_envs=self.num_envs,
            device=self.device,
            cfg=env_cfg,
            sim_dt=float(env_cfg.sim.dt),
            gravity_mag=gravity_mag,
        )

    # -- ActionTerm API ------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        a = torch.clip(actions, -1.0, 1.0)
        cmd = a * self.cfg.scale
        limit = float(self.cfg.cmd_rate_limit)
        if limit > 0.0:
            lo = self._prev_cmd_vel - limit
            hi = self._prev_cmd_vel + limit
            cmd = torch.min(torch.max(cmd, lo), hi)
        self._processed_actions[:] = cmd
        self._prev_cmd_vel[:] = cmd

    def apply_actions(self):
        episode_time = self._env.episode_length_buf.to(dtype=torch.float32) * float(self._env.physics_dt)
        self._plant.step(self._processed_actions, self._asset, episode_time)

        # Keep the (non-sensor) USD camera prim in sync with the rigid body on render ticks,
        # mirroring the Direct env's ``_sync_robot_usd_pose`` call in ``_apply_action``.
        robot_usd = getattr(self._env, "_robot_usd", None)
        if robot_usd is not None:
            interval = max(int(self._env.cfg.sim.render_interval), 1)
            if self._env._sim_step_counter % interval == 0:
                self._env._sync_robot_usd_pose()

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._prev_cmd_vel[env_ids] = 0.0
        self._plant.reset(env_ids, depth_target=float(self._env.cfg.starting_depth))


@configclass
class VelocityPIDActionCfg(ActionTermCfg):
    """Configuration for :class:`VelocityPIDAction`.

    ``scale`` maps the unit policy action to a bounded target velocity (~0.5 m/s per horizontal
    axis and ~0.5 rad/s yaw, matching the Direct ``_pre_physics_step`` gains). ``cmd_rate_limit``
    applies a per-step slew limit in cmd_vel units (0 = off), identical to the Direct behaviour.
    """

    class_type: type = VelocityPIDAction
    asset_name: str = "robot"
    scale: float = 0.5
    cmd_rate_limit: float = 0.0
