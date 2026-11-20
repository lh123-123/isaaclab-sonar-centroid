# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms + reset event for the v7 flow-hover AUV task."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat


def texture_lost(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when the texture-lost streak exceeds the configured limit."""
    env._ensure_flow_updated()
    return env._texture_lost_streak >= int(env.cfg.texture_lost_limit_steps)


def depth_exceeded(env: ManagerBasedRLEnv) -> torch.Tensor:
    depth_err = (env._robot.data.root_pos_w[:, 2] - float(env.cfg.starting_depth)).abs()
    return depth_err > float(env.cfg.depth_limit)


def attitude_exceeded(env: ManagerBasedRLEnv) -> torch.Tensor:
    roll, pitch, _ = euler_xyz_from_quat(env._robot.data.root_quat_w)
    roll_err = roll - float(env.cfg.trim_roll_rad)
    pitch_err = pitch - float(env.cfg.trim_pitch_rad)
    att_err = torch.sqrt(roll_err * roll_err + pitch_err * pitch_err)
    return att_err > float(env.cfg.att_limit)


def out_of_bounds(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when the vehicle leaves the (margin-shrunk) workspace box."""
    oob = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if bool(env.cfg.use_boundaries):
        pos = env._robot.data.root_pos_w
        origins = env.scene.env_origins
        half_x = float(env.cfg.max_auv_x) - float(env.cfg.margin2wall)
        half_y = float(env.cfg.max_auv_y) - float(env.cfg.margin2wall)
        oob = (
            (torch.abs(pos[:, 0] - origins[:, 0]) > half_x)
            | (torch.abs(pos[:, 1] - origins[:, 1]) > half_y)
            | (torch.abs(pos[:, 2] - float(env.cfg.starting_depth)) > float(env.cfg.max_auv_z))
        )
    return oob


def reset_auv_buffers(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset event: clear the per-env optical-flow / texture buffers for reset envs.

    The plant (PID + thruster state) is reset separately by ``VelocityPIDAction.reset`` which
    the action manager invokes in the same ``_reset_idx`` pass.
    """
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)
    env._flow_image[env_ids] = 0.0
    env._texture_mask[env_ids] = 0.0
    env._flow_rms[env_ids] = 0.0
    env._texture_count[env_ids] = 0.0
    env._prev_rgb_ok[env_ids] = False
    env._texture_lost_streak[env_ids] = 0
