# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the v7 flow-hover AUV task.

Each function returns the *un-weighted* per-term value of shape ``(num_envs,)``; the manager
multiplies by ``RewTerm.weight`` and sums, reproducing the Direct env's single-expression reward:

    rew = rew_flow*flow_term + rew_alive*alive + rew_depth*depth_term + rew_att*att_term
          - rew_action*smooth
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat


def _present(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (env._texture_count >= env.cfg.min_texture_px).to(torch.float32)


def flow_nulling(env: ManagerBasedRLEnv, sigma_flow: float = 0.05) -> torch.Tensor:
    env._ensure_flow_updated()
    present = _present(env)
    flow_rms_n = (env._flow_rms / float(env.cfg.cam_width)).clamp(0.0, 1.0)
    return present * torch.exp(-(flow_rms_n**2) / (sigma_flow**2))


def alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    env._ensure_flow_updated()
    return _present(env)


def depth_hold(env: ManagerBasedRLEnv, sigma_depth: float = 1.5) -> torch.Tensor:
    depth_err = env._robot.data.root_pos_w[:, 2] - float(env.cfg.starting_depth)
    return torch.exp(-(depth_err**2) / (sigma_depth**2))


def attitude_hold(env: ManagerBasedRLEnv, sigma_att: float = 0.3) -> torch.Tensor:
    roll, pitch, _ = euler_xyz_from_quat(env._robot.data.root_quat_w)
    roll_err = roll - float(env.cfg.trim_roll_rad)
    pitch_err = pitch - float(env.cfg.trim_pitch_rad)
    att_err = torch.sqrt(roll_err * roll_err + pitch_err * pitch_err)
    return torch.exp(-(att_err**2) / (sigma_att**2))


def action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    a = env.action_manager.action.clamp(-1.0, 1.0)
    pa = env.action_manager.prev_action.clamp(-1.0, 1.0)
    return torch.sum((a - pa) ** 2, dim=-1)
