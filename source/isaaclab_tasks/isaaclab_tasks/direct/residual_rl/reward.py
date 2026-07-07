# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for residual RL position/orientation tracking."""

from __future__ import annotations

import torch


@torch.jit.script
def compute_residual_rewards(
    rew_scale_pos: float,
    rew_scale_pos_lin: float,
    rew_scale_ang: float,
    rew_scale_u_rl: float,
    rew_scale_progress: float,
    rew_scale_vel_near: float,
    rew_scale_overshoot: float,
    rew_scale_success: float,
    rew_scale_terminal: float,
    success_pos_threshold: float,
    near_goal_dist: float,
    is_timeout: torch.Tensor,
    prev_pos_err_norm: torch.Tensor,
    pos_err_norm: torch.Tensor,
    pos_err_norm_sq: torch.Tensor,
    overshoot_x: torch.Tensor,
    lin_vel_norm_sq: torch.Tensor,
    ang_err: torch.Tensor,
    residual: torch.Tensor,
    wrench_opposition: torch.Tensor,
    rew_scale_wrench_opp: float,
    success_bonus_pending: torch.Tensor,
) -> torch.Tensor:
    """Dense tracking + progress + overshoot/velocity penalties for residual PPO."""
    rew_pos = rew_scale_pos * torch.exp(-pos_err_norm_sq) + rew_scale_pos_lin * (-pos_err_norm)
    rew_ang = rew_scale_ang * torch.exp(-ang_err)
    rew_u_rl = -rew_scale_u_rl * torch.sum(residual * residual, dim=1)
    rew_wrench_opp = -rew_scale_wrench_opp * wrench_opposition
    rew_progress = rew_scale_progress * (prev_pos_err_norm - pos_err_norm)

    near_goal = (pos_err_norm < near_goal_dist).float()
    rew_vel = -rew_scale_vel_near * lin_vel_norm_sq * near_goal
    rew_overshoot = -rew_scale_overshoot * overshoot_x * overshoot_x

    rew_success = (
        rew_scale_success
        * success_bonus_pending.float()
        * (pos_err_norm < success_pos_threshold).float()
    )

    rew_terminal = rew_scale_terminal * (-pos_err_norm) * is_timeout.float()

    return (
        rew_pos
        + rew_ang
        + rew_u_rl
        + rew_wrench_opp
        + rew_progress
        + rew_vel
        + rew_overshoot
        + rew_success
        + rew_terminal
    )
