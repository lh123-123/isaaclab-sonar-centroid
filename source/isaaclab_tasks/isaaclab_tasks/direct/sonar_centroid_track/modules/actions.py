# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Module M9: discrete or continuous actions → body-Y velocity (sway)."""

from __future__ import annotations

import torch

from .constants import ACTION_LEFT, ACTION_RIGHT, DEFAULT_MAX_LATERAL_VEL


def continuous_to_lateral_velocity(
    actions: torch.Tensor,
    *,
    max_lateral_velocity: float = DEFAULT_MAX_LATERAL_VEL,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map Box(-1,1) action to body-frame lateral velocity (m/s)."""
    a = actions.view(-1).clamp(-1.0, 1.0)
    n = num_envs if num_envs is not None else a.shape[0]
    dev = device if device is not None else a.device
    max_v = float(max_lateral_velocity)
    return a[:n].to(device=dev) * max_v


def discrete_to_lateral_velocity(
    actions: torch.Tensor,
    *,
    max_lateral_velocity: float = DEFAULT_MAX_LATERAL_VEL,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map Discrete(3) actions to body-frame lateral velocity (sway)."""
    a = actions.view(-1).long()
    n = num_envs if num_envs is not None else a.shape[0]
    dev = device if device is not None else a.device
    vy = torch.zeros(n, device=dev)
    vy[a == ACTION_LEFT] = -float(max_lateral_velocity)
    vy[a == ACTION_RIGHT] = float(max_lateral_velocity)
    return vy
