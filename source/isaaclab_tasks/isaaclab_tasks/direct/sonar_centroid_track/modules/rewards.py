# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Module M8: reward from fixed-point tracking error."""

from __future__ import annotations

import torch

from .constants import (
    MIN_VALID_RATIO,
    REWARD_W_ACTION_SWITCH,
    REWARD_W_SAFETY,
    REWARD_W_U,
    REWARD_W_V,
)
from .sonar_centroid_tracker import SonarCentroidTracker


def compute_reward(
    tracker: SonarCentroidTracker,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    *,
    min_depth: float,
    max_depth: float,
    depth: torch.Tensor,
    w_u: float = REWARD_W_U,
    w_v: float = REWARD_W_V,
    w_switch: float = REWARD_W_ACTION_SWITCH,
    w_safety: float = REWARD_W_SAFETY,
    min_valid_ratio: float = MIN_VALID_RATIO,
    continuous_actions: bool = False,
) -> torch.Tensor:
    """Minimize ||p_sonar|| in fixed-point frame; optional action-switch penalty."""
    num_envs = actions.shape[0]
    device = actions.device
    reward = torch.zeros(num_envs, device=device)
    a = actions.view(-1)
    pa = prev_actions.view(-1)

    for i in range(num_envs):
        p = tracker.detected_rel(env_id=i)
        if p is not None:
            reward[i] -= w_u * abs(p.col) / max(float(tracker.polar_width), 1.0)
            reward[i] -= w_v * abs(p.row) / max(float(tracker.polar_height), 1.0)

        if continuous_actions:
            reward[i] -= w_switch * abs(a[i].item() - pa[i].item())
        elif int(a[i].item()) != int(pa[i].item()):
            reward[i] -= w_switch

        unsafe = tracker.valid_ratio(env_id=i) < min_valid_ratio
        unsafe = unsafe or bool((depth[i] < min_depth or depth[i] > max_depth).item())
        if unsafe:
            reward[i] -= w_safety

    return reward
