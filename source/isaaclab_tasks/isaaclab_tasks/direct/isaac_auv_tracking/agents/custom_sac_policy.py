# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Asymmetric SAC policy (Holo CustomSACPolicy): Actor=image, Critic=state+action."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.sac.policies import SACPolicy


class StateCritic(nn.Module):
    """Q-networks on belief state + action (no image features)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        n_critics: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_critics = n_critics
        self.q_networks = nn.ModuleList()
        for _ in range(n_critics):
            q_net = create_mlp(
                input_dim=state_dim + action_dim,
                output_dim=1,
                net_arch=net_arch,
                activation_fn=activation_fn,
            )
            self.q_networks.append(nn.Sequential(*q_net))

    def forward(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        state = obs["state"]
        qvalue_input = torch.cat([state, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        state = obs["state"]
        qvalue_input = torch.cat([state, actions], dim=1)
        return self.q_networks[0](qvalue_input)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class CustomSACPolicy(SACPolicy):
    """Actor uses image encoder; Critic uses state only. ``share_features_extractor=False``."""

    def __init__(self, *args, **kwargs):
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor: Optional[nn.Module] = None) -> StateCritic:
        del features_extractor  # unused — critic reads obs['state'] directly
        state_dim = int(self.observation_space["state"].shape[0])
        action_dim = get_action_dim(self.action_space)
        if isinstance(self.net_arch, dict):
            critic_arch = self.net_arch.get("qf", [512, 256])
        else:
            critic_arch = [512, 256]
        return StateCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            net_arch=list(critic_arch),
            activation_fn=self.activation_fn,
            n_critics=2,
        ).to(self.device)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)
