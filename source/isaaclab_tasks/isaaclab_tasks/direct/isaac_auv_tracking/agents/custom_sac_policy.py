# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""SAC policies for Direct-v1: asymmetric, concat-hybrid, and gated-residual critics."""

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


class ImageStateCritic(nn.Module):
    """Q-networks on concat(image features, belief state, action)."""

    def __init__(
        self,
        features_extractor: nn.Module,
        state_dim: int,
        action_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        n_critics: int = 2,
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_critics = n_critics
        feat_dim = int(features_extractor.features_dim)
        self.q_networks = nn.ModuleList()
        for _ in range(n_critics):
            q_net = create_mlp(
                input_dim=feat_dim + state_dim + action_dim,
                output_dim=1,
                net_arch=net_arch,
                activation_fn=activation_fn,
            )
            self.q_networks.append(nn.Sequential(*q_net))

    def _q_input(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        img_feat = self.features_extractor(obs)
        state = obs["state"]
        if state.dim() > 2:
            state = state.reshape(state.shape[0], -1)
        return torch.cat([img_feat, state, actions], dim=1)

    def forward(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        qvalue_input = self._q_input(obs, actions)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        return self.q_networks[0](self._q_input(obs, actions))

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


class HybridImageStateSACPolicy(SACPolicy):
    """Actor=image encoder; Critic=image features ⊕ belief state ⊕ action.

    ``share_features_extractor=False`` so the critic has its own Compact CNN.
    """

    def __init__(self, *args, **kwargs):
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor: Optional[nn.Module] = None) -> ImageStateCritic:
        if features_extractor is None:
            features_extractor = self.features_extractor_class(
                self.observation_space,
                **self.features_extractor_kwargs,
            )
        state_dim = int(self.observation_space["state"].shape[0])
        action_dim = get_action_dim(self.action_space)
        if isinstance(self.net_arch, dict):
            critic_arch = self.net_arch.get("qf", [512, 256])
        else:
            critic_arch = [512, 256]
        return ImageStateCritic(
            features_extractor=features_extractor,
            state_dim=state_dim,
            action_dim=action_dim,
            net_arch=list(critic_arch),
            activation_fn=self.activation_fn,
            n_critics=2,
        ).to(self.device)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)


class GatedImageBeliefCritic(nn.Module):
    """Q = Q_belief(state, a) + σ(g(state)) · Q_img(image, a).

    Gate is state-only and biased closed at init so the image path cannot drown belief.
    """

    def __init__(
        self,
        features_extractor: nn.Module,
        state_dim: int,
        action_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        n_critics: int = 2,
        gate_bias: float = -2.0,
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_critics = n_critics
        feat_dim = int(features_extractor.features_dim)
        self.q_belief = nn.ModuleList()
        self.q_img = nn.ModuleList()
        self.gates = nn.ModuleList()
        for _ in range(n_critics):
            self.q_belief.append(
                nn.Sequential(
                    *create_mlp(
                        input_dim=state_dim + action_dim,
                        output_dim=1,
                        net_arch=net_arch,
                        activation_fn=activation_fn,
                    )
                )
            )
            self.q_img.append(
                nn.Sequential(
                    *create_mlp(
                        input_dim=feat_dim + action_dim,
                        output_dim=1,
                        net_arch=net_arch,
                        activation_fn=activation_fn,
                    )
                )
            )
            gate = nn.Sequential(
                nn.Linear(state_dim, 64),
                activation_fn(),
                nn.Linear(64, 1),
            )
            nn.init.constant_(gate[-1].bias, float(gate_bias))
            self.gates.append(gate)

    def _state(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        state = obs["state"]
        if state.dim() > 2:
            state = state.reshape(state.shape[0], -1)
        return state

    def _q_values(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        img_feat = self.features_extractor(obs)
        state = self._state(obs)
        b_in = torch.cat([state, actions], dim=1)
        i_in = torch.cat([img_feat, actions], dim=1)
        outs = []
        gates_out = []
        for q_b, q_i, gate in zip(self.q_belief, self.q_img, self.gates):
            g = torch.sigmoid(gate(state))
            gates_out.append(g)
            outs.append(q_b(b_in) + g * q_i(i_in))
        self._last_gate = gates_out[0].detach()
        return tuple(outs)

    def forward(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self._q_values(obs, actions)

    def q1_forward(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        return self._q_values(obs, actions)[0]

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class GatedResidualSACPolicy(SACPolicy):
    """Actor=image; Critic=belief backbone + gated image residual."""

    def __init__(self, *args, **kwargs):
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor: Optional[nn.Module] = None) -> GatedImageBeliefCritic:
        if features_extractor is None:
            features_extractor = self.features_extractor_class(
                self.observation_space,
                **self.features_extractor_kwargs,
            )
        state_dim = int(self.observation_space["state"].shape[0])
        action_dim = get_action_dim(self.action_space)
        if isinstance(self.net_arch, dict):
            critic_arch = self.net_arch.get("qf", [512, 256])
        else:
            critic_arch = [512, 256]
        return GatedImageBeliefCritic(
            features_extractor=features_extractor,
            state_dim=state_dim,
            action_dim=action_dim,
            net_arch=list(critic_arch),
            activation_fn=self.activation_fn,
            n_critics=2,
        ).to(self.device)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)
