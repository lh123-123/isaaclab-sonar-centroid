# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO policy: sonar CNN + feed-forward actor/critic heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

class SonarCNN(nn.Module):
    """CNN encoder for sonar intensity images."""

    def __init__(self, image_height: int = 48, image_width: int = 96, feature_dim: int = 64, in_channels: int = 1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 4 * 8, feature_dim)
        self.activation = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.activation(x)


class Lh2SonarCnnActorCritic(nn.Module):
    """Policy encoder: ``[sonar(H*W), tail]`` -> CNN/MLP -> actor/critic heads.

    Supports two observation modes:
    - Image mode: flattened sonar image + tail.
    - Relative mode: low-dimensional vector (e.g. sonar-derived relative state).
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        # Floor on Gaussian std so PPO cannot collapse exploration to ~0 (log may show 0.00 if raw std rounds).
        min_std: float = 0.05,
        sonar_height: int = 32,
        sonar_width: int = 64,
        sonar_frame_stack: int = 1,
        cnn_feature_dim: int = 64,
        tail_mlp_hidden: int = 32,
        **kwargs,
    ):
        if kwargs:
            print(
                "Lh2SonarCnnActorCritic.__init__ ignoring unexpected keys: "
                f"{sorted(kwargs.keys())}"
            )
        super().__init__()

        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 256, 256]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 256, 256]

        self.sonar_h = sonar_height
        self.sonar_w = sonar_width
        self.sonar_frames = max(1, int(sonar_frame_stack))
        self.sonar_flat = sonar_height * sonar_width * self.sonar_frames
        self.vector_mode = num_actor_obs < self.sonar_flat
        if self.vector_mode:
            self.tail_dim = num_actor_obs
            self.base_tail_dim = num_actor_obs
        else:
            tail_dim = num_actor_obs - self.sonar_flat
            if tail_dim <= 0:
                raise ValueError(
                    f"num_actor_obs={num_actor_obs} incompatible with sonar_flat={self.sonar_flat}"
                )
            self.tail_dim = tail_dim
            self.base_tail_dim = tail_dim

        activation_fn = resolve_nn_activation(activation)

        if self.vector_mode:
            self.vector_encoder = nn.Sequential(
                nn.Linear(num_actor_obs, 128),
                activation_fn,
                nn.Linear(128, cnn_feature_dim),
                activation_fn,
            )
            fused_dim = cnn_feature_dim
        else:
            self.sonar_cnn = SonarCNN(
                image_height=sonar_height,
                image_width=sonar_width,
                feature_dim=cnn_feature_dim,
                in_channels=self.sonar_frames,
            )
            self.tail_mlp = nn.Sequential(
                nn.Linear(self.base_tail_dim, tail_mlp_hidden),
                activation_fn,
            )
            fused_dim = cnn_feature_dim + tail_mlp_hidden

        def build_mlp(input_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(activation_fn)
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation_fn)
            layers.append(nn.Linear(hidden_dims[-1], out_dim))
            return nn.Sequential(*layers)

        self.actor = build_mlp(fused_dim, actor_hidden_dims, num_actions)
        # Asymmetric critic: accepts privileged critic observations directly.
        self.critic_encoder = nn.Sequential(
            nn.Linear(num_critic_obs, 128),
            activation_fn,
            nn.Linear(128, cnn_feature_dim),
            activation_fn,
        )
        self.critic = build_mlp(cnn_feature_dim, critic_hidden_dims, 1)

        print(
            "Lh2SonarCnnActorCritic: "
            f"vector_mode={self.vector_mode}, sonar={self.sonar_frames}x{sonar_height}x{sonar_width}, "
            f"actor_obs_dim={num_actor_obs}, critic_obs_dim={num_critic_obs}, "
            f"tail_dim={self.tail_dim}, fused_dim={fused_dim}"
        )
        print(f"Actor MLP: {self.actor}")
        print(f"Critic encoder: {self.critic_encoder}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        self.min_std = float(min_std)
        if self.min_std < 0.0:
            raise ValueError(f"min_std must be >= 0, got {self.min_std}")
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)
        print(f"Lh2SonarCnnActorCritic: min_std={self.min_std} (policy std floor)")

    def _effective_std(self, std: torch.Tensor) -> torch.Tensor:
        if self.min_std <= 0.0:
            return std
        return torch.clamp(std, min=self.min_std)

    def _extract_features(self, observations: torch.Tensor) -> torch.Tensor:
        if self.vector_mode:
            return self.vector_encoder(observations)
        b = observations.shape[0]
        sonar = observations[:, : self.sonar_flat].contiguous().view(b, self.sonar_frames, self.sonar_h, self.sonar_w)
        tail_start = self.sonar_flat
        tail_end = tail_start + self.base_tail_dim
        tail_base = observations[:, tail_start:tail_end]

        sonar_feat = self.sonar_cnn(sonar)
        tail_feat = self.tail_mlp(tail_base)
        return torch.cat([sonar_feat, tail_feat], dim=-1)

    def _encode_for_rollout(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass for rollout/inference in feed-forward PPO mode."""
        return self._extract_features(observations)

    def _encode_for_training(self, observations: torch.Tensor) -> torch.Tensor:
        return self._extract_features(observations)

    def _encode_critic(self, critic_observations: torch.Tensor) -> torch.Tensor:
        return self.critic_encoder(critic_observations)

    def reset(self, dones=None):
        # Non-recurrent mode: no persistent hidden state to reset.
        return

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor(self._encode_for_training(observations))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        std = self._effective_std(std)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        mean = self.actor(self._encode_for_rollout(observations))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        std = self._effective_std(std)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor):
        return self.actor(self._encode_for_rollout(observations))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        return self.critic(self._encode_critic(critic_observations))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
