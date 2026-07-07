from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class Lh2FlsBeamActorCritic(nn.Module):
    """Beam 1D-CNN + MLP actor；非对称 critic（policy 维 + 机体 bearing 每步变化），输入维较大时首层加宽。"""

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
        min_std: float = 0.05,
        beam_count: int = 128,
        beam_channels: int = 3,
        beam_conv_channels: list[int] | None = None,
        beam_kernel_size: int = 5,
        beam_feature_dim: int = 64,
        tail_mlp_hidden: int = 32,
        **kwargs,
    ):
        if kwargs:
            _ = kwargs
        super().__init__()

        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128, 64]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 128, 64]
        if beam_conv_channels is None:
            beam_conv_channels = [32, 64]

        self.beam_count = int(beam_count)
        self.beam_channels = int(beam_channels)
        self.beam_flat = self.beam_count * self.beam_channels
        if num_actor_obs <= self.beam_flat:
            raise ValueError(
                f"num_actor_obs={num_actor_obs} must be > beam_flat={self.beam_flat} (need beam + tail)."
            )
        self.tail_dim = num_actor_obs - self.beam_flat
        activation_fn = resolve_nn_activation(activation)

        conv_layers: list[nn.Module] = []
        in_ch = self.beam_channels
        k = int(max(1, beam_kernel_size))
        pad = k // 2
        for out_ch in beam_conv_channels:
            conv_layers.append(nn.Conv1d(in_ch, int(out_ch), kernel_size=k, padding=pad))
            conv_layers.append(activation_fn)
            in_ch = int(out_ch)
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.beam_conv = nn.Sequential(*conv_layers)
        self.beam_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, beam_feature_dim),
            activation_fn,
        )
        self.tail_mlp = nn.Sequential(
            nn.Linear(self.tail_dim, tail_mlp_hidden),
            activation_fn,
        )

        def build_mlp(input_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), activation_fn]
            for i in range(len(hidden_dims) - 1):
                layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), activation_fn])
            layers.append(nn.Linear(hidden_dims[-1], out_dim))
            return nn.Sequential(*layers)

        fused_dim = beam_feature_dim + tail_mlp_hidden
        self.actor = build_mlp(fused_dim, actor_hidden_dims, num_actions)

        self.critic_encoder = nn.Sequential(
            nn.Linear(num_critic_obs, 256),
            activation_fn,
            nn.Linear(256, beam_feature_dim),
            activation_fn,
        )
        self.critic = build_mlp(beam_feature_dim, critic_hidden_dims, 1)

        self.noise_std_type = noise_std_type
        self.min_std = float(min_std)
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)
        _ = (num_actor_obs, num_critic_obs)

    def _effective_std(self, std: torch.Tensor) -> torch.Tensor:
        if self.min_std <= 0.0:
            return std
        return torch.clamp(std, min=self.min_std)

    def _encode_actor(self, observations: torch.Tensor) -> torch.Tensor:
        b = observations.shape[0]
        beam = observations[:, : self.beam_flat].contiguous().view(b, self.beam_count, self.beam_channels)
        beam = beam.transpose(1, 2)  # [B, C, beam_count]
        tail = observations[:, self.beam_flat :]
        beam_feat = self.beam_proj(self.beam_conv(beam))
        tail_feat = self.tail_mlp(tail)
        return torch.cat([beam_feat, tail_feat], dim=-1)

    def _encode_critic(self, critic_observations: torch.Tensor) -> torch.Tensor:
        return self.critic_encoder(critic_observations)

    def reset(self, dones=None):
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
        mean = self.actor(self._encode_actor(observations))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        self.distribution = Normal(mean, self._effective_std(std))

    def act(self, observations: torch.Tensor, **kwargs):
        mean = self.actor(self._encode_actor(observations))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        self.distribution = Normal(mean, self._effective_std(std))
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor):
        return self.actor(self._encode_actor(observations))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        return self.critic(self._encode_critic(critic_observations))

