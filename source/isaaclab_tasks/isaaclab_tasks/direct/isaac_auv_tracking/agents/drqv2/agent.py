# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# Portions Copyright (c) Facebook, Inc. and its affiliates (MIT) — DrQ-v2 agent.

"""DrQ-v2 agent for Direct-v1 sonar polar images (64x64)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        assert h == w, "RandomShiftsAug expects square images"
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift = shift * (2.0 / (h + 2 * self.pad))
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    """DrQ CNN encoder; ``repr_dim`` inferred for arbitrary HxW (e.g. 64x64)."""

    def __init__(self, obs_shape: tuple[int, int, int]):
        super().__init__()
        assert len(obs_shape) == 3
        c, h, w = int(obs_shape[0]), int(obs_shape[1]), int(obs_shape[2])
        self.convnet = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            self.repr_dim = int(self.convnet(dummy).reshape(1, -1).shape[1])
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        return h.reshape(h.shape[0], -1)


class Actor(nn.Module):
    def __init__(self, repr_dim: int, action_shape: tuple[int, ...], feature_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, std: float):
        h = self.trunk(obs)
        mu = torch.tanh(self.policy(h))
        std_t = torch.ones_like(mu) * std
        return utils.TruncatedNormal(mu, std_t)


class Critic(nn.Module):
    def __init__(self, repr_dim: int, action_shape: tuple[int, ...], feature_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        ad = action_shape[0]
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + ad, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + ad, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        return self.Q1(h_action), self.Q2(h_action)


class DrQV2Agent:
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_shape: tuple[int, ...],
        device: torch.device | str,
        lr: float = 1e-4,
        feature_dim: int = 50,
        hidden_dim: int = 1024,
        critic_target_tau: float = 0.01,
        num_expl_steps: int = 2000,
        update_every_steps: int = 2,
        stddev_schedule: str = "linear(1.0,0.1,500000)",
        stddev_clip: float = 0.3,
        use_tb: bool = True,
    ):
        self.device = torch.device(device)
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.action_shape = tuple(int(x) for x in action_shape)

        self.encoder = Encoder(self.obs_shape).to(self.device)
        self.actor = Actor(self.encoder.repr_dim, self.action_shape, feature_dim, hidden_dim).to(self.device)
        self.critic = Critic(self.encoder.repr_dim, self.action_shape, feature_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(self.encoder.repr_dim, self.action_shape, feature_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.aug = RandomShiftsAug(pad=4)
        self.train()
        self.critic_target.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step: int, eval_mode: bool = False):
        """``obs``: (C,H,W) or (N,C,H,W) uint8/float → actions (act_dim,) or (N, act_dim)."""
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, device=self.device)
        else:
            obs = obs.to(self.device)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        with torch.no_grad():
            feat = self.encoder(obs.float())
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(feat, stddev)
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)
        action = action.clamp(-1.0, 1.0).cpu().numpy()
        return action[0] if squeeze else action

    def update_critic(self, obs, action, reward, discount, next_obs, step: int) -> dict:
        metrics: dict = {}
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + discount * target_V

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, step: int) -> dict:
        metrics: dict = {}
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()
        return metrics

    def update(self, batch, step: int) -> dict:
        """``batch``: (obs, action, reward, discount, next_obs) numpy or tensors."""
        metrics: dict = {}
        if step % self.update_every_steps != 0:
            return metrics

        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))
        metrics.update(self.update_actor(obs.detach(), step))
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics

    def save(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "obs_shape": self.obs_shape,
            "action_shape": self.action_shape,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        saved_shape = tuple(ckpt.get("obs_shape", self.obs_shape))
        if saved_shape != self.obs_shape:
            raise ValueError(
                f"obs_shape mismatch: agent={self.obs_shape} ckpt={saved_shape}. "
                "Rebuild agent with matching --frame_stack / image size."
            )
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt.get("critic_target", ckpt["critic"]))
        return ckpt
