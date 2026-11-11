# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Log gated-critic sigma(g(state)) to TensorBoard when the policy has gates."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeRewardLogCallback(BaseCallback):
    """Write episode reward scalars, and flush them to TensorBoard on episode end.

    ``train.py`` disables SB3's periodic flush (``log_interval=None``) and relies on
    ``LogEveryNTimesteps``, so a run that ends before the interval elapses leaves an empty
    event file — episodes are what actually fill ``rollout/ep_rew_mean``, so flushing when an
    episode finishes guarantees the curve appears as soon as the first episode is done.
    """

    def __init__(self, min_steps_between_dumps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.min_steps_between_dumps = int(min_steps_between_dumps)
        self._last_dump_timestep = -1

    def _on_step(self) -> bool:
        episodes_finished = 0
        for info in self.locals.get("infos", []):
            episode = info.get("episode") if isinstance(info, dict) else None
            if not episode:
                continue
            episodes_finished += 1
            # "r"/"l" are consumed by SB3's own dump_logs; the rest are reward-term means.
            for key, value in episode.items():
                if key in ("r", "l") or value is None:
                    continue
                try:
                    self.logger.record(f"Episode_Reward/{key}", float(value))
                except (TypeError, ValueError):
                    continue
        if episodes_finished == 0:
            return True
        if self.num_timesteps - self._last_dump_timestep < self.min_steps_between_dumps:
            return True
        self._last_dump_timestep = self.num_timesteps
        self.model.dump_logs()
        return True


class CriticGateLogCallback(BaseCallback):
    def __init__(self, log_freq: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = int(log_freq)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True
        policy = getattr(self.model, "policy", None)
        critic = getattr(policy, "critic", None) if policy is not None else None
        gate = getattr(critic, "_last_gate", None)
        if gate is None:
            return True
        g = gate.detach().float().reshape(-1)
        if g.numel() == 0:
            return True
        self.logger.record("train/critic_gate_mean", float(g.mean().cpu()))
        self.logger.record("train/critic_gate_std", float(g.std(unbiased=False).cpu()))
        return True
