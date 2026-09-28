# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Log gated-critic sigma(g(state)) to TensorBoard when the policy has gates."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


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
