# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Uint8 image replay with online n-step returns (vectorized envs)."""

from __future__ import annotations

from collections import deque

import numpy as np


class NStepReplayBuffer:
    """Stores n-step transitions built online per environment.

    Each stored item is ``(obs, action, nstep_reward, discount, next_obs)``.
    On episode end, remaining pending steps are flushed with ``discount=0``
    (no bootstrap past terminal).
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        batch_size: int = 256,
        nstep: int = 3,
        gamma: float = 0.99,
        num_envs: int = 1,
    ):
        self.capacity = int(capacity)
        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.action_dim = int(action_dim)
        self.batch_size = int(batch_size)
        self.nstep = int(nstep)
        self.gamma = float(gamma)
        self.num_envs = int(num_envs)

        self.obs = np.empty((self.capacity, *self.obs_shape), dtype=np.uint8)
        self.next_obs = np.empty((self.capacity, *self.obs_shape), dtype=np.uint8)
        self.actions = np.empty((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.discounts = np.empty((self.capacity, 1), dtype=np.float32)

        self._idx = 0
        self._size = 0
        self._pending: list[deque] = [deque() for _ in range(self.num_envs)]

    def __len__(self) -> int:
        return self._size

    def _store(self, obs, action, reward: float, discount: float, next_obs) -> None:
        self.obs[self._idx] = obs
        self.next_obs[self._idx] = next_obs
        self.actions[self._idx] = action
        self.rewards[self._idx, 0] = reward
        self.discounts[self._idx, 0] = discount
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _emit_one(self, q: deque, n: int, force_terminal: bool) -> None:
        r_sum = 0.0
        for k in range(n):
            r_sum += (self.gamma**k) * float(q[k]["reward"])
        first = q[0]
        last = q[n - 1]
        if force_terminal or bool(last["done"]):
            discount = 0.0
        else:
            discount = self.gamma**n
        self._store(first["obs"], first["action"], r_sum, discount, last["next_obs"])
        q.popleft()

    def add_batch(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        obs = np.asarray(obs)
        action = np.asarray(action, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs)
        done = np.asarray(done, dtype=bool).reshape(-1)
        n_env = obs.shape[0]
        while len(self._pending) < n_env:
            self._pending.append(deque())
        self.num_envs = max(self.num_envs, n_env)

        for i in range(n_env):
            q = self._pending[i]
            q.append(
                {
                    "obs": np.array(obs[i], copy=True),
                    "action": np.array(action[i], copy=True),
                    "reward": float(reward[i]),
                    "next_obs": np.array(next_obs[i], copy=True),
                    "done": bool(done[i]),
                }
            )
            if done[i]:
                while len(q) > 0:
                    n = min(self.nstep, len(q))
                    self._emit_one(q, n, force_terminal=True)
                q.clear()
            else:
                while len(q) >= self.nstep:
                    self._emit_one(q, self.nstep, force_terminal=False)

    def sample(self):
        assert self._size >= self.batch_size, "Not enough samples in replay buffer"
        idxs = np.random.randint(0, self._size, size=self.batch_size)
        return (
            self.obs[idxs].astype(np.float32),
            self.actions[idxs],
            self.rewards[idxs],
            self.discounts[idxs],
            self.next_obs[idxs].astype(np.float32),
        )
