# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sonar-aware frame stack for Direct-v1 DrQ-v2.

Direct-v1 emits a 3×64×64 uint8 image that is a *replicated* single polar frame
(not true RGB). Stacking fake-RGB×K would waste channels. Instead we:

1. Take one polar channel (``[:, 0]``).
2. Keep the last ``k`` frames along the channel axis → ``(k, 64, 64)``.

This matches DrQ's temporal channel-stack idea while fitting OceanSim sonar.
"""

from __future__ import annotations

import numpy as np


def polar_frame_from_policy_image(images: np.ndarray) -> np.ndarray:
    """``(N, 3, H, W)`` or ``(N, 1, H, W)`` uint8 → ``(N, 1, H, W)`` polar."""
    img = np.asarray(images)
    if img.ndim != 4:
        raise ValueError(f"expected (N,C,H,W), got {img.shape}")
    if img.shape[1] >= 1:
        frame = img[:, 0:1]
    else:
        raise ValueError(f"bad channel dim {img.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


class SonarFrameStack:
    """Per-env temporal stack of sonar polar frames."""

    def __init__(self, num_envs: int, frame_stack: int = 3, height: int = 64, width: int = 64):
        self.num_envs = int(num_envs)
        self.k = int(frame_stack)
        self.h = int(height)
        self.w = int(width)
        if self.k < 1:
            raise ValueError("frame_stack must be >= 1")
        # (N, K, H, W)
        self._buf = np.zeros((self.num_envs, self.k, self.h, self.w), dtype=np.uint8)

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (self.k, self.h, self.w)

    def reset(self, images: np.ndarray, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Fill stack with the current polar frame (repeat). Returns stacked obs."""
        frame = polar_frame_from_policy_image(images)  # (N,1,H,W)
        if env_ids is None:
            for t in range(self.k):
                self._buf[:, t] = frame[:, 0]
        else:
            ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
            if frame.shape[0] != len(ids):
                raise ValueError("images rows must match env_ids length")
            for row, i in enumerate(ids):
                for t in range(self.k):
                    self._buf[int(i), t] = frame[row, 0]
        return self.get() if env_ids is None else self.get(env_ids)

    def update(self, images: np.ndarray) -> np.ndarray:
        """Push newest polar frame, drop oldest. ``images`` shape ``(N, C, H, W)``."""
        frame = polar_frame_from_policy_image(images)
        n = frame.shape[0]
        if n != self.num_envs:
            raise ValueError(f"num_envs mismatch: stack={self.num_envs} batch={n}")
        self._buf[:, :-1] = self._buf[:, 1:]
        self._buf[:, -1] = frame[:, 0]
        return self.get()

    def get(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        if env_ids is None:
            return self._buf.copy()
        ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        return self._buf[ids].copy()

    def on_dones(self, images: np.ndarray, dones: np.ndarray) -> None:
        """After a done, re-fill that env's stack with the *reset* observation frame."""
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        if not dones.any():
            return
        ids = np.where(dones)[0]
        # ``images`` is full (N,...) post-reset obs from VecEnv
        frame = polar_frame_from_policy_image(images)
        for i in ids:
            for t in range(self.k):
                self._buf[i, t] = frame[i, 0]
