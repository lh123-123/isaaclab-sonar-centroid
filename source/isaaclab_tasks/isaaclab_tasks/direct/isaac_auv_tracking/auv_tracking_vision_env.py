# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AUV tracking with Dict obs: sonar image (Actor) + belief state (Critic). Holo v1 style."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from isaaclab.utils import configclass

from .auv_tracking_env import AUVTrackingEnv, AUVTrackingEnvCfg


@configclass
class AUVTrackingVisionEnvCfg(AUVTrackingEnvCfg):
    """Same dynamics/reward/KF as v0; policy obs is Dict{image, state}."""

    # Holo v1 image size; polar resized into this canvas
    vision_image_size: tuple[int, int] = (64, 64)
    observation_space = {
        "image": gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
        "state": 4,
    }


class AUVTrackingVisionEnv(AUVTrackingEnv):
    cfg: AUVTrackingVisionEnvCfg

    def _polar_to_policy_image(self) -> torch.Tensor:
        """``pipeline.polar_image`` [N,H,W] → uint8 [N,3,64,64] for SB3 image path."""
        polar = self.pipeline.polar_image
        # Per-env normalize to [0,1] (avoid all-zero frames)
        flat_max = polar.amax(dim=(-2, -1), keepdim=True).clamp(min=1.0e-6)
        norm = (polar / flat_max).clamp(0.0, 1.0)
        h, w = int(self.cfg.vision_image_size[0]), int(self.cfg.vision_image_size[1])
        x = norm.unsqueeze(1)  # [N,1,H,W]
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = x.expand(-1, 3, -1, -1)
        return (x * 255.0).clamp(0.0, 255.0).to(torch.uint8)

    def _get_observations(self) -> dict:
        if self._needs_belief_refresh.any():
            refresh_ids = self._needs_belief_refresh.nonzero(as_tuple=False).squeeze(-1)
            self._update_belief(refresh_ids)
            self._needs_belief_refresh[refresh_ids] = False

        self.kf.predict()
        return {
            "policy": {
                "image": self._polar_to_policy_image(),
                "state": self._build_policy_obs(),
            }
        }
