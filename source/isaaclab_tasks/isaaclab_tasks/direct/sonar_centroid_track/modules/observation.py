# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Module M7: discrete PPO observation (4D image-only or 7D full)."""

from __future__ import annotations

import torch

from .constants import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    DEFAULT_LATERAL_AMP_M,
    POLAR_HEIGHT,
    POLAR_WIDTH,
    POLICY_OBS_DIM,
    POLICY_OBS_DIM_IMAGE,
    POLICY_OBS_DIM_IMAGE_RANGE,
    RANGE_ACCEPT_M,
)
from .coordinates import PolarPoint, normalize_uv
from .sonar_centroid_tracker import SonarCentroidTracker


class DiscreteObservationBuilder:
    """Build policy observation from image-detected centroid (4D or 7D)."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        lateral_amp_m: float = DEFAULT_LATERAL_AMP_M,
        target_period_s: float = 8.0,
        polar_width: int = POLAR_WIDTH,
        polar_height: int = POLAR_HEIGHT,
        image_only: bool = False,
        include_range_obs: bool = False,
        continuous_actions: bool = False,
    ):
        self.num_envs = num_envs
        self.device = device
        self.lateral_amp_m = float(lateral_amp_m)
        self.target_period_s = float(target_period_s)
        self.polar_width = int(polar_width)
        self.polar_height = int(polar_height)
        self.image_only = bool(image_only)
        self.include_range_obs = bool(include_range_obs)
        self.continuous_actions = bool(continuous_actions)
        if self.include_range_obs:
            self.obs_dim = POLICY_OBS_DIM_IMAGE_RANGE
        elif self.image_only:
            self.obs_dim = POLICY_OBS_DIM_IMAGE
        else:
            self.obs_dim = POLICY_OBS_DIM
        if self.continuous_actions:
            action_dim = 2 if self.include_range_obs else 1
            self._prev_action = torch.zeros(num_envs, action_dim, dtype=torch.float32, device=device)
        else:
            self._prev_action = torch.ones(num_envs, dtype=torch.long, device=device) * ACTION_STAY

    def set_prev_action(self, actions: torch.Tensor) -> None:
        if self.continuous_actions:
            a = actions.view(self.num_envs, -1).float()
            if a.shape[1] == 1 and self.include_range_obs:
                pad = torch.zeros(self.num_envs, 1, device=self.device)
                a = torch.cat([a, pad], dim=1)
                self._prev_action = a.clone()
            else:
                self._prev_action = a.clone()
        else:
            self._prev_action = actions.view(-1).long().clone()

    def build(
        self,
        tracker: SonarCentroidTracker,
        *,
        robot_sway_m: torch.Tensor,
        target_dy_m: torch.Tensor,
        target_dy_vel_m: torch.Tensor,
        range_ref_gt_m: torch.Tensor | None = None,
        min_range: float = 0.5,
        max_range: float = 15.0,
        range_accept_m: float = RANGE_ACCEPT_M,
        sonar_range_meas_m: torch.Tensor | None = None,
        sonar_range_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build policy obs. When ``include_range_obs``, slot 3 is
        ``clip((range_sonar − setpoint) / range_accept_m, ±1)`` from peak-beam only
        (no sim GT). ±accept_m is the OK band."""
        obs = torch.zeros(self.num_envs, self.obs_dim, device=self.device)
        amp = max(self.lateral_amp_m, 1.0e-3)
        vel_scale = max(amp / max(self.target_period_s, 1.0e-3), 1.0e-3)

        for i in range(self.num_envs):
            p_sonar = tracker.detected_rel(env_id=i)
            if p_sonar is not None:
                u_n, v_n = normalize_uv(p_sonar, self.polar_width, self.polar_height)
                obs[i, 0] = u_n
                obs[i, 1] = v_n
            obs[i, 2] = tracker.valid_ratio(env_id=i)

            if self.include_range_obs:
                d_range_norm = 0.0
                if (
                    range_ref_gt_m is not None
                    and sonar_range_meas_m is not None
                    and sonar_range_valid is not None
                    and float(sonar_range_valid[i].item()) >= 0.5
                ):
                    from .coordinates import range_err_norm

                    d_range_norm = range_err_norm(
                        float(sonar_range_meas_m[i].item()),
                        range_ref_gt_m[i].item(),
                        accept_m=range_accept_m,
                    )
                obs[i, 3] = d_range_norm
                if self.continuous_actions:
                    obs[i, 4] = self._prev_action[i, 0]
                    obs[i, 5] = self._prev_action[i, 1] if self._prev_action.shape[1] > 1 else 0.0
                continue

            if self.continuous_actions:
                prev = float(self._prev_action[i, 0].item()) if self._prev_action.ndim > 1 else float(
                    self._prev_action[i].item()
                )
            else:
                a = int(self._prev_action[i].item())
                prev = -1.0 if a == ACTION_LEFT else (1.0 if a == ACTION_RIGHT else 0.0)
            if self.image_only:
                obs[i, 3] = prev
            else:
                obs[i, 3] = robot_sway_m[i] / amp
                obs[i, 4] = target_dy_m[i] / amp
                obs[i, 5] = target_dy_vel_m[i] / vel_scale
                obs[i, 6] = prev

        return obs
