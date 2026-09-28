# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AUV tracking Direct-v2: stacked polar residual + reference polar, v0 reward/done."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import gymnasium as gym

from isaaclab.utils import configclass

from .auv_tracking_env import AUVTrackingEnv, AUVTrackingEnvCfg
from .flow_obs import (
    FLOW_REF_BEARING_RAD,
    FLOW_REF_RANGE_M,
    FLOW_STATE_DIM,
    build_flow_state,
    centroid_displacement,
    normalize_polar_fixed,
    pack_flow_image,
    photometric_residual,
)
from .sonar_meas import detect_centroid_polar


@configclass
class AUVTrackingFlowEnvCfg(AUVTrackingEnvCfg):
    """Same dynamics/KF as v0; policy obs is Dict{flow image, geometry}.

    Reward adds two terms on top of v0:
      - rew_scale_speed:  penalize AUV speed ≠ target speed (suppress orbiting)
      - rew_scale_ref:    reward sonar centroid range ≈ r* (stable distance)
    """

    vision_image_size: tuple[int, int] = (64, 64)
    flow_ref_range_m: float = FLOW_REF_RANGE_M
    flow_ref_bearing_rad: float = FLOW_REF_BEARING_RAD
    # v0 KF used 0.25; that mask includes the dim far-range tail and pulls the
    # intensity centroid ~0.2 m past the bright core / sphere center. Flow state
    # (r, α) and v2 KF share this detector — use the library default 0.35.
    sonar_threshold_ratio: float = 0.35
    # OceanSim raw sonar is clamped to [0,1]; polar is already log1p-compressed
    # so its max is log1p(1.0)=0.693. Dividing by log1p(50)=3.93 (sonar_flow_hover's
    # ceiling for raw~50 sensors) crushes the image to [0,0.176] → nearly black.
    flow_polar_log_max: float = float(np.log1p(1.0))
    observation_space = {
        "image": gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
        "state": FLOW_STATE_DIM,
    }
    # --- v2 reward additions (on top of v0 uncertainty reward) ---
    # Speed match: exp(-|v_auv - v_tgt|² / sigma²). 0 = off (v0 default).
    # Sweep winner: 0.2 (best speed_err / tot / gt_r_std vs 0.1/0.0/0.3).
    rew_scale_speed: float = 0.2
    # Reference distance: exp(-(r_centroid - r*)² / sigma²), gated by valid detect.
    # Sweep: keep 0.1; 0.2 over-penalizes and destabilizes range.
    rew_scale_ref: float = 0.1
    rew_ref_sigma: float = 0.5
    # Early terminate: if target out of sonar FOV/range for this many consecutive
    # steps, reset the episode (don't wait for 100-step rolling reward threshold).
    lost_target_reset_steps: int = 30


class AUVTrackingFlowEnv(AUVTrackingEnv):
    cfg: AUVTrackingFlowEnvCfg

    def __init__(self, cfg: AUVTrackingFlowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if self.pipeline is None:
            raise RuntimeError(
                "Isaac-AUV-Tracking-Direct-v2 needs OceanSim polar frames "
                "(set kf_measurement='sonar')."
            )
        n = self.num_envs
        self._prev_polar = torch.zeros_like(self.pipeline.polar_image)
        self._prev_r = torch.zeros(n, device=self.device)
        self._prev_a = torch.zeros(n, device=self.device)
        self._prev_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._flow_ready = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._flow_r = torch.zeros(n, device=self.device)
        self._flow_a = torch.zeros(n, device=self.device)
        self._flow_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._flow_dr = torch.zeros(n, device=self.device)
        self._flow_dalpha = torch.zeros(n, device=self.device)
        self._flow_valid_flow = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._flow_image = torch.zeros(
            n, 3, int(cfg.vision_image_size[0]), int(cfg.vision_image_size[1]),
            device=self.device, dtype=torch.uint8,
        )
        self._flow_state = torch.zeros(n, FLOW_STATE_DIM, device=self.device)
        self._out_of_fov_count = torch.zeros(n, dtype=torch.long, device=self.device)

    def _detect_flow_centroid(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ocfg = self.cfg.oceansim_sonar_cfg
        r, a, valid = detect_centroid_polar(
            self.pipeline.polar_image,
            self.gt,
            prior_azimuth=None,
            prior_range=None,
            prior_gate_deg=float(self.cfg.sonar_prior_gate_deg),
            hori_fov_deg=float(ocfg.hori_fov),
            min_range=float(ocfg.min_range),
            max_range=float(ocfg.max_range),
            range_window_m=float(self.cfg.sonar_range_window_m),
            detect_max_range=float(self.cfg.sensor_range),
        )
        min_peak = float(self.cfg.sonar_min_peak)
        if min_peak > 0.0:
            peak = self.pipeline.polar_image.amax(dim=(-2, -1))
            valid = valid * (peak >= min_peak).to(dtype=valid.dtype)
        return r, a, valid > 0.5

    def _refresh_flow_obs(self) -> None:
        self._ensure_sonar()
        polar = self.pipeline.polar_image
        curr_01 = normalize_polar_fixed(polar, float(self.cfg.flow_polar_log_max))
        prev_01 = normalize_polar_fixed(self._prev_polar, float(self.cfg.flow_polar_log_max))
        residual = photometric_residual(curr_01, prev_01, self._flow_ready)
        self._flow_image = pack_flow_image(curr_01, residual, size=self.cfg.vision_image_size)

        r, a, valid = self._detect_flow_centroid()
        dr, dalpha, flow_valid = centroid_displacement(
            r, a, valid, self._prev_r, self._prev_a, self._prev_valid, self._flow_ready
        )
        r_star = torch.full((self.num_envs,), float(self.cfg.flow_ref_range_m), device=self.device)
        a_star = torch.full((self.num_envs,), float(self.cfg.flow_ref_bearing_rad), device=self.device)
        self._flow_state = build_flow_state(r, a, valid, r_star, a_star, dr, dalpha, flow_valid)

        self._flow_r = r
        self._flow_a = a
        self._flow_valid = valid
        self._flow_dr = dr
        self._flow_dalpha = dalpha
        self._flow_valid_flow = flow_valid

        self._prev_polar = polar.detach().clone()
        self._prev_r = r.detach().clone()
        self._prev_a = a.detach().clone()
        self._prev_valid = valid.detach().clone()
        self._flow_ready[:] = True

    def _get_observations(self) -> dict:
        if self._needs_belief_refresh.any():
            refresh_ids = self._needs_belief_refresh.nonzero(as_tuple=False).squeeze(-1)
            self._update_belief(refresh_ids)
            self._needs_belief_refresh[refresh_ids] = False

        self.kf.predict()
        self._refresh_flow_obs()
        self.extras["log_det_cov"] = self._log_det_cov().mean()
        self.extras["flow_valid_frac"] = self._flow_valid_flow.float().mean()
        return {
            "policy": {
                "image": self._flow_image,
                "state": self._flow_state,
            }
        }

    def _compute_step_reward(self) -> torch.Tensor:
        """v0 uncertainty reward + speed match + r* reference distance."""
        rew = super()._compute_step_reward()
        # r* reference: reward sonar centroid range ≈ r* (only when valid)
        scale_ref = float(self.cfg.rew_scale_ref)
        if scale_ref > 0.0 and hasattr(self, "_flow_r"):
            r_star = float(self.cfg.flow_ref_range_m)
            sigma_r = max(float(self.cfg.rew_ref_sigma), 1.0e-6)
            r_err = (self._flow_r - r_star) ** 2
            r_ref = torch.exp(-r_err / (sigma_r * sigma_r))
            r_ref = r_ref * self._flow_valid.float()  # gate: no reward when lost
            rew = rew + scale_ref * r_ref
        return torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=-50.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """v0 done + early terminate on sustained target loss."""
        done_by_reward, time_out = super()._get_dones()

        # Check if target is in sonar FOV/range (geometric GT, terminal condition only)
        agent_pos, agent_yaw = self._agent_pose_2d()
        _, gt_r, gt_a = self.kf.get_observation(self._target_pos, agent_pos, agent_yaw)
        in_fov = (gt_r <= float(self.cfg.sensor_range)) & (torch.abs(gt_a) <= self._fov_half)

        # Count consecutive out-of-FOV steps
        self._out_of_fov_count = torch.where(
            in_fov, torch.zeros_like(self._out_of_fov_count), self._out_of_fov_count + 1
        )
        lost_too_long = self._out_of_fov_count >= int(self.cfg.lost_target_reset_steps)
        done_by_reward = done_by_reward | lost_too_long

        return done_by_reward, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        if not hasattr(self, "_flow_ready"):
            return
        self._flow_ready[env_ids] = False
        self._prev_polar[env_ids] = 0.0
        self._prev_valid[env_ids] = False
        self._prev_r[env_ids] = 0.0
        self._prev_a[env_ids] = 0.0
        self._flow_valid[env_ids] = False
        self._flow_valid_flow[env_ids] = False
        self._out_of_fov_count[env_ids] = 0
        self._flow_dr[env_ids] = 0.0
        self._flow_dalpha[env_ids] = 0.0


# ---------------------------------------------------------------------------
# Ablation variants: same env, same reward, different observation channel.
# ---------------------------------------------------------------------------


@configclass
class AUVTrackingFlowStateEnvCfg(AUVTrackingFlowEnvCfg):
    """State-only ablation: 8-D flow state, no image, no CNN."""

    observation_space = FLOW_STATE_DIM


class AUVTrackingFlowStateEnv(AUVTrackingFlowEnv):
    """Returns only the flow state vector (no image). MLP-only policy."""

    cfg: AUVTrackingFlowStateEnvCfg

    def _get_observations(self) -> dict:
        if self._needs_belief_refresh.any():
            refresh_ids = self._needs_belief_refresh.nonzero(as_tuple=False).squeeze(-1)
            self._update_belief(refresh_ids)
            self._needs_belief_refresh[refresh_ids] = False
        self.kf.predict()
        self._refresh_flow_obs()
        self.extras["log_det_cov"] = self._log_det_cov().mean()
        self.extras["flow_valid_frac"] = self._flow_valid_flow.float().mean()
        return {"policy": self._flow_state}


@configclass
class AUVTrackingFlowImageEnvCfg(AUVTrackingFlowEnvCfg):
    """Image-only ablation: 3×64×64 flow image, no state vector, no MLP concat."""

    observation_space = {
        "image": gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
    }


class AUVTrackingFlowImageEnv(AUVTrackingFlowEnv):
    """Returns only the flow image (no state). CNN-only policy."""

    cfg: AUVTrackingFlowImageEnvCfg

    def _get_observations(self) -> dict:
        if self._needs_belief_refresh.any():
            refresh_ids = self._needs_belief_refresh.nonzero(as_tuple=False).squeeze(-1)
            self._update_belief(refresh_ids)
            self._needs_belief_refresh[refresh_ids] = False
        self.kf.predict()
        self._refresh_flow_obs()
        self.extras["log_det_cov"] = self._log_det_cov().mean()
        self.extras["flow_valid_frac"] = self._flow_valid_flow.float().mean()
        return {"policy": {"image": self._flow_image}}
