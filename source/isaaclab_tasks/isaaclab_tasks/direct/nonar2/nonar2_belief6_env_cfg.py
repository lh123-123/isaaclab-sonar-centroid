# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Belief6 + SAC — large observation noise, clamped -log(det) reward (/home/lh/1 style v18)."""

import gymnasium as gym
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .agents.policy_dims import BELIEF6_BEAM_OBS_DIM, COLUMN_PROFILE_OBS_DIM
from .nonar2_env_cfg import Nonar2EnvCfg


@configclass
class Nonar2Belief6EnvCfg(Nonar2EnvCfg):
    """8-d belief obs + large EKF R + clamped -log(det) reward."""

    policy_obs_mode: str = "belief6"
    sonar_history_len: int = 1
    observation_space = BELIEF6_BEAM_OBS_DIM
    belief_obs_log_sigma: bool = True
    belief6_include_det_azimuth: bool = False
    belief6_use_sin_cos_az: bool = True

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=12.0,
        replicate_physics=True,
    )

    # Discrete PPO: action_space handled by task registration
    # v_discrete: 21-d sonar column profile obs

    max_angular_velocity: float = 0.80
    max_dtheta: float = 0.08

    detection_conf_threshold: float = 0.20
    detection_peak_threshold: float = 0.08
    detection_peak_relative_min: float = 0.35
    detection_mean_threshold: float = 0.01
    detection_range_min: float = 1.0
    detection_range_max: float = 4.5
    #: beams: raw 图提 61 束 → 距门内取最强束（与 raw 肉眼对比一致）
    detection_mode: str = "beams"
    sonar_preprocess_for_detection: bool = False
    detection_col_contrast: bool = True
    detection_use_belief_prior: bool = False
    detection_prior_gate_deg: float = 35.0
    detection_prior_mu_norm_min: float = 0.35
    belief_process_noise: float = 0.05
    belief_lost_inflation: float = 1.01
    belief_lost_threshold: float = 2.0

    target_out_of_fov_grace_steps: int = 10
    tracking_bearing_max_deg: float = 40.0

    # v18: large EKF R + clamped -log(det) — /home/lh/1 style
    reward_mode: str = "covariance"
    r_facing_target: float = 0.0
    r_in_fov: float = 0.0
    r_out_of_fov: float = 0.0
    r_bearing_align: float = 0.0
    r_wrong_turn: float = 0.0
    r_det_center: float = 0.0
    reward_detect_positive: float = 0.0
    reward_detect_negative: float = 0.0
    reward_track_scale: float = 0.0
    r_reacquire: float = 0.0
    reward_detect_scale_by_facing: bool = False
    reward_detect_scale_by_det_az: bool = False
    align_use_velocity_hint: bool = False

    # v38: simple angle reward: +5.0 - 0.15*|az_deg|, miss=-5.0
    r_detect: float = 5.0
    r_deviation_k: float = 0.15
    r_miss: float = -5.0
    r_smooth: float = 0.010

    reward_scale: float = 1.0
    reward_clip_min: float = -15.0
    reward_clip_max: float = 10.0


@configclass
class Nonar2DiscretePpoCfg(Nonar2Belief6EnvCfg):
    """Discrete PPO: 3 actions (left/stay/right), 21-d sonar column profile obs."""

    policy_obs_mode: str = "column_profile"
    observation_space = COLUMN_PROFILE_OBS_DIM
    column_profile_stride: int = 3
    column_profile_contrast: bool = True

    #: Discrete PPO: peak reward when target is at boresight (0°).
    discrete_reward_center: float = 2.0
    #: Linear penalty per degree off center (GT body bearing).
    discrete_reward_center_k: float = 0.12
    #: Floor when detected but far off-center.
    discrete_reward_min: float = -0.5
    #: Reward when target not detected.
    discrete_reward_miss: float = -1.0

    action_space = gym.spaces.Discrete(3)

