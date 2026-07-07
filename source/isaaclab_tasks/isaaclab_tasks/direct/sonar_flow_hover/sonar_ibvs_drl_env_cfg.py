# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for sonar IBVS + DRL dynamic target tracking and hover."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.nonar2.oceansim_sonar_sensor_cfg import OceanSimSonarSensorCfg

from .shared_platform import (
    BEACON2_CFG,
    BEACON_CFG,
    OCEANSIM_SONAR_CFG,
    ROBOT_CFG,
    SCENE_CFG,
    SIM_CFG,
    SONAR_MOUNT_OFFSET_LOCAL,
    VIEWER_CFG,
    WORKSPACE_HEIGHT,
)

from .modules.constants import (
    BODY_STATE_DIM,
    LATERAL_1D_ACTION_DIM,
    LATERAL_1D_COMPACT_OBS_DIM,
    LATERAL_1D_OBS_DIM,
    MAX_TRACK_POINTS,
    MIN_VALID_FEATURES,
    POLICY_ACTION_DIM,
    POLICY_OBS_DIM,
)

# 20 points × 2 errors + 20 valid flags + 8 body + 4 prev action = 72
_MAX_TRACK_POINTS = MAX_TRACK_POINTS
_BODY_STATE_DIM = BODY_STATE_DIM
_POLICY_ACTION_DIM = POLICY_ACTION_DIM
_POLICY_OBS_DIM = POLICY_OBS_DIM


@configclass
class SonarIbvsDrlEnvCfg(DirectRLEnvCfg):
    """AUV sonar sparse-feature IBVS + DRL (SAC)."""

    decimation = 5
    episode_length_s = 30.0

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(_POLICY_ACTION_DIM,), dtype=np.float32)
    observation_space = _POLICY_OBS_DIM
    state_space = 0

    sim = SIM_CFG
    scene: InteractiveSceneCfg = SCENE_CFG.replace(num_envs=4)
    viewer = VIEWER_CFG
    workspace_height: float = WORKSPACE_HEIGHT

    robot_cfg: RigidObjectCfg = ROBOT_CFG
    beacon_cfg: RigidObjectCfg = BEACON_CFG
    use_second_beacon: bool = False
    beacon2_cfg: RigidObjectCfg = BEACON2_CFG
    beacon2_offset_local: tuple[float, float, float] = (3.0, 0.7, 0.0)

    station_origin_local: tuple[float, float, float] = (0.0, 0.0, WORKSPACE_HEIGHT)
    beacon_offset_local: tuple[float, float, float] = (3.0, 0.0, 0.0)
    spawn_radius: float = 0.2
    eval_spawn_radius: float = 0.2
    spawn_yaw_range_deg: float = 5.0

    reference_snapshot_path: str = "logs/station_ref/station_ref.npz"
    reference_mode: str = "file"  # file | capture | geometry

    sonar_mount_offset_local: tuple[float, float, float] = SONAR_MOUNT_OFFSET_LOCAL
    oceansim_sonar_cfg: OceanSimSonarSensorCfg = OCEANSIM_SONAR_CFG
    cart_image_size: int = 128
    max_extract_features: int = 30
    max_track_features: int = _MAX_TRACK_POINTS
    feature_roi_radius: float = 40.0
    # Polar teach/extract: ellipse ROI + row/intensity gates (61x110 typical)
    feature_roi_radius_polar_az: float = 14.0
    feature_roi_radius_polar_range: float = 18.0
    feature_polar_row_band: float = 20.0
    feature_polar_min_intensity_ratio: float = 0.35
    fb_error_threshold: float = 1.0
    feature_patch_half_size: int = 6
    feature_relocate_window: int = 18
    feature_relocate_min_score: float = 0.55
    feature_ref_min_self_score: float = 0.72
    feature_ref_min_peak_margin: float = 0.10
    feature_track_min_peak_margin: float = 0.03
    feature_tracking_mode: str = "lk"  # patch | lk
    # cart: IBVS on Cartesian sonar; polar: track on polar (Δaz / Δrange)
    perception_frame: str = "cart"  # cart | polar
    keep_cart_for_viz: bool = True
    skip_median_filter: bool = False
    feature_contour_threshold_ratio: float = 0.25
    feature_contour_min_area: float = 4.0

    max_lin_vel: float = 0.5
    max_heave_vel: float = 0.3
    max_ang_vel: float = 0.5

    min_depth: float = 1.0
    max_depth: float = 20.0
    min_valid_features: int = MIN_VALID_FEATURES

    freeze_robot_pose: bool = False
    debug_disable_lost_terminate: bool = False
    eval_mode: bool = False

    use_boundaries: bool = True
    max_robot_x: float = 5.0
    max_robot_y: float = 5.0
    max_robot_z: float = 2.0

    obs_ang_vel_scale: float = 1.0
    obs_depth_scale: float = 10.0
    obs_depth_rate_scale: float = 1.0

    rew_w1_pixel: float = 1.0
    rew_w2_action: float = 0.1
    rew_w3_safety: float = 5.0

    current_speed_max: float = 0.0
    target_speed_max: float = 0.0

    # When True: only body-Y (sway) for AUV and target; surge/heave/yaw locked.
    lateral_1d_mode: bool = False
    lateral_1d_obs_mode: str = "full"  # compact (4-d centroid) | full (legacy)
    # Subtract teach-vs-geometry offset at station origin so zero = correct track.
    geometric_centroid_bias_correct: bool = True
    # lateral 1D pixel reward: |signed err| (default) or signed_err^2
    lateral_reward_pixel_sq: bool = False
    lateral_target_amp_m: float = 1.5
    lateral_target_period_s: float = 8.0

    curriculum_stage: int = 1
    spawn_use_teach_yaw: bool = True

    print_ibvs_debug: bool = False
    ibvs_debug_interval: int = 50
    # Aliases for legacy scripts / play.py --flow_debug
    print_flow_debug: bool = False
    flow_debug_interval: int = 50


@configclass
class SonarIbvsDrlLateral1DCfg(SonarIbvsDrlEnvCfg):
    """1D lateral train: 4-d centroid-error obs, 1-d sway; pure sonar IBVS."""

    lateral_1d_mode: bool = True
    lateral_1d_obs_mode: str = "compact"  # compact (7-d) | full (69-d legacy)
    perception_frame: str = "polar"
    keep_cart_for_viz: bool = False
    skip_median_filter: bool = True
    lateral_target_moving: bool = True
    max_extract_features: int = 4
    max_track_features: int = 4
    min_valid_features: int = 2
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(LATERAL_1D_ACTION_DIM,), dtype=np.float32)
    observation_space = LATERAL_1D_COMPACT_OBS_DIM
    spawn_radius: float = 0.15
    spawn_yaw_range_deg: float = 0.0
    current_speed_max: float = 0.0
    target_speed_max: float = 0.0
    # Peak target lateral speed ≈ amp * 2π/T (1.5m, 8s → ~1.18 m/s); robot must match.
    max_lin_vel: float = 2.0
    episode_length_s: float = 60.0
    freeze_robot_pose: bool = False
    lost_feature_terminate_steps: int = 80
    rew_w1_pixel: float = 2.0 / 61.0  # |lateral centroid error| in px (was norm * 2.0)
    rew_w2_action: float = 0.05
    rew_w3_safety: float = 1.0
    scene: InteractiveSceneCfg = SCENE_CFG.replace(num_envs=8)


@configclass
class SonarIbvsDrlLateral1DDebugCfg(SonarIbvsDrlLateral1DCfg):
    """GUI debug for 1D lateral following (both AUV and target move in sway)."""

    lateral_target_moving: bool = False
    perception_frame: str = "polar"
    lateral_1d_obs_mode: str = "compact"
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(LATERAL_1D_ACTION_DIM,), dtype=np.float32)
    observation_space = LATERAL_1D_COMPACT_OBS_DIM
    eval_mode: bool = True
    scene: InteractiveSceneCfg = SCENE_CFG.replace(num_envs=1)
    episode_length_s: float = 3600.0
    use_boundaries: bool = False
    eval_spawn_radius: float = 0.0
    spawn_radius: float = 0.0
    debug_disable_lost_terminate: bool = True
    reference_mode: str = "file"
    print_ibvs_debug: bool = True
    ibvs_debug_interval: int = 10


@configclass
class SonarIbvsDrlStage1Cfg(SonarIbvsDrlEnvCfg):
    """Curriculum stage 1: static target, small initial offset, no disturbance."""

    curriculum_stage: int = 1
    spawn_radius: float = 0.2
    spawn_yaw_range_deg: float = 5.0
    current_speed_max: float = 0.0
    target_speed_max: float = 0.0
    episode_length_s: float = 50.0


@configclass
class SonarIbvsDrlStage2Cfg(SonarIbvsDrlEnvCfg):
    curriculum_stage: int = 2
    spawn_radius: float = 1.5
    spawn_yaw_range_deg: float = 20.0
    current_speed_max: float = 0.1
    target_speed_max: float = 0.0


@configclass
class SonarIbvsDrlEvalCfg(SonarIbvsDrlEnvCfg):
    eval_mode: bool = True
    scene: InteractiveSceneCfg = SCENE_CFG.replace(num_envs=1)
    eval_spawn_radius: float = 0.5
    print_ibvs_debug: bool = True


@configclass
class SonarIbvsDrlDebugCfg(SonarIbvsDrlEnvCfg):
    """GUI debug: frozen robot at teach pose for feature extraction."""

    eval_mode: bool = True
    scene: InteractiveSceneCfg = SCENE_CFG.replace(num_envs=1)
    episode_length_s: float = 3600.0
    use_boundaries: bool = False
    eval_spawn_radius: float = 0.0
    spawn_radius: float = 0.0
    freeze_robot_pose: bool = True
    debug_disable_lost_terminate: bool = True
    reference_mode: str = "capture"
    print_ibvs_debug: bool = True
    ibvs_debug_interval: int = 10


# Backward-compatible aliases
SonarFlowHoverEnvCfg = SonarIbvsDrlEnvCfg
SonarFlowHoverTemplateCfg = SonarIbvsDrlEnvCfg
SonarFlowHoverEvalCfg = SonarIbvsDrlEvalCfg
SonarFlowHoverTemplateEvalCfg = SonarIbvsDrlEvalCfg
SonarFlowHoverDebugCfg = SonarIbvsDrlDebugCfg
SonarFlowHoverSweepCfg = SonarIbvsDrlDebugCfg
