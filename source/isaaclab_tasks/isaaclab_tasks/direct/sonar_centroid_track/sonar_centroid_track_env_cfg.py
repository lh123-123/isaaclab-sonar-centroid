# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for sonar centroid tracking (discrete PPO, polar-only)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.nonar2.oceansim_sonar_sensor_cfg import OceanSimSonarSensorCfg

from .modules.constants import (
    DEFAULT_FB_ERROR_THRESHOLD,
    DEFAULT_LATERAL_AMP_M,
    DEFAULT_MAX_LATERAL_VEL,
    DEFAULT_THRESHOLD_RATIO,
    MIN_VALID_RATIO,
    NUM_DISCRETE_ACTIONS,
    POLICY_ACTION_DIM,
    POLICY_OBS_DIM,
    POLICY_OBS_DIM_IMAGE,
)
from .shared_platform import (
    BEACON_CFG,
    OCEANSIM_SONAR_CFG,
    ROBOT_CFG,
    SCENE_CFG,
    SIM_CFG,
    SONAR_MOUNT_OFFSET_LOCAL,
    VIEWER_CFG,
    WORKSPACE_HEIGHT,
)
from .modules.target_motion import sine_cycle_s

_PPO_AMP_M = 1.3
_PPO_TARGET_VY_PEAK = 0.28 * 1.5  # 1.5× triangle leg speed (0.28 → 0.42 m/s peak)
_PPO_SINE_PERIOD_S = sine_cycle_s(_PPO_AMP_M, _PPO_TARGET_VY_PEAK)
_PPO_ROBOT_MAX_VY = _PPO_TARGET_VY_PEAK * 1.25  # must exceed ball peak (0.525 m/s)


@configclass
class SonarCentroidTrackEnvCfg(DirectRLEnvCfg):
    """Polar centroid tracking with fixed-point origin and discrete sway actions."""

    decimation = 5
    episode_length_s = 60.0

    action_space = gym.spaces.Discrete(NUM_DISCRETE_ACTIONS)
    observation_space = POLICY_OBS_DIM
    state_space = 0

    sim = SIM_CFG
    scene: InteractiveSceneCfg = SCENE_CFG.replace(num_envs=4)
    viewer = VIEWER_CFG
    workspace_height: float = WORKSPACE_HEIGHT

    robot_cfg: RigidObjectCfg = ROBOT_CFG
    beacon_cfg: RigidObjectCfg = BEACON_CFG

    station_origin_local: tuple[float, float, float] = (0.0, 0.0, WORKSPACE_HEIGHT)
    beacon_offset_local: tuple[float, float, float] = (3.0, 0.0, 0.0)
    spawn_radius: float = 0.15
    eval_spawn_radius: float = 0.0
    spawn_yaw_range_deg: float = 0.0

    sonar_mount_offset_local: tuple[float, float, float] = SONAR_MOUNT_OFFSET_LOCAL
    sonar_mount_rpy_deg: tuple[float, float, float] | None = None
    sonar_mount_visual: bool = False
    sonar_housing_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    oceansim_sonar_cfg: OceanSimSonarSensorCfg = OCEANSIM_SONAR_CFG

    skip_median_filter: bool = True
    sonar_warmup_frames: int = 4

    fixed_point_snapshot_path: str = "logs/centroid_track/fixed_point.npz"
    auto_calibrate_fixed_point: bool = True
    fixed_point_calib_settle_steps: int = 8

    max_track_features: int = 12
    max_extract_features: int = 16
    feature_roi_radius: float = 40.0
    feature_roi_radius_polar_az: float = 14.0
    feature_roi_radius_polar_range: float = 16.0
    feature_polar_row_band: float = 18.0
    feature_polar_min_intensity_ratio: float = 0.20
    fb_error_threshold: float = 1.2
    feature_polar_threshold_ratio: float = DEFAULT_THRESHOLD_RATIO

    max_lateral_velocity: float = DEFAULT_MAX_LATERAL_VEL
    lateral_target_amp_m: float = DEFAULT_LATERAL_AMP_M
    lateral_target_period_s: float = 8.0
    lateral_target_vy: float = 0.28
    lateral_target_mode: str = "sine"
    lateral_target_moving: bool = False
    image_only_obs: bool = False
    continuous_actions: bool = False

    min_valid_ratio: float = MIN_VALID_RATIO
    min_depth: float = 1.0
    max_depth: float = 20.0
    max_robot_x: float = 8.0
    max_robot_y: float = 8.0
    max_robot_z: float = 3.0
    use_boundaries: bool = True
    lost_track_terminate_steps: int = 80
    immediate_lost_terminate: bool = False
    debug_disable_lost_terminate: bool = False

    freeze_robot_pose: bool = False
    eval_mode: bool = False
    print_centroid_debug: bool = False
    centroid_debug_interval: int = 20
    print_policy_obs_debug: bool = False
    policy_obs_debug_interval: int = 10

    # WarpAUV integration (see sonar_centroid_track_auv_env.py).
    auv_physics_mode: bool = False


@configclass
class SonarCentroidTrackDebugCfg(SonarCentroidTrackEnvCfg):
    """GUI / Phase 1 debug: frozen robot, static target, verbose logging."""

    scene: InteractiveSceneCfg = SCENE_CFG.replace(num_envs=1)
    episode_length_s = 3600.0
    eval_mode: bool = True
    spawn_radius: float = 0.0
    eval_spawn_radius: float = 0.0
    freeze_robot_pose: bool = True
    lateral_target_moving: bool = False
    use_boundaries: bool = False
    debug_disable_lost_terminate: bool = True
    print_centroid_debug: bool = True
    centroid_debug_interval: int = 10


@configclass
class SonarCentroidTrackDiscretePpoCfg(SonarCentroidTrackEnvCfg):
    """Discrete PPO: image-only obs, sine target 0→+A→0→−A→0 (one period)."""

    scene: InteractiveSceneCfg = SCENE_CFG.replace(num_envs=8)
    observation_space = POLICY_OBS_DIM_IMAGE
    image_only_obs: bool = True
    lateral_target_moving: bool = True
    lateral_target_mode: str = "sine"
    lateral_target_amp_m: float = _PPO_AMP_M
    lateral_target_period_s: float = _PPO_SINE_PERIOD_S
    lateral_target_vy: float = _PPO_TARGET_VY_PEAK
    max_lateral_velocity: float = _PPO_ROBOT_MAX_VY
    spawn_radius: float = 0.10
    episode_length_s: float = _PPO_SINE_PERIOD_S
    immediate_lost_terminate: bool = True
    lost_track_terminate_steps: int = 1
    debug_disable_lost_terminate: bool = False


@configclass
class SonarCentroidTrackContinuousPpoCfg(SonarCentroidTrackEnvCfg):
    """Continuous PPO: Box(-1,1) vy, image-only 4D obs, sine target."""

    scene: InteractiveSceneCfg = SCENE_CFG.replace(num_envs=8)
    continuous_actions: bool = True
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(POLICY_ACTION_DIM,), dtype=np.float32
    )
    observation_space = POLICY_OBS_DIM_IMAGE
    image_only_obs: bool = True
    lateral_target_moving: bool = True
    lateral_target_mode: str = "sine"
    lateral_target_amp_m: float = _PPO_AMP_M
    lateral_target_period_s: float = _PPO_SINE_PERIOD_S
    lateral_target_vy: float = _PPO_TARGET_VY_PEAK
    max_lateral_velocity: float = _PPO_ROBOT_MAX_VY
    spawn_radius: float = 0.10
    episode_length_s: float = _PPO_SINE_PERIOD_S
    immediate_lost_terminate: bool = True
    lost_track_terminate_steps: int = 1
    debug_disable_lost_terminate: bool = False


@configclass
class SonarCentroidTrackContinuousEvalCfg(SonarCentroidTrackEnvCfg):
    """Play / eval — continuous vy, matches continuous PPO training."""

    scene: InteractiveSceneCfg = SCENE_CFG.replace(num_envs=1)
    continuous_actions: bool = True
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(POLICY_ACTION_DIM,), dtype=np.float32
    )
    observation_space = POLICY_OBS_DIM_IMAGE
    image_only_obs: bool = True
    eval_mode: bool = True
    lateral_target_moving: bool = True
    lateral_target_mode: str = "sine"
    lateral_target_amp_m: float = _PPO_AMP_M
    lateral_target_period_s: float = _PPO_SINE_PERIOD_S
    lateral_target_vy: float = _PPO_TARGET_VY_PEAK
    max_lateral_velocity: float = _PPO_ROBOT_MAX_VY
    spawn_radius: float = 0.0
    eval_spawn_radius: float = 0.0
    episode_length_s: float = _PPO_SINE_PERIOD_S
    immediate_lost_terminate: bool = True
    lost_track_terminate_steps: int = 1
    print_centroid_debug: bool = True


@configclass
class SonarCentroidTrackEvalCfg(SonarCentroidTrackEnvCfg):
    """Play / eval — must match trained PPO obs & target motion."""

    scene: InteractiveSceneCfg = SCENE_CFG.replace(num_envs=1)
    observation_space = POLICY_OBS_DIM_IMAGE
    image_only_obs: bool = True
    eval_mode: bool = True
    lateral_target_moving: bool = True
    lateral_target_mode: str = "sine"
    lateral_target_amp_m: float = _PPO_AMP_M
    lateral_target_period_s: float = _PPO_SINE_PERIOD_S
    lateral_target_vy: float = _PPO_TARGET_VY_PEAK
    max_lateral_velocity: float = _PPO_ROBOT_MAX_VY
    spawn_radius: float = 0.0
    eval_spawn_radius: float = 0.0
    episode_length_s: float = _PPO_SINE_PERIOD_S
    immediate_lost_terminate: bool = True
    lost_track_terminate_steps: int = 1
    print_centroid_debug: bool = True
