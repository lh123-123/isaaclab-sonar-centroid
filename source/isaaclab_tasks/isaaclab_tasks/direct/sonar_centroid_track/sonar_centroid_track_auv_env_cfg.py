# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for sonar centroid tracking on WarpAUV (6 thrusters + hydro)."""

from __future__ import annotations

import gymnasium as gym
import math
import numpy as np

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .auv_platform import (
    AUV_BEACON_CFG,
    AUV_SCENE_CFG,
    AUV_SIM_CFG,
    AUV_SONAR_MOUNT_OFFSET_LOCAL,
    AUV_SONAR_HOUSING_OFFSET_LOCAL,
    AUV_SONAR_MOUNT_VISUAL,
    AUV_STARTING_DEPTH_M,
    AUV_VIEWER_CFG,
    OCEANSIM_SONAR_CFG,
    WARPAUV_ROBOT_CFG,
)
from .modules.constants import (
    DEFAULT_MAX_RESIDUAL_SURGE,
    DEFAULT_MAX_RESIDUAL_SWAY,
    POLICY_AUV_RESIDUAL_ACTION_DIM,
    POLICY_OBS_DIM_IMAGE,
    POLICY_OBS_DIM_IMAGE_RANGE,
)
from .sonar_centroid_track_env_cfg import SonarCentroidTrackEnvCfg

# Match Cuboid PPO sine: dy = A·sin(2πt/T), peak |vy| = 0.28 m/s at A=1.3 m.
_AUV_SINE_AMP_M = 1.3
_AUV_TARGET_VY_PEAK = 0.28
_AUV_SINE_PERIOD_S = 2.0 * math.pi * _AUV_SINE_AMP_M / _AUV_TARGET_VY_PEAK
_AUV_MAX_VY = _AUV_TARGET_VY_PEAK * 1.25


@configclass
class SonarCentroidTrackAUVEnvCfg(SonarCentroidTrackEnvCfg):
    """Base cfg: WarpAUV rigid body + OceanSim sonar + analytic hydrodynamics."""

    decimation = 2
    episode_length_s = 120.0

    sim = AUV_SIM_CFG
    scene: InteractiveSceneCfg = AUV_SCENE_CFG
    viewer = AUV_VIEWER_CFG

    robot_cfg: RigidObjectCfg = WARPAUV_ROBOT_CFG
    beacon_cfg: RigidObjectCfg = AUV_BEACON_CFG

    workspace_height: float = AUV_STARTING_DEPTH_M
    station_origin_local: tuple[float, float, float] = (0.0, 0.0, AUV_STARTING_DEPTH_M)
    sonar_mount_offset_local: tuple[float, float, float] = AUV_SONAR_MOUNT_OFFSET_LOCAL
    sonar_mount_visual: bool = AUV_SONAR_MOUNT_VISUAL
    sonar_housing_offset_local: tuple[float, float, float] = AUV_SONAR_HOUSING_OFFSET_LOCAL

    # WarpAUV uses 6 PWM actions; Phase A0 applies hover (zeros) internally.
    auv_physics_mode: bool = True
    auv_hover_only: bool = True
    continuous_actions: bool = True
    num_auv_actions: int = 6
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    # Hydrodynamics (defaults from isaac-auv-env WarpAUVEnvCfg).
    auv_com_to_cob_offset: tuple[float, float, float] = (0.0, 0.0, 0.01)
    auv_water_rho: float = 997.0
    auv_water_beta: float = 0.001306
    auv_rotor_constant: float = 0.1 / 100.0
    auv_dyn_time_constant: float = 0.05
    auv_volume: float = 0.022747843530591776
    auv_mass: float = 22.701

    min_depth: float = AUV_STARTING_DEPTH_M - 2.0
    max_depth: float = AUV_STARTING_DEPTH_M + 2.0
    max_robot_z: float = 3.0
    use_boundaries: bool = False

    oceansim_sonar_cfg = OCEANSIM_SONAR_CFG

    # Position PD + thruster allocation (Phase A1+).
    auv_pos_kp: float = 12.0
    auv_pos_kd: float = 3.0
    auv_kp_yaw: float = 80.0
    auv_kd_yaw: float = 14.0
    auv_max_wrench: float = 200.0
    auv_max_force: float = 280.0
    auv_max_torque: float = 55.0
    auv_vy_ff: float = 55.0
    auv_drag_ff: float = 0.0
    auv_direct_pwm: bool = False

    # ``decoupled``: mission-frame scalar PD (sway/surge/heave + attitude hold).
    # ``coupled``: legacy 3D position target + quaternion PD.
    auv_control_mode: str = "decoupled"
    auv_kp_sway: float = 12.0
    auv_kd_sway: float = 3.0
    auv_kp_surge: float = 10.0
    auv_kd_surge: float = 2.5
    auv_kp_depth: float = 14.0
    auv_kd_depth: float = 3.5
    auv_ki_depth: float = 0.0
    auv_ki_depth_limit: float = 40.0
    auv_kp_roll: float = 120.0
    auv_kd_roll: float = 18.0
    auv_kp_pitch: float = 120.0
    auv_kd_pitch: float = 18.0
    # Heave+attitude kept under PWM saturation (see thruster_allocation.allocate_thrusters).
    auv_alloc_priority: bool = True
    # When depth/attitude error is large, shrink Fx/Fy (mission) so station recovers first.
    auv_station_gate: bool = True
    auv_station_gate_depth_m: float = 0.12
    auv_station_gate_att_rad: float = 0.10
    auv_station_gate_min_scale: float = 0.05
    auv_max_force_horiz: float = 180.0
    auv_max_force_heave: float = 320.0

    # Phase A2 compensatory PPO: fixed base law + 2D residual (Δsway, Δsurge).
    auv_base_law: str = "position_lead"  # "position_lead" | "image_range"
    # Step C: blend lateral base  α*GT_lead + (1−α)*image_P  (1=GT only, 0=image only).
    auv_gt_lateral_blend: float = 1.0
    auv_residual_ppo: bool = False
    auv_residual_action_dim: int = POLICY_AUV_RESIDUAL_ACTION_DIM
    # position_lead base (frozen Phase A1 PD baseline — see eval_centroid_track_p_control_auv.sh)
    auv_pl_kp_sway: float = 6.5
    auv_pl_lead_s: float = 0.65
    auv_pl_kp_vy: float = 0.6
    # image_range base (legacy Phase A2 scaffold)
    auv_image_p_kp: float = 1.2
    auv_image_p_kff: float = 0.0
    auv_image_p_deadband: float = 0.003
    auv_kp_range: float = 1.2
    auv_kp_range_fwd: float = 0.35
    auv_kp_range_back: float = 3.0
    auv_kp_surge_hold: float = 2.5
    auv_max_surge: float = 1.0
    auv_max_residual_sway: float = DEFAULT_MAX_RESIDUAL_SWAY
    auv_max_residual_surge: float = DEFAULT_MAX_RESIDUAL_SURGE
    include_range_obs: bool = False
    reward_w_range: float = 0.8
    reward_w_too_close: float = 2.5
    reward_w_residual: float = 0.02
    reward_w_lateral_action: float = 0.0
    reward_w_sway: float = 0.0
    reward_w_u: float = 1.0
    reward_w_v: float = 0.3
    reward_w_safety: float = 10.0
    min_close_range_m: float = 0.5
    range_accept_m: float = 0.5
    # Sonar range: A=reset bias calib, B=azimuth gate, C=hold_meas / fixed / reset_gt reference.
    sonar_range_reset_calib: bool = True
    sonar_range_ref_mode: str = "hold_meas"  # "hold_meas" | "reset_gt" | "fixed"
    sonar_range_ref_fixed_m: float = 3.0
    sonar_beam_azimuth_gate_deg: float = 15.0
    sonar_beam_prior_use_teach: bool = False
    sonar_beam_peak_valid_min: float = 0.5
    sonar_range_ema_alpha: float = 0.0
    sonar_range_bias_recalib_sway_m: float = 0.40
    sonar_range_bias_recalib_period_steps: int = 120
    sonar_range_deadband_m: float = 0.15
    sonar_range_close_margin_m: float = 0.25
    # When True, PD range-hold never commands forward surge (retreat + hold only).
    auv_inhibit_surge_forward: bool = False


@configclass
class SonarCentroidTrackAUVDebugCfg(SonarCentroidTrackAUVEnvCfg):
    """Phase A0: WarpAUV hover + static beacon + sonar / fixed-point validation."""

    scene: InteractiveSceneCfg = AUV_SCENE_CFG.replace(num_envs=1)
    episode_length_s = 3600.0
    eval_mode: bool = True
    spawn_radius: float = 0.0
    eval_spawn_radius: float = 0.0
    lateral_target_moving: bool = False
    auv_hover_only: bool = True
    auv_physics_mode: bool = True
    freeze_robot_pose: bool = False
    debug_disable_lost_terminate: bool = True
    print_centroid_debug: bool = True
    centroid_debug_interval: int = 10
    observation_space = POLICY_OBS_DIM_IMAGE
    image_only_obs: bool = True


@configclass
class SonarCentroidTrackAUVMotionCfg(SonarCentroidTrackAUVEnvCfg):
    """Phase A1: sine moving target + thruster lateral tracking."""

    lateral_target_moving: bool = True
    lateral_target_mode: str = "sine"
    lateral_target_amp_m: float = _AUV_SINE_AMP_M
    lateral_target_period_s: float = _AUV_SINE_PERIOD_S
    lateral_target_vy: float = _AUV_TARGET_VY_PEAK
    max_lateral_velocity: float = _AUV_MAX_VY
    auv_hover_only: bool = False
    continuous_actions: bool = True
    image_only_obs: bool = True
    observation_space = POLICY_OBS_DIM_IMAGE


@configclass
class SonarCentroidTrackAUVMotionDebugCfg(SonarCentroidTrackAUVMotionCfg):
    """Phase A1 eval: sine target + image P-control via sway/surge overrides."""

    scene: InteractiveSceneCfg = AUV_SCENE_CFG.replace(num_envs=1)
    eval_mode: bool = True
    spawn_radius: float = 0.0
    eval_spawn_radius: float = 0.0
    debug_disable_lost_terminate: bool = True
    print_centroid_debug: bool = False


@configclass
class SonarCentroidTrackAUVResidualPpoCfg(SonarCentroidTrackAUVMotionCfg):
    """Phase A2: compensatory PPO — position+lead PD base + 2D residual (hold_meas range)."""

    scene: InteractiveSceneCfg = AUV_SCENE_CFG.replace(num_envs=2)
    # First half of sine (0 → +A → 0); ~875 steps at 60 Hz control.
    episode_length_s: float = _AUV_SINE_PERIOD_S / 2.0
    spawn_radius: float = 0.08
    immediate_lost_terminate: bool = False
    lost_track_terminate_steps: int = 5
    debug_disable_lost_terminate: bool = False
    # Shorter reset warmup than debug (12→6 sonar renders per episode).
    sonar_warmup_frames: int = 2
    fixed_point_calib_settle_steps: int = 4

    auv_residual_ppo: bool = True
    # Stronger lateral PD once depth station is OK (≈1-DoF follow).
    auv_pos_kp: float = 18.0
    auv_pos_kd: float = 4.5
    auv_kp_sway: float = 22.0
    auv_kd_sway: float = 5.5
    auv_vy_ff: float = 90.0
    auv_pl_kp_sway: float = 10.0
    auv_pl_lead_s: float = 0.70
    auv_pl_kp_vy: float = 1.0
    # Station layer: keep depth/attitude OK so task ≈ 1-DoF lateral follow.
    auv_kp_depth: float = 55.0
    auv_kd_depth: float = 16.0
    auv_ki_depth: float = 12.0
    auv_ki_depth_limit: float = 50.0
    auv_kp_roll: float = 220.0
    auv_kd_roll: float = 34.0
    auv_kp_pitch: float = 220.0
    auv_kd_pitch: float = 34.0
    auv_kp_yaw: float = 140.0
    auv_kd_yaw: float = 24.0
    auv_kp_surge: float = 40.0
    auv_kd_surge: float = 12.0
    auv_max_force_horiz: float = 260.0
    auv_max_force_heave: float = 360.0
    auv_max_torque: float = 90.0
    auv_station_gate: bool = True
    auv_station_gate_depth_m: float = 0.12
    auv_station_gate_att_rad: float = 0.20
    auv_station_gate_min_scale: float = 0.40
    # Gate Fy only (default); keep Fx surge brake un-gated.
    auv_station_gate_surge: bool = False
    # Heave-first PWM; leave attitude+horizontal to share residual headroom.
    auv_alloc_priority: bool = True
    auv_alloc_priority_attitude: bool = False

    # Step A (sim→real): surge from beam hold_meas, not hard task-origin zero.
    auv_inhibit_surge_forward: bool = False
    auv_kp_range_fwd: float = 0.40
    auv_kp_range_back: float = 3.0
    auv_kp_surge_hold: float = 0.0  # in-band freezes current surge
    auv_max_surge: float = 1.0
    range_accept_m: float = 0.50
    sonar_range_deadband_m: float = 0.10
    sonar_range_ref_mode: str = "hold_meas"
    # Match Cblend05 train: α=0.5, residual ±0.45 / ±0.70.
    auv_gt_lateral_blend: float = 0.5
    auv_max_residual_sway: float = 0.45
    auv_max_residual_surge: float = 0.70
    # Step B: dirty depth/IMU for station loops (train). Soft sigma; delay=0 (delay+high gains jitters).
    auv_station_obs_noise: bool = True
    auv_noise_depth_m: float = 0.01
    auv_noise_att_rad: float = 0.01
    auv_noise_vz: float = 0.01
    auv_noise_ang_vel: float = 0.01
    auv_noise_delay_steps: int = 0
    # Beam range too-close does not reset episode (reward penalty still applies).
    debug_disable_too_close_terminate: bool = True
    # Lateral-first reward: drop anti-sway action tax; train GT sway + stronger image P.
    reward_w_u: float = 5.0
    reward_w_sway: float = 4.0
    reward_w_range: float = 1.2
    reward_w_too_close: float = 5.0
    reward_w_residual: float = 0.08
    reward_w_lateral_action: float = 0.0

    include_range_obs: bool = True
    observation_space = POLICY_OBS_DIM_IMAGE_RANGE
    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(POLICY_AUV_RESIDUAL_ACTION_DIM,),
        dtype=np.float32,
    )


@configclass
class SonarCentroidTrackAUVResidualEvalCfg(SonarCentroidTrackAUVResidualPpoCfg):
    """Phase A2 eval / play — matches residual PPO training cfg."""

    scene: InteractiveSceneCfg = AUV_SCENE_CFG.replace(num_envs=1)
    eval_mode: bool = True
    spawn_radius: float = 0.0
    eval_spawn_radius: float = 0.0
    print_centroid_debug: bool = True
    centroid_debug_interval: int = 20
    # Same station noise as train — GUI is for seeing train-time behavior, not cosmetics.
