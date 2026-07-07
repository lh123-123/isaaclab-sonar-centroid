# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for WarpAUV residual RL environment."""

from __future__ import annotations

import gymnasium as gym
import importlib
import numpy as np

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

_warpauv = importlib.import_module("isaaclab_tasks.direct.isaac-auv-env.warpauv_env")
WarpAUVEnvCfg = _warpauv.WarpAUVEnvCfg

_WARPAUV_SCENE = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)


@configclass
class ResidualRLEnvCfg(WarpAUVEnvCfg):
    """Residual RL config: SAC outputs bounded thruster residuals on top of imperfect PD."""

    # --- Residual control ---
    residual_mode: str = "wrench"  # "wrench": Δw on PD wrench; "thruster": legacy u_base+u_rl
    delta: float = 0.15  # thruster mode: max |u_rl| per PWM channel
    delta_wrench: float = 0.15  # wrench mode: max |Δw_i| per component (v3.1: match v2 δ)
    max_total_wrench: float = 2.5  # wrench mode: clamp on w_pd + Δw_rl (PD max ≈ 2.0)
    base_controller_imperfect: bool = True

    # --- Observation: base WarpAUV (17) + previous residual (6) = 23 ---
    num_observations: int = 23
    observation_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)
    state_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)

    # --- Reward weights (progress + penalties; RL should beat PD-only by reducing overshoot) ---
    rew_scale_pos: float = 0.3
    rew_scale_pos_lin: float = 0.15
    rew_scale_ang: float = 0.3
    rew_scale_u_rl: float = 0.5
    rew_scale_wrench_opp: float = 0.15  # wrench mode: penalize Δw opposing w_pd
    rew_scale_progress: float = 2.0
    rew_scale_vel_near: float = 0.05
    rew_scale_overshoot: float = 1.0
    rew_scale_success: float = 1.0
    rew_scale_terminal: float = 3.0
    success_pos_threshold: float = 0.2  # meters
    near_goal_dist: float = 1.0
    rew_clip_min: float = -5.0
    rew_clip_max: float = 5.0

    # --- Observation normalization (fixed scales, avoid RunningStandardScaler drift) ---
    obs_pos_scale: float = 5.0
    obs_vel_scale: float = 2.0
    obs_wrench_scale: float = 0.15  # match delta_wrench so prev Δw obs ∈ [-1, 1]

    # --- Curriculum (1: no DR, 2: thruster bias, 3: + current, 4: full DR) ---
    curriculum_stage: int = 1

    # --- Extended domain randomization ---
    class residual_domain_randomization:
        thruster_bias_range: tuple[float, float] = (-0.1, 0.1)
        thruster_gain_range: tuple[float, float] = (0.85, 1.15)
        current_speed_range: tuple[float, float] = (0.0, 0.5)  # m/s
        added_mass_scale_range: tuple[float, float] = (0.8, 1.2)

    # --- Evaluation / debug ---
    print_eval_debug: bool = False
    eval_silent: bool = False

    # --- Test 1: point reaching (fixed start → fixed target) ---
    point_reach_mode: bool = False
    test1_start_pos: tuple[float, float, float] = (0.0, 0.0, 8.0)
    test1_target_pos: tuple[float, float, float] = (5.0, 0.0, 8.0)
    test1_target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    test1_start_pos_noise: float = 0.0
    target_frame_scale: float = 0.25  # RGB frame at goal pose (debug vis)

    # --- Training defaults ---
    scene: InteractiveSceneCfg = _WARPAUV_SCENE.replace(num_envs=512)
    episode_length_s: float = 5.0


@configclass
class ResidualRLStage2Cfg(ResidualRLEnvCfg):
    """Curriculum stage 2: thruster bias/gain randomization."""

    curriculum_stage: int = 2


@configclass
class ResidualRLStage3Cfg(ResidualRLEnvCfg):
    """Curriculum stage 3: thruster DR + water current."""

    curriculum_stage: int = 3


@configclass
class ResidualRLStage4Cfg(ResidualRLEnvCfg):
    """Curriculum stage 4: full domain randomization."""

    curriculum_stage: int = 4


@configclass
class ResidualRLEvalCfg(ResidualRLEnvCfg):
    """Fixed-seed evaluation: single env, eval_mode, debug metrics."""

    eval_mode: bool = True
    scene: InteractiveSceneCfg = _WARPAUV_SCENE.replace(num_envs=1)
    print_eval_debug: bool = True
    cap_episode_length: bool = True
    episode_length_s: float = 10.0


@configclass
class ResidualRLTest1Cfg(ResidualRLEnvCfg):
    """Test 1 — point reaching: start (0,0,8) → target (5,0,8) in env frame.

    WarpAUV uses positive Z as depth; plan coordinate (5, 0, -8) maps to (5, 0, 8) here.
    """

    point_reach_mode: bool = True
    test1_start_pos: tuple[float, float, float] = (0.0, 0.0, 8.0)
    test1_target_pos: tuple[float, float, float] = (5.0, 0.0, 8.0)
    test1_target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    test1_start_pos_noise: float = 0.0

    eval_mode: bool = True
    curriculum_stage: int = 1
    debug_vis: bool = True
    scene: InteractiveSceneCfg = _WARPAUV_SCENE.replace(num_envs=1)
    episode_length_s: float = 15.0
    cap_episode_length: bool = True
    use_boundaries: bool = True
    max_auv_x: float = 10.0
    max_auv_y: float = 10.0
    max_auv_z: float = 10.0
    print_eval_debug: bool = False
    eval_silent: bool = True

    class domain_randomization:
        use_custom_randomization = False
        com_to_cob_offset_radius = 0.0
        volume_range = [0.022747843530591776, 0.022747843530591776]
        mass_range = [2.2701e+01, 2.2701e+01]


@configclass
class ResidualRLTest1TrainCfg(ResidualRLTest1Cfg):
    """Test 1 training variant: parallel envs, small start noise, no eval_mode."""

    eval_mode: bool = False
    debug_vis: bool = False
    scene: InteractiveSceneCfg = _WARPAUV_SCENE.replace(num_envs=128)
    test1_start_pos_noise: float = 0.1
    episode_length_s: float = 15.0
    print_eval_debug: bool = False
    eval_silent: bool = True
