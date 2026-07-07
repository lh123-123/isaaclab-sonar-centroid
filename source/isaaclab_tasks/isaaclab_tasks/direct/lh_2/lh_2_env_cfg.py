# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from .oceansim_sonar_sensor_cfg import OceanSimSonarSensorCfg


@configclass
class SonarCfg:
    """声呐传感器配置（与 :attr:`oceansim_sonar_cfg` 量程应对齐，供奖励/简易探测逻辑使用）。

    注意：几何 ``_simulate_sonar`` 仅用水平角与 ``hori_fov``；``vert_fov`` 未参与当前 atan2 判据，
    与 OceanSim 成像 ``vert_fov``（45°）可不一致，属简化模型与真实传感器的差异。
    """
    max_range: float = 3.0
    min_range: float = 0.1
    hori_fov: float = 90.0
    vert_fov: float = 10.0
    hori_res: int = 128
    vert_res: int = 64
    frequency: float = 50.0
    intensity_threshold: float = 0.3


@configclass
class Lh2EnvCfg(DirectRLEnvCfg):
    """水下声呐旋转跟随：机器人仅绕世界 Z 转动；目标在水平面内绕机器人、固定半径、随机周向角速度运动。

    水平距离限制在 ``[target_orbit_radius_min, target_orbit_radius_max]`` 圆环带内；默认半径 ``target_orbit_radius``（1.5 m）。
    可选 ``target_orbit_randomize_radius_on_reset``；周向见 ``target_orbit_angular_vel_*`` / ``target_orbit_angular_noise_std``。
    """

    decimation = 4
    episode_length_s = 10.0
    
    action_space = 1
    # Policy obs (beam mode): beam_count * [angle, range, intensity] + tail.
    # Tail 默认 [ωz, pad, last_action]；若 ``policy_append_orbit_w_inst`` 且 ``orbit_robot`` 再 +1（tanh 归一化 w_inst）。
    # With beam_count=128, tail=4: 128*3 + 4 = 388.
    observation_space = 388
    # Asymmetric critic: 与 policy 相同维 + 1（机体 bearing 每步变化）。
    state_space = 389

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 50, 
        render_interval=decimation,
        gravity=(0.0, 0.0, 0.0),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10.0, replicate_physics=True)

    # Robot/target live near z=10 in env frame; frame the workspace from the env origin (not world 0,0,0).
    viewer: ViewerCfg = ViewerCfg(
        eye=(10.0, -10.0, 8.0),
        lookat=(0.0, 0.0, 10.0),
        origin_type="env",
        env_index=0,
    )

    robot_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.3, 0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                # 原 10/10 使 yaw 跟指令严重发肉；单自由度旋转任务用较小阻尼即可。
                linear_damping=3.0,
                angular_damping=1.0,
                max_linear_velocity=10.0,
                # Isaac Lab USD 刚体：单位为 **deg/s**（见 RigidBodyPropertiesCfg 文档），不是 rad/s。
                # 曾误写 10 → 仿真 ω 被钳在 ~10°/s，而 action_scale_angular=0.42rad/s≈24°/s，指令与实测严重脱节。
                max_angular_velocity=40.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 10.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.18,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=2.0,
                angular_damping=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, 0.0, 10.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    #: 目标在 env 系内固定 x（仅 ``random`` / ``waypoint_cycle`` 模式使用）
    target_fixed_x: float = 2.0
    # Target motion mode:
    # - "orbit_robot": 目标在水平面内以机器人为圆心、**半径固定**（见 ``target_orbit_radius*``），
    #   周向为**随机角速度**（每步噪声 + 限幅），非匀速圆周；reset 时小球在机体 +X 前向水平距离 R 处。
    # - "random": 沿 y 随机侧向运动（``target_fixed_x`` 固定）
    # - "waypoint_cycle": 沿 y 路点循环
    target_motion_mode: str = "orbit_robot"
    #: 目标与机器人水平距离允许的最小半径（m），圆环带内径。
    target_orbit_radius_min: float = 1.0
    #: 目标与机器人水平距离允许的最大半径（m），圆环带外径。
    target_orbit_radius_max: float = 2.5
    #: 当前默认轨道半径（m），会夹紧到 ``[min, max]``；每环境张量见 ``Lh2Env._target_orbit_radius_env``。
    target_orbit_radius: float = 1.5
    #: 若 True，每次 reset 在 ``[min, max]`` 上均匀随机轨道半径（否则每环境恒为 ``target_orbit_radius`` 夹紧值）。
    target_orbit_randomize_radius_on_reset: bool = False
    #: （已弃用匀速）保留作文档参考；随机周向角速度见 ``target_orbit_angular_vel_*`` / ``target_orbit_angular_noise_std``。
    target_orbit_wz: float = 0.15
    #: 小球绕机器人瞬时角速度 ``w_inst`` 的下限（rad/s，绕 +Z）。
    target_orbit_angular_vel_min: float = -0.30
    #: 小球绕机器人瞬时角速度 ``w_inst`` 的上限（rad/s）。
    target_orbit_angular_vel_max: float = 0.30
    #: 每环境步对 ``w_inst`` 叠加的高斯噪声标准差（rad/s），再夹紧到上下限，形成随机周向运动。
    target_orbit_angular_noise_std: float = 0.08
    # Lateral waypoints (env-local y) for deterministic cyclic motion.
    target_waypoint_cycle_y: tuple[float, ...] = (0.4, -0.4, 0.75, -0.75)
    # Constant speed (m/s) for waypoint-cycle motion.
    target_waypoint_speed: float = 0.24
    # Switch to next waypoint when |y - y_wp| <= threshold.
    target_waypoint_reach_threshold: float = 0.01
    #: 目标左右平移范围：y ∈ [-target_lateral_half_range, +…]，总宽度 2×该值。
    #: Slightly narrower than 2.0 to improve early-stage controllability.
    target_lateral_half_range: float = 0.8

    sonar_cfg: SonarCfg = SonarCfg()

    #: 声呐安装点相对 Robot 根 prim 的平移（机体坐标，与 ``SonarMount`` USD Xform 一致；用于几何检测/居中奖励与真实传感器对齐）
    sonar_mount_offset_local: tuple[float, float, float] = (0.6, 0.0, 0.0)

    oceansim_sonar_cfg: OceanSimSonarSensorCfg = OceanSimSonarSensorCfg(
        min_range=0.1,
        max_range=3.0,
        range_res=0.05,
        hori_fov=90.0,
        vert_fov=45.0,
        angular_res=2.0,
        hori_res=96,
        height=48,
        width=96,
        beam_count=128,
        include_unlabelled=True,
        binning_method="sum",
        normalizing_method="range",
        attenuation=0.06,
        gau_noise_param=0.06,
        ray_noise_param=0.02,
        intensity_offset=0.05,
        intensity_gain=1.8,
        sonar_initialize_viewport=False,
        use_replicator_orchestrator_recovery=False,
    )

    #: 机器人单自由度：绕世界 Z 轴角速度指令 ``omega_z = action * action_scale_angular``。
    action_scale_linear = 3.2
    #: 与 ``max_angular_velocity`` 取相同数值，使 |action|=1 时饱和到该角速度（rad/s）。
    #: 小球轨道 ``|w_inst|≤max(|target_orbit_angular_vel_min|,|target_orbit_angular_vel_max|)``（当前 0.30），
    #: 机体 yaw 上限取约 **1.4×** 留跟踪余量；与目标周向难度一并调。
    action_scale_angular = 0.42
    # Freeze robot and target motion for first N env-steps after reset（0 = 不冻结）。
    startup_freeze_steps: int = 0
    
    max_linear_velocity = 0.80
    #: 绕 Z 最大角速度（rad/s），应与 ``action_scale_angular`` 一致；约 24°/s（当前比球 w_inst 上限略高）。
    max_angular_velocity = 0.42
    
    # --- Simplified reward scales（旋转跟随：机体方位对准目标）---
    rew_scale_distance = 0.0
    rew_scale_heading = 0.0
    # Core: minimize |bearing|（机体 +x 前方指向目标的水平方位角，rad）。
    rew_scale_lateral = 3.20
    # Encourage per-step improvement of lateral error.
    rew_scale_lateral_progress = 2.50
    # Prefer operating in the range band where sonar tracking is most reliable.
    rew_scale_range_band = 0.0
    # Encourage vy to match target vy once visible.
    rew_scale_velocity_match = 0.35
    # Closing term complements progress when target speed changes.
    rew_scale_closing = 0.0
    # Acquisition: keep target visible.
    rew_scale_sonar_detection = 0.30
    # Alive reward scale (multiplies a step-progress term in [0, alive_progress_max]).
    rew_scale_alive = 0.05
    # Upper bound for step-progress alive reward term.
    alive_progress_max: float = 1.0
    # Centering in sonar FOV (world-angle based in env logic).
    rew_scale_centering = 0.0
    # Velocity-direction shaping.
    rew_scale_wrong_direction = 0.0
    rew_scale_direction_align = 0.0
    # Directly shape policy output sign.
    rew_scale_action_direction_align = 0.0
    rew_scale_action_wrong_direction = -1.80
    # Event bonus when recovering after target direction reversal.
    rew_scale_reversal_success = 0.0
    # Penalize error growth.
    rew_scale_diverge = -0.60
    # Stabilize near alignment.
    rew_scale_idle_speed = -0.02
    rew_scale_action_smooth = -0.004
    # Additional anti-saturation regularization on raw action magnitude.
    rew_scale_action_magnitude = -0.004
    # Safety / failure penalties.
    rew_scale_collision = -10.0
    rew_scale_out_of_range = -5.0
    rew_scale_out_of_view = -1.0
    # Linear ramp for out-of-view penalty with consecutive lost steps.
    out_of_view_penalty_ramp_steps: int = 12
    
    # 距离 shaping / 距离带（与圆环 1.0–2.5 m、当前 R≈1.5 m 对齐）
    target_tracking_distance = 1.5
    min_tracking_range = 0.92
    max_tracking_range = 2.72
    # 方位角误差尺度（rad）。
    lateral_tracking_sigma = 0.45
    # 角速度匹配尺度（rad/s）：对 ``|ωz - ω_des|`` 的指数核宽度（与 max_angular_velocity 同量级）。
    velocity_match_sigma = 0.20
    # 判定“已对齐”的方位角阈值（rad）。
    lateral_idle_threshold = 0.14
    # Velocity-match reward gate on |bearing| (rad). Use ≥π so bearing-PD match stays on during large errors / reversals.
    velocity_match_lateral_threshold: float = 3.2
    #: ``legacy_orbit_w``：``|ωz - target_orbit_w_inst|``（仅适合纯定圆 + 随机 w_inst）。
    #: ``bearing_pd``（默认）：``ω_des = -kp*bearing - kd*ψ̇``（``ψ̇`` 为机体 bearing 每 env 步差分），
    #: 适合圆环带内随机运动等一般目标轨迹，不依赖轨道角速度标量。
    velocity_match_mode: str = "bearing_pd"
    #: PD 期望角速度 ``ω_des``（rad/s）系数；bearing、``ψ̇`` 均为 rad、rad/s。
    velocity_match_kp: float = 1.0
    velocity_match_kd: float = 0.75
    #: 与 ``rew_scale_action_wrong_direction`` 对称：|bearing| 超过 ``direction_lateral_threshold`` 且 ω 指令与 bearing 同号时给正奖励。
    #: 幅值 ≈ ``rew_scale_direction_turn_bonus * clamp(|bearing|/direction_turn_bonus_span,0,1)``（再乘可见度 detected_f）。
    rew_scale_direction_turn_bonus: float = 0.55
    #: 方位误差（rad）线性升到满分 bonus 的上限；约 1.2 rad 时在正确转向侧拿满该项。
    direction_turn_bonus_span: float = 1.2
    direction_lateral_threshold = 0.10
    # 判定速度方向的死区，过滤接近 0 的抖动速度。
    direction_velocity_eps = 0.03
    # 判定动作方向的死区（策略输出接近 0 时不计方向奖惩）。
    direction_action_eps = 0.03
    # 持续动作方向错误的重置参数（用于打破“恒定单边动作”局部最优）。
    wrong_direction_reset_enabled: bool = False
    wrong_direction_reset_lateral_threshold: float = 0.22
    wrong_direction_reset_steps: int = 24
    # 动作平滑项仅在 |dy| 小于该阈值时启用（远离目标时允许激进反向）
    smooth_lateral_threshold = 0.38
    max_tracking_distance = 4.5
    min_safe_distance = 0.42
    # Robot lateral workspace limit: reset when |robot_y| exceeds this value.
    robot_lateral_limit: float = 3.0
    
    # Difficulty curriculum for target speed dynamics: start easier, then increase variability.
    target_speed_range_start = (0.12, 0.12)
    target_speed_range_final = (0.12, 0.12)
    #: Per-step probability of updating target speed goal.
    target_speed_change_prob_start = 0.0
    target_speed_change_prob_final = 0.0
    #: Speed-change acceleration limit (m/s²), lower is smoother.
    target_speed_accel_limit_start = 1.0
    target_speed_accel_limit_final = 1.0
    #: Curriculum horizon in common vectorized env steps.
    target_speed_curriculum_env_steps: int = 160_000
    #: Per env-step probability of flipping ``target_y_sign`` (smaller => fewer random reversals).
    target_direction_change_prob = 0.03
    
    # 略大于目标最大偏移与 x 间距，避免边界拉回逻辑过早收紧轨迹
    target_boundary_range = 5.0
    # 视野中心角奖励尺度（度），越小越严格要求贴近中心
    centering_sigma_deg: float = 12.0

    # 目标离开声呐视野后是否允许按连续丢失步数触发重置（非单步立刻重置）
    reset_when_target_out_of_view: bool = True
    # 连续超出视野多少个环境步后触发重置（仅在上项为 True 时生效）
    out_of_view_reset_steps: int = 10

    # Optional temporal buffer kept for debugging/ablation only (not fed to policy by default).
    beam_history_len: int = 4

    # Curriculum phase label for logging only (rewards are always the simplified tracking stack below).
    training_phase: str = "tracking"
    auto_phase_switch: bool = True
    phase_switch_visible_ratio: float = 0.90
    phase_switch_warmup_steps: int = 200
    visible_ratio_ema_alpha: float = 0.98
    # Soft-gate floor for tracking rewards after switching to tracking phase.
    tracking_gate_min: float = 0.25
    # Keep direction terms strong even when target is temporarily out of view.
    direction_out_of_view_scale: float = 1.0

    # Reversal success event parameters.
    reversal_eval_horizon: int = 12
    reversal_success_margin: float = 0.12

    #: Policy observation mode.
    #: - "image": sonar image + robot state tail (vx, vy, last_action)
    #: - "beam": beam triplets [angle(rad), range(norm), intensity(fused)] + tail
    #: - "relative_position": sonar-image-derived relative target state vector
    policy_obs_mode: str = "beam"
    # Beam-mode preprocessing and fusion controls.
    beam_obs_softmax_enabled: bool = True
    beam_obs_softmax_temperature: float = 1.00
    # Gate softmax enhancement only when target-like response is sufficiently strong.
    beam_obs_gate_peak_threshold: float = 0.005
    beam_obs_gate_mean_threshold: float = 0.0
    # If True, allow beam softmax gate when geometric sonar says target is detectable.
    beam_obs_gate_use_geom_detect: bool = True
    #: If True, reward/done「可见」与 OceanSim 波束门控一致（每步在几何声呐后只 update 一次 OceanSim）。
    #: If False,沿用几何 ``sonar_range`` 阈值（与 policy 输入可能不一致）。
    reward_detection_use_oceansim_beam: bool = True
    #: 若 True：奖励/终止的可见度同时参考波束门控与几何（见 ``reward_visibility_combine``）。
    #: 若 False：仅用波束门控（无几何并入）。
    reward_beam_detection_or_geom: bool = True
    #: 在 ``reward_beam_detection_or_geom`` 为 True 时，波束门控与几何可见度如何合并（均为 0/1 浮点）：
    #: - ``"min"``：两者都可见才可见（默认；避免松 beam 门单独把 ``visible_ratio`` 顶满）。
    #: - ``"max"``：任一可见即可见（旧 OR 语义；波束门松时易饱和为 1）。
    reward_visibility_combine: str = "min"
    # Fusion weight for enhanced intensity when gate is open.
    beam_obs_softmax_mix: float = 0.10
    # Intensity log compression gain before normalization.
    beam_obs_intensity_log_gain: float = 8.0
    # Relative-position frontend controls (sonar-image moment estimator).
    rel_est_conf_threshold: float = 0.66
    rel_est_ema_alpha: float = 0.90
    # If True, pass gate whenever a valid blob is present in sonar image (FOV-only gating).
    # Keep False when using confidence-threshold-as-visibility logic.
    rel_est_gate_fov_only: bool = False
    # State update threshold: update rel_est_state only when confidence exceeds this value.
    rel_est_update_conf_threshold: float = 0.66
    # Composite confidence weights: peak + area + compactness + temporal consistency.
    rel_est_conf_w_peak: float = 0.35
    rel_est_conf_w_area: float = 0.25
    rel_est_conf_w_compact: float = 0.20
    rel_est_conf_w_temporal: float = 0.20
    # Normalization scales for confidence terms.
    rel_est_area_ref: float = 0.01
    rel_est_compact_sigma: float = 0.015
    rel_est_temporal_sigma: float = 0.30
    # Scale calibration for estimated forward range component (x) from sonar-image moments.
    rel_est_x_scale: float = 0.88
    # Gain compensation for estimated lateral position (y) from sonar-image moments.
    rel_est_y_gain: float = 1.25
    # Piecewise gain: use stronger gain when |rel_y_raw| exceeds this threshold.
    rel_est_y_piecewise_threshold: float = 0.9
    rel_est_y_gain_high: float = 1.35
    # When detection is lost, decay the last estimated relative state instead of hard-zeroing.
    rel_est_lost_decay: float = 0.98
    # Clamp for estimated relative y-velocity fed to policy.
    rel_est_vy_clip: float = 1.0
    # Zero rel_vy for first N env steps after reset to suppress initialization spikes.
    rel_est_vy_warmup_steps: int = 2
    # If True, allow detection whenever GT says target is in sonar FOV/range (sim-only helper).
    rel_est_gate_with_in_fov_gt: bool = False
    #: If True, append ``tanh(w_inst / policy_orbit_w_inst_norm_scale)`` to policy tail（仅 ``target_motion_mode=="orbit_robot"``）。
    #: 显式给出小球绕转方向，减轻仅靠声呐序列推断 w 符号的难度，利于反向周向时策略对称翻转。
    policy_append_orbit_w_inst: bool = True
    #: 与 ``|target_orbit_angular_vel_*|`` 同量级即可（rad/s）。
    policy_orbit_w_inst_norm_scale: float = 0.35

    #: If True, append one scalar to the policy tail: ``tanh((target_y - robot_y) / sigma)`` (world frame).
    #: Keep disabled in relative-position mode.
    policy_lateral_hint_enabled: bool = False
    #: Scale for lateral hint (m); larger => weaker near-origin sensitivity.
    policy_lateral_hint_sigma: float = 1.5
    #: Stage-1 linear decay steps for lateral hint (vectorized env steps; ``common_step_counter``).
    #: During this stage, hint weight decays from 1.0 to ``policy_lateral_hint_min_weight``.
    #: With ``num_steps_per_env=64``, 20k steps is about 313 iterations.
    policy_lateral_hint_curriculum_env_steps: int = 20_000
    #: Stage-1 target floor. Keep a small hint after fast decay so policy does not see an abrupt drop.
    policy_lateral_hint_min_weight: float = 0.10
    #: Stage-2 linear tail steps: decay from ``policy_lateral_hint_min_weight`` to 0.
    #: Set 0 to drop to zero immediately after stage-1.
    policy_lateral_hint_tail_env_steps: int = 8_000

    # Debug print: y-axis / velocity (legacy). Off by default.
    debug_print_y: bool = False
    debug_print_interval: int = 5
    # If True, y/beam debug prints below are emitted only when run_mode == "play".
    debug_print_only_in_play: bool = True
    # Beam raw / summary prints (verbose). Off by default.
    debug_print_beam_raw: bool = False
    # Print beam summary every N env steps for env0.
    debug_print_beam_interval: int = 20
    # If True, print all env0 beam triplets [angle_deg, range_m, intensity] at debug interval.
    # Warning: very verbose and can noticeably slow down training.
    debug_print_beam_full: bool = False
    # Number of leading beams to print each time (legacy, ignored by summary print).
    debug_print_beam_count: int = 0
    # Runtime mode tag set by launcher script ("train" or "play").
    run_mode: str = "train"
    #: 训练与回放控制台按间隔打印 env0：机体前向与目标水平方位差（度）；字段名沿用历史。
    print_ball_world_angle: bool = True
    #: 按 env0 的 ``episode_length_buf`` 步数间隔打印。
    ball_world_angle_print_interval: int = 20
    # 若 True：当 bearing 与 ω 指令异号时在死区外将 ω **整段置零**。
    # 策略长期单极输出（如恒 +1）时会长时间 effective_wz=0、与 net_wz_cmd 不一致；默认关闭，交给方向惩罚与 RL。
    dy_sign_guard: bool = False
    # 方位角门控死区（rad）；|bearing| 小于此值时不门控。
    dy_sign_guard_deadband: float = 0.07

