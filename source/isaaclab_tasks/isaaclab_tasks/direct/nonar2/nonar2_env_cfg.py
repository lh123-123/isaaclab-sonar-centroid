# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for 1-DoF forward-looking sonar underwater tracking (Nonar2)."""

import gymnasium as gym
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from .oceansim_sonar_sensor_cfg import OceanSimSonarSensorCfg


@configclass
class Nonar2EnvCfg(DirectRLEnvCfg):
    """1-DoF sonar tracking with OceanSim imaging sonar.

    Control: continuous Δθ per step (mapped to yaw rate).
    Observation: belief state + detection + sonar heading (Phase 0, no CNN yet).
    """

    decimation = 5
    episode_length_s = 30.0

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    # [μx, μy, σx, σy, θ, azimuth, range, conf, vx, vy]
    observation_space = 10
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 50,
        render_interval=decimation,
        gravity=(0.0, 0.0, 0.0),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=12.0,
        replicate_physics=True,
    )

    # Height of robot/target in world Z (env-local); underwater workspace plane.
    workspace_height: float = 10.0

    viewer: ViewerCfg = ViewerCfg(
        eye=(4.0, -4.0, 11.8),
        lookat=(1.2, 0.0, 10.0),
        origin_type="env",
        env_index=0,
    )

    robot_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.35, 0.25),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.75)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                linear_damping=3.0,
                angular_damping=0.0,
                max_linear_velocity=5.0,
                max_angular_velocity=60.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=6.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 10.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.35,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.05)),
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
            pos=(3.0, 0.0, 10.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # --- Sonar (OceanSim) ---
    sonar_mount_offset_local: tuple[float, float, float] = (0.6, 0.0, 0.2)

    oceansim_sonar_cfg: OceanSimSonarSensorCfg = OceanSimSonarSensorCfg(
        min_range=0.5,
        max_range=6.0,
        range_res=0.05,
        hori_fov=120.0,
        vert_fov=20.0,
        angular_res=2.0,
        # Native OceanSim binning: H=(max-min)/range_res=110, W=hori_fov/angular_res+1=61
        hori_res=128,
        height=110,
        width=61,
        beam_count=61,
        beam_topk=1,
        include_unlabelled=True,
        binning_method="sum",
        normalizing_method="range",
        attenuation=0.08,
        gau_noise_param=0.12,
        ray_noise_param=0.04,
        intensity_offset=0.0,
        intensity_gain=1.4,
        sonar_initialize_viewport=False,
        use_replicator_orchestrator_recovery=False,
    )

    # --- Target motion ---
    #: "orbit" | "constant_velocity" (linear XY) | "fixed_radius_cv" (circular arc, const speed).
    target_motion_mode: str = "orbit"
    #: Fixed orbit radius (m) for fixed_radius_cv.
    target_fixed_radius: float = 3.0
    #: Tangential speed (m/s) for fixed_radius_cv; ω = speed / radius (sign flips at phase limits).
    target_fixed_speed_min: float = 0.25
    target_fixed_speed_max: float = 0.40
    #: World XY speed (m/s) for constant_velocity mode.
    target_cv_speed_min: float = 0.20
    target_cv_speed_max: float = 0.50
    #: Keep target inside this annulus (m); velocity reflects on inner/outer radius.
    target_cv_radius_min: float = 1.8
    target_cv_radius_max: float = 4.5
    target_cv_spawn_radius_min: float = 2.0
    target_cv_spawn_radius_max: float = 4.0
    #: Initial spawn azimuth half-width (deg) about world +X from env origin.
    target_cv_spawn_half_fov_deg: float = 75.0

    # --- Target orbit (when target_motion_mode == "orbit") ---
    target_orbit_frame: str = "world"
    #: World XY phase limits (deg): target sweeps arc from -90° to +90° about env origin.
    target_orbit_world_phase_min_deg: float = -90.0
    target_orbit_world_phase_max_deg: float = 90.0
    target_orbit_radius_min: float = 2.0
    target_orbit_radius_max: float = 4.0
    target_orbit_radius: float = 3.0
    target_orbit_randomize_radius_on_reset: bool = True
    #: Slow random drift of orbit-radius / angular-rate references (not applied directly to pose).
    target_orbit_radius_noise_std: float = 0.03
    target_orbit_angular_vel_min: float = -0.35
    target_orbit_angular_vel_max: float = 0.35
    target_orbit_angular_noise_std: float = 0.02
    #: Low-pass toward reference each control step (0–1; smaller = smoother).
    target_orbit_w_smooth_alpha: float = 0.12
    target_orbit_r_smooth_alpha: float = 0.10
    #: Jerk limits on realized ω and ṙ (rad/s², m/s).
    target_orbit_w_accel_max: float = 0.15
    target_orbit_r_speed_max: float = 0.12
    #: Fade orbit ω to zero within this many degrees of ±target phase limit (smooth turnaround).
    target_orbit_phase_soft_margin_deg: float = 15.0

    # --- Motion limits (robot yaw only; target world phase uses target_orbit_world_phase_* ) ---
    motion_yaw_min_deg: float = -90.0
    motion_yaw_max_deg: float = 90.0

    # --- Control (10 Hz); |action|=1 → max_angular_velocity (rad/s) ---
    control_dt: float = 0.1
    max_angular_velocity: float = 0.80  # ~46 deg/s
    max_dtheta: float = 0.08  # max_angular_velocity * control_dt (~4.6 deg/step)

    # --- Detection / belief ---
    detection_conf_threshold: float = 0.15
    detection_peak_threshold: float = 0.20
    #: ROI mean gate for beam mode; image_roi uses peak SNR instead (see SonarDetector).
    detection_mean_threshold: float = 0.03
    #: Accept beam peaks only inside this range band (m), aligned with orbit radius.
    detection_range_min: float = 1.5
    detection_range_max: float = 4.5
    #: Run preprocess_sonar before beam extraction for detection (Belief6).
    sonar_preprocess_for_detection: bool = False
    #: "beams" (per-beam peak) or "image_roi" (range-band column energy on raw image).
    detection_mode: str = "beams"
    #: Subtract column/beam median before argmax (clutter rejection).
    detection_col_contrast: bool = True
    #: image_roi: Gaussian weight toward band center (m); 0 = off.
    detection_range_weight_sigma: float = 0.0
    #: image_roi: ignore this many azimuth columns at each FOV edge.
    detection_azimuth_margin_cols: int = 0
    #: Prefer beam peak near EKF μ bearing within ±gate (reduces clutter argmax).
    detection_use_belief_prior: bool = False
    detection_prior_gate_deg: float = 35.0
    detection_prior_mu_norm_min: float = 0.35
    belief_lost_threshold: float = 2.0
    belief_process_noise: float = 0.05
    belief_lost_inflation: float = 1.02
    belief_lost_inflation_speed: float = 0.05
    #: Use log(1+σ) for σ channels in belief6 observations.
    belief_obs_log_sigma: bool = False
    #: Reward term scales (Belief6 minimal preset overrides these).
    reward_mode: str = "legacy"
    #: Covariance reward (``reward_mode == "covariance"``).
    r_cov_info: float = 0.35
    cov_logdet_scale: float = 6.0
    r_cov_detect: float = 0.10
    r_cov_miss: float = -0.15
    r_cov_center: float = 0.30
    cov_center_gate_deg: float = 25.0
    r_cov_smooth: float = 0.015
    covariance_det_eps: float = 1.0e-8
    reward_smooth_use_abs: bool = False
    #: Lost when not detected; if False, use σ trace only.
    reward_lost_requires_no_detect: bool = True
    reward_track_scale: float = -0.1
    reward_detect_positive: float = 1.0
    reward_detect_negative: float = -0.5
    #: When True, positive detect reward is multiplied by target_facing_quality in [0, 1].
    reward_detect_scale_by_facing: bool = False
    #: When True, positive detect reward is scaled by (1 - |det_az|/gate) when detected.
    reward_detect_scale_by_det_az: bool = False
    detection_centering_gate_deg: float = 25.0
    reward_action_smooth_scale: float = 0.01

    # --- Observations ---
    # "structured" (10-d) Phase 0 | "belief_sac" (262-d) for SAC + CNN features
    policy_obs_mode: str = "structured"
    sonar_feature_dim: int = 256
    sonar_history_len: int = 1

    # --- Reward (scaled/clipped for SAC stability) ---
    reward_scale: float = 0.08
    reward_clip_min: float = -10.0
    reward_clip_max: float = 15.0
    r_reacquire: float = 1.0
    #: Bonus per step while target is geometrically inside sonar FOV.
    r_in_fov: float = 0.0
    #: Penalty per step when target is outside sonar FOV (negative value, e.g. -0.25).
    r_out_of_fov: float = 0.0
    #: Reward scale × action × ref_azimuth when signs match (turn toward target).
    r_bearing_align: float = 0.0
    bearing_align_mu_norm_min: float = 0.35
    #: Extra penalty when action×bearing<0 and |bearing| above threshold (v9).
    r_wrong_turn: float = 0.0
    wrong_turn_az_threshold_deg: float = 15.0
    #: Minimum |action| before wrong-turn penalty applies.
    wrong_turn_min_action: float = 0.05
    #: Floor on |ref_az|/threshold scaling for wrong-turn penalty (small-offset turns).
    wrong_turn_az_weight_floor: float = 0.35
    #: Penalty scale for |det_az| when detected (-scale * |det_az| / gate).
    r_det_center: float = 0.0
    det_center_gate_deg: float = 20.0
    #: When det lost, use EKF μ bearing-rate × lead time as align/wrong ref (orbit).
    align_use_velocity_hint: bool = False
    bearing_lead_time: float = 0.4
    bearing_hint_max_deg: float = 20.0
    #: Use sin/cos(det_az) instead of raw azimuth in belief6 obs (8-d).
    belief6_use_sin_cos_az: bool = False
    #: Bonus for keeping target ahead in body frame (quality × r_facing_target; angle only, no range gate).
    r_facing_target: float = 0.3
    #: Half-width of "facing" cone around robot +X (deg); narrower than sonar FOV.
    tracking_bearing_max_deg: float = 25.0

    # --- Termination ---
    terminate_on_target_out_of_fov: bool = True
    sonar_fov_margin_deg: float = 2.0
    #: Consecutive steps out of FOV before reset (0 = immediate reset).
    target_out_of_fov_grace_steps: int = 0
    #: Reset episode when target orbit phase reaches ±target_orbit_world_phase_* limits.
    terminate_on_target_orbit_boundary: bool = True
    #: Phase tolerance (deg) for boundary termination.
    target_orbit_boundary_eps_deg: float = 2.0
    #: Fallback: consecutive steps without beam detection (only if terminate_on_target_out_of_fov=False).
    max_out_of_view_steps: int = 1

    # --- Debug logging (off during training) ---
    print_entity_positions: bool = False
    print_entity_positions_interval: int = 20
    print_sonar_obs_debug: bool = False
    #: Print raw / processed sonar stats + beam peak (Belief6 play debug).
    print_sonar_pipeline_debug: bool = False
    #: Save raw/processed PNG + peak marker when pipeline debug is on.
    sonar_debug_save_images: bool = False
    sonar_debug_save_dir: str = "logs/nonar2_sonar_debug"
    #: Print EKF μ, σ, innovation vs detection, and μ vs GT (play / tune Module C).
    print_belief_pipeline_debug: bool = False
    #: Print policy observation vector (Belief6 7-d or belief_sac 262-d).
    print_policy_obs_debug: bool = False
    #: Print per-term reward breakdown (Module E).
    print_reward_debug: bool = False
    #: Print episode evaluation summary (detection rate, align rate, bearing error).
    print_eval_debug: bool = False
    #: Silence eval console prints (only log to TensorBoard).
    eval_silent: bool = False
    #: Directory for TensorBoard custom eval scalars. Set by train/play script.
    tb_log_dir: str = ""
    #: Compact play log: net policy action vs executed ω (one line per interval).
    print_play_debug: bool = False
