"""AUV Target Tracking environment for IsaacLab.

Open-water belief tracking (HoloOcean v0 core, without obstacles).
Physics from isaac-auv-env WarpAUV; agent control matches HoloOcean PID:
  RL action [a_v, a_ω] ∈ [-1,1] → cmd_vel [v, ω] → PID → 6 PWM.
Observation: [r_b, alpha_b, log_det_cov, observed] (4D)

KF measurement: OceanSim sonar (belief-gated polar intensity centroid + range bias),
not geometric GT. Policy obs / action / reward interfaces unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import torch

from .assets.warpauv import WARPAUV_CFG
from .batch_lqr import BatchHoloTargetMotion
from .batch_pid import BatchPIDController
from .kalman_filter_batch import BatchKalmanFilter, BatchUnscentedKalmanFilter
from .ocean_current import clip_current_velocity, create_ocean_current_field
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .sonar_meas import body_polar_from_xy, detect_centroid_polar
from .thruster_dynamics import (
    DynamicsFirstOrder,
    ConversionFunctionBasic,
    get_gsrov_thruster_geometry,
    get_thruster_com_and_orientations,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, sample_uniform

from isaaclab_tasks.direct.nonar2.oceansim_sonar_sensor_cfg import OceanSimSonarSensorCfg
from isaaclab_tasks.direct.sonar_centroid_track.auv_platform import (
    AUV_SONAR_HOUSING_OFFSET_LOCAL,
    AUV_SONAR_MOUNT_OFFSET_LOCAL,
    AUV_SONAR_MOUNT_VISUAL,
)
from isaaclab_tasks.direct.sonar_centroid_track.modules.sonar_gt import SonarGtDetector
from isaaclab_tasks.direct.sonar_centroid_track.modules.sonar_pipeline import PolarSonarPipeline
from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import sonar_pos_w
from isaaclab_tasks.direct.sonar_flow_hover.shared_platform import (
    OCEANSIM_SONAR_CFG,
    OceanSimSonarSensor,
    mount_oceansim_sonar_on_robots,
)


@configclass
class OceanCurrentCfg:
    """Optional ocean currents (default off = calm water). Holo-like water velocity API."""

    enabled: bool = False
    type: str = "uniform"  # uniform | vortex | sine_wave | random
    strength: float = 0.3
    level: str | None = None  # off|mild|medium|strong; None → use strength
    affect_target: bool = False
    clip_speed: float = 5.0
    # Holo v0 (`v0_config_fast.yml` ocean_current.params): random field, max_velocity=2.0,
    # spatial_scale=5.0, seed=42. Keys are pre-declared so Hydra CLI overrides merge cleanly.
    params: dict = {
        "direction": [1.0, 0.0, 0.0],
        "max_velocity": 2.0,
        "spatial_scale": 5.0,
        "seed": 42.0,
    }


@configclass
class AUVTrackingEnvCfg(DirectRLEnvCfg):
    # Control period ≈ 0.04 s, matching the bridge's ~25 Hz MANUAL_CONTROL rate (logged P50
    # 46 ms ≈ 21.8 Hz; use 25 Hz here so the sim's policy/timestamps line up with the PC's
    # send cadence, and the flow per-step is naturally 5× smaller). Render interval equals
    # decimation so camera updates once per policy step (no stale frames between steps).
    decimation = 5
    # Holo make()/examples commonly use t_steps=200 @ 0.2 s → 40 s
    episode_length_s = 40.0
    # HoloOcean v0 PID: action_dim=2 in [-1,1], then scale [1.5, 0.5].
    # Must be bounded Box — int→Box(±inf) can make (high-low)*u NaN in some RL wrappers.
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    observation_space = 4
    state_space = 0
    action_scale_v = 1.5
    action_scale_w = 0.5
    # Per-env-step slew limit on cmd_vel [v, w] (same units). 0.0 = off (legacy).
    # Cuts bang-bang without touching plant dynamics (unlike thruster lag, which
    # cost too much phase: lag-0.2 run peaked then went timid, 539 -> 131).
    cmd_rate_limit: float = 0.0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    robot_cfg: RigidObjectCfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Holo map size 40×40 (half ±20); spacing > 2*half
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=45.0, replicate_physics=True)
    # Frame AUV/ball at starting_depth (default viewer looks at ground z=0)
    viewer: ViewerCfg = ViewerCfg(eye=(18.0, 18.0, 16.0), lookat=(0.0, 0.0, 8.0))
    debug_vis = False
    use_boundaries = True
    # Holo scenario size [40,40,20], bottom_corner [-20,-20,-20]
    max_auv_x = 20
    max_auv_y = 20
    max_auv_z = 7
    starting_depth = 8
    # Holo margin2wall: in_bound shrinks workspace by this margin
    margin2wall = 1.0

    # Target / sensing (v0-like; no obstacle channels)
    # Sphere teleported each step from 2D LQR (same pattern as lh_2/nonar2 beacon).
    # Do NOT use kinematic_enabled: on GPU PhysX the visual mesh often stays frozen
    # while root_pos_w buffers still update (false "moving" in smoke logs).
    target_radius = 0.35
    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.35,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.25, 0.05)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=2.0,
                angular_damping=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, 0.0, 8.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    # Holo RRT_2d + LQR chase stand-in (open water)
    target_speed = 0.5  # RRT_2d.speed
    target_max_speed = 0.6
    target_waypoint_reach_dist = 0.3  # RRT_2d tick threshold
    target_rrt_step_size = 2.0
    target_tau_vel = 0.7  # first-order lag ≈ ROV inertia
    # --- per-episode target speed envelope ---
    # All three default to "v0 exactly": spread 1 => cruise stays the constant target_speed,
    # peak factor 0 => the fixed target_max_speed is used, max_accel 0 => no acceleration limit.
    # Raising them makes the target livelier, which is a *deliberate* departure from Holo
    # (Holo's RRT_2d runs a constant 0.5 m/s against the agent's 1.5 m/s action scale).
    #
    # target_cruise_spread: cruise ~ U[target_speed, target_speed * spread] resampled per episode.
    #   BlueROV-ish intent is U[0.5, 1.0] => spread 2.0.
    # target_peak_factor: speed ceiling as a multiple of the episode's cruise (1.2 = +20%).
    #   Applied on top of the soft-start ramp, so it cannot make the target burst at t=0.
    # target_max_accel: |dv/dt| in m/s^2 (BlueROV envelope ~1.0). The soft start already caps
    #   early acceleration at cruise / target_soft_start_s, so this is a later-phase bound only.
    target_cruise_spread = 1.0
    target_peak_factor = 0.0
    target_max_accel = 0.0
    # Episode target-motion schedule (from reset). Kept at v0 values on purpose: the target
    # ramps from rest, stays leashed near the agent, and only then runs free -- that is what
    # gives the policy a chance to catch up, so it must not be shortened when speed goes up.
    #   0–3s  冷启动: cruise 0→target_speed + leash
    #   3–5s  easy:   full cruise + leash
    #   5s+   正常:   free RRT, no leash
    target_soft_start_s = 3.0
    target_easy_s = 5.0
    target_leash_range = 3.2  # < sensor_range=4; leave margin for FOV / lag
    # Bearing of the target about the agent's heading. 180 leaves the full circle (Holo / v0).
    target_bearing_limit_deg = 180.0
    sensor_range = 4.0
    sensor_fov_deg = 100.0
    sensor_r_sd = 0.2
    sensor_b_sd = 0.01
    # OceanSim FLS → KF measurement (adapted for r=0.35 tracking sphere).
    # Cap sonar max_range to Holo sensor_range (4 m): far bins were latching clutter (~6 m).
    # height = (max_range - min_range) / range_res = (4.0 - 0.5) / 0.05 = 70.
    oceansim_sonar_cfg: OceanSimSonarSensorCfg = OCEANSIM_SONAR_CFG.replace(
        max_range=4.0,
        height=70,
        intensity_gain=1.8,
        attenuation=0.05,
    )
    sonar_mount_offset_local: tuple[float, float, float] = AUV_SONAR_MOUNT_OFFSET_LOCAL
    sonar_housing_offset_local: tuple[float, float, float] = AUV_SONAR_HOUSING_OFFSET_LOCAL
    sonar_mount_visual: bool = AUV_SONAR_MOUNT_VISUAL
    sonar_mount_rpy_deg: tuple[float, float, float] | None = None
    skip_median_filter: bool = True
    sonar_warmup_frames: int = 4
    sonar_prior_gate_deg: float = 20.0
    # Acoustic surface → sphere center. Debug used 0.29 m on r=0.5 m; scale for r=0.35 m.
    sonar_range_bias_m: float = 0.20
    sonar_range_window_m: float = 0.8
    sonar_threshold_ratio: float = 0.25
    sonar_min_pixels: int = 2
    # Reject weak polar peaks (play clutter ~0.37; true ball ~0.69). 0 → disabled.
    sonar_min_peak: float = 0.45
    # OceanSim reflectivity semantic on Target mesh (default unlabelled=1.0).
    sonar_target_reflectivity: float = 2.0
    # After first lock: reject peaks far from KF prior (stops max-range clutter latch).
    sonar_consist_range_m: float = 1.5
    # Before first good hit in episode: allow ungated detect inside Holo FOV/range.
    sonar_cold_start_ungated: bool = True
    target_init_cov = 50.0
    # Put the ball near the center of the sonar lobe so the first ping is likely
    # to hit. Sensor is 4 m / 100° FOV; Holo a2t was [1,4] × ±π/5 (range edge).
    target_init_dist_min = 1.5
    target_init_dist_max = 2.5
    target_init_ang_min = -math.pi / 10.0
    target_init_ang_max = math.pi / 10.0
    # Holo belief init: lin/ang_dist_range_t2b from true target
    belief_init_dist_min = 0.0
    belief_init_dist_max = 2.0
    belief_init_ang_min = -math.pi / 2.0
    belief_init_ang_max = math.pi / 2.0
    # Holo agent_init_yaw ~ U[-π/2, π/2]
    agent_init_yaw_min = -math.pi / 2.0
    agent_init_yaw_max = math.pi / 2.0
    # Holo KF vel clip
    kf_vel_limit = 3.0
    const_q = 0.5
    kf_type = "ekf"

    # Reward: Holo v0 reward_param (smooth term is commented out in Holo code → 0)
    rew_scale_uncertainty_mean = 0.2
    rew_scale_uncertainty_std = 0.0
    rew_scale_smooth = 0.0
    # Match AUV XY speed to target (0 = off, Holo default). Symmetric: rewards tracking the
    # target's speed (lag AND lead both hurt). Capability headroom to catch up when the target
    # speeds up comes from action_scale_v ≫ target cruise, not from preferring to go faster.
    # rew_scale_lag is an optional asymmetric alternative; leave it 0 when this is on.
    rew_scale_speed = 0.0
    rew_speed_sigma = 0.5
    # --- optional geometry shaping (all default 0 ⇒ v0 / Holo bit-identical) ---
    # Preferred standoff (m). Gaussian on |gt_r - r*|; stops the "see target anywhere in FOV"
    # saturation of pure -logdet from drifting out to sensor_range and then losing lock.
    rew_scale_range: float = 0.0
    rew_range_star: float = 2.0
    rew_range_sigma: float = 0.6
    # Bearing centering: Gaussian on |gt_α| / fov_half. Rewards keeping the target near image
    # centre so the vision encoder's spatial signal is used, not just "anywhere inside FOV".
    rew_scale_bearing: float = 0.0
    rew_bearing_sigma_frac: float = 0.45  # σ as a fraction of fov_half (~0.45 → soft at ~20° when half=45°)
    # Asymmetric lag-only speed term (optional). Prefer rew_scale_speed for tracking; use this
    # only if you explicitly want "never punish being faster". Default 0.
    rew_scale_lag: float = 0.0
    rew_lag_sigma: float = 0.3
    # Holo is_col: proximity / OOB → min(0, r) - c_penalty (does NOT terminate)
    rew_c_penalty = 5.0
    proximity_margin = 1.0
    # Holo: rolling mean reward over 100 steps < -3.5 → done
    # 100 steps @ 5 Hz = 20 s done-window → keep wall-clock same: 500 steps @ 25 Hz
    rew_done_window = 500
    rew_done_threshold = -3.5

    # WarpAUV dynamics
    com_to_cob_offset = [0.0, 0.0, 0.01]
    water_rho = 997.0
    water_beta = 0.001306
    rotor_constant = 0.1 / 100.0
    dyn_time_constant = 0.05
    # First-order thruster lag. False keeps the historical instantaneous response
    # (DynamicsFirstOrder alpha forced to 0); True makes dyn_time_constant a real knob.
    thruster_first_order_lag: bool = False
    # Thruster set + thrust model.
    #   "warpauv6" - the calibrated WarpAUV proxy (6 thrusters, quadratic PWM curve).
    #   "gsrov8"   - the real GS-ROV hull (4 vertical + 4 horizontal, T ~ u*|u| in Newtons).
    thruster_layout: str = "warpauv6"
    # ROVMAKER subsea thruster, vendor datasheet at 6S/24 V: 1000-2000 us pulse width,
    # 1500 us neutral, 2.0 kgf forward / 1.8 kgf reverse. These are *datasheet* numbers,
    # not measured on our hull (see ROV真机测试总结_20261101.md).
    thruster_max_thrust_fwd_n: float = 19.613
    thruster_max_thrust_rev_n: float = 17.652
    thruster_neutral_us: float = 1500.0
    thruster_span_us: float = 500.0
    thruster_rated_voltage_v: float = 24.0
    # Propeller thrust goes as rpm^2 and rpm goes as pack voltage, so T_max ~ (V/V_rated)^2.
    # The bench pack measured 22.2-23.1 V, i.e. ~87 % of rated. Left at rated by default so
    # the calibrated dynamics stay unchanged; randomise it (3S-6S is a 4x range) for robustness.
    thruster_battery_voltage_v: float = 24.0
    thruster_deadband: float = 0.05
    volume = 0.022747843530591776
    mass = 2.2701e01
    # Principal inertia diagonal [Ixx, Iyy, Izz] (kg*m^2) used by the quadratic-drag
    # and viscous-torque models. Defaults are the WarpAUV values; subclasses that swap
    # the hull (e.g. GS-ROV) override this.
    inertia_diag: tuple = (0.37, 0.97, 1.19)

    # Ocean currents (default calm water). See docs/superpowers/specs/2026-08-18-auv-ocean-current-design.md
    ocean_current: OceanCurrentCfg = OceanCurrentCfg()
    # KF measurement: "sonar" (OceanSim), "gt" (geometric polar), or "cam" (v4 RGB, see cam_meas)
    kf_measurement: str = "sonar"
    # Target trim attitude (rad) held by the low-level attitude PID. A free-floating hull
    # settles nose-up by atan(com_to_cob_x / com_to_cob_z) due to buoyancy torque; command
    # that trim so the vertical thrusters do not fight the natural floating attitude.
    trim_roll_rad: float = 0.0
    trim_pitch_rad: float = 0.0


class AUVTrackingEnv(DirectRLEnv):
    cfg: AUVTrackingEnvCfg

    def _use_gt_meas(self) -> bool:
        return str(getattr(self.cfg, "kf_measurement", "sonar")).lower() == "gt"

    def _use_cam_meas(self) -> bool:
        return str(getattr(self.cfg, "kf_measurement", "sonar")).lower() == "cam"

    def _skip_sonar_pipeline(self, cfg: AUVTrackingEnvCfg | None = None) -> bool:
        c = self.cfg if cfg is None else cfg
        return str(getattr(c, "kf_measurement", "sonar")).lower() in ("gt", "cam")

    def __init__(self, cfg: AUVTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        n_envs = cfg.scene.num_envs
        # Device may be finalized inside DirectRLEnv; use cfg hint then refresh after super.
        dev = getattr(cfg.sim, "device", "cuda:0")
        self._sonar_warmed = False
        self._sonar_locked = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        skip_sonar = self._skip_sonar_pipeline(cfg)
        if not skip_sonar:
            self.pipeline = PolarSonarPipeline(cfg, n_envs, dev)
            self.gt = SonarGtDetector(
                threshold_ratio=float(cfg.sonar_threshold_ratio),
                min_pixels=int(cfg.sonar_min_pixels),
                polar_width=int(cfg.oceansim_sonar_cfg.width),
            )
            ox, oy, oz = cfg.sonar_mount_offset_local
            self._sonar_mount_offset_b = (
                torch.tensor((ox, oy, oz), device=dev, dtype=torch.float32).unsqueeze(0).repeat(n_envs, 1)
            )
        else:
            self.pipeline = None
            self.gt = None
            self._sonar_mount_offset_b = None

        super().__init__(cfg, render_mode, **kwargs)
        self._sonar_locked = self._sonar_locked.to(self.device)

        if not self._skip_sonar_pipeline():
            self.pipeline.device = self.device
            self.pipeline.sonar_image = self.pipeline.sonar_image.to(self.device)
            self.pipeline.polar_image = self.pipeline.polar_image.to(self.device)
            self._sonar_mount_offset_b = self._sonar_mount_offset_b.to(self.device)
            self.pipeline.warmup(self.oceansim_sonar, self.sim, n_frames=int(self.cfg.sonar_warmup_frames))
            self._sonar_warmed = True

        # Ocean current buffer + optional field (Holo-like set_ocean_currents API)
        self._ocean_current_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._ocean_current_field = create_ocean_current_field(self.cfg.ocean_current)

        # RL action [a_v, a_ω]; low-level PWM from PID
        self._actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._cmd_vel = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_cmd_vel = torch.zeros(self.num_envs, 2, device=self.device)
        self.thruster_layout = str(getattr(self.cfg, "thruster_layout", "warpauv6")).lower()
        if self.thruster_layout == "gsrov8":
            geometry = get_gsrov_thruster_geometry(self.device)
        else:
            self.thruster_layout = "warpauv6"
            geometry = get_thruster_com_and_orientations(self.device)
        self.num_thrusters = int(geometry[0].shape[0])
        self._pwm = torch.zeros(self.num_envs, self.num_thrusters, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._target_state = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._observed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._needs_belief_refresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_col = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cached_rew = torch.zeros(self.num_envs, device=self.device)
        # Holo reward_queue (per-env rolling window for early terminate)
        self._rew_window = int(self.cfg.rew_done_window)
        self._rew_hist = torch.zeros(self.num_envs, self._rew_window, device=self.device)
        self._rew_hist_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._rew_hist_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._control_dt = self.cfg.sim.dt * self.cfg.decimation
        self._fov_half = math.radians(self.cfg.sensor_fov_deg * 0.5)

        self.kf = self._init_kf()

        base_offsets, base_quats = geometry
        self.thruster_com_offsets = base_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = base_quats.repeat(self.num_envs, 1)

        self._gravity_mag = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        inr = tuple(getattr(self.cfg, "inertia_diag", (0.37, 0.97, 1.19)))
        self.inertia_tensors[:, 0] = float(inr[0])
        self.inertia_tensors[:, 1] = float(inr[1])
        self.inertia_tensors[:, 2] = float(inr[2])
        self.masses = torch.full((self.num_envs, 1), self.cfg.mass, device=self.device)
        self.com_to_cob_offsets = torch.tensor(self.cfg.com_to_cob_offset, device=self.device).repeat(self.num_envs, 1)
        self.volumes = torch.full((self.num_envs, 1), self.cfg.volume, device=self.device)

        self.force_calc = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_dyn = DynamicsFirstOrder(
            self.num_envs,
            self.num_thrusters,
            self.cfg.dyn_time_constant,
            self.device,
            enable_lag=bool(getattr(self.cfg, "thruster_first_order_lag", False)),
        )
        self.thruster_conv = ConversionFunctionBasic(self.cfg.rotor_constant)

        # Rated thrust scaled by (V/V_rated)^2; see the cfg docstrings for provenance.
        self._rated_voltage = float(getattr(self.cfg, "thruster_rated_voltage_v", 24.0))
        self._battery_voltage = float(getattr(self.cfg, "thruster_battery_voltage_v", self._rated_voltage))
        voltage_scale = (self._battery_voltage / max(self._rated_voltage, 1e-6)) ** 2
        self._thrust_fwd_n = float(getattr(self.cfg, "thruster_max_thrust_fwd_n", 19.613)) * voltage_scale
        self._thrust_rev_n = float(getattr(self.cfg, "thruster_max_thrust_rev_n", 17.652)) * voltage_scale

        self.pid = BatchPIDController(
            num_envs=self.num_envs,
            device=self.device,
            dt=float(self.cfg.sim.dt),
            thruster_layout=self.thruster_layout,
            thruster_offsets=base_offsets if self.thruster_layout == "gsrov8" else None,
            thruster_quats=base_quats if self.thruster_layout == "gsrov8" else None,
            trim_roll_rad=float(getattr(self.cfg, "trim_roll_rad", 0.0)),
            trim_pitch_rad=float(getattr(self.cfg, "trim_pitch_rad", 0.0)),
        )
        self.target_motion = BatchHoloTargetMotion(
            num_envs=self.num_envs,
            device=self.device,
            cruise_speed=self.cfg.target_speed,
            waypoint_reach_dist=self.cfg.target_waypoint_reach_dist,
            rrt_step_size=self.cfg.target_rrt_step_size,
            tau_vel=self.cfg.target_tau_vel,
            max_speed=self.cfg.target_max_speed,
            soft_start_s=float(self.cfg.target_soft_start_s),
            easy_s=float(self.cfg.target_easy_s),
            leash_range=float(self.cfg.target_leash_range),
            workspace_half_xy=(float(self.cfg.max_auv_x), float(self.cfg.max_auv_y)),
            workspace_margin=float(self.cfg.margin2wall),
            cruise_speed_spread=float(getattr(self.cfg, "target_cruise_spread", 1.0)),
            speed_peak_factor=float(getattr(self.cfg, "target_peak_factor", 0.0)),
            max_accel=float(getattr(self.cfg, "target_max_accel", 0.0)),
            bearing_limit_rad=math.radians(float(getattr(self.cfg, "target_bearing_limit_deg", 180.0))),
        )
        self._reset_idx(self._robot._ALL_INDICES)

    def _init_kf(self):
        kwargs = dict(
            num_envs=self.num_envs,
            dim=4,
            device=self.device,
            dt=self._control_dt,
            const_q=self.cfg.const_q,
            sensor_r_sd=self.cfg.sensor_r_sd,
            sensor_b_sd=self.cfg.sensor_b_sd,
        )
        if self.cfg.kf_type == "ukf":
            return BatchUnscentedKalmanFilter(**kwargs)
        return BatchKalmanFilter(**kwargs)

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self.cfg.target_cfg.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(3.0, 0.0, self.cfg.starting_depth),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
        # Keep sphere radius in sync with cfg.target_radius
        if self.cfg.target_cfg.spawn is not None and hasattr(self.cfg.target_cfg.spawn, "radius"):
            self.cfg.target_cfg.spawn.radius = self.cfg.target_radius
        self._robot = RigidObject(self.cfg.robot_cfg)
        self._target = RigidObject(self.cfg.target_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["robot"] = self._robot
        self.scene.rigid_objects["target"] = self._target
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.oceansim_sonar = None
        if not self._skip_sonar_pipeline():
            self.oceansim_sonar = OceanSimSonarSensor(
                self.cfg.oceansim_sonar_cfg,
                num_envs=self.num_envs,
                device=self.device,
            )
            mount_oceansim_sonar_on_robots(
                self.oceansim_sonar,
                num_envs=self.num_envs,
                sonar_mount_offset_local=self.cfg.sonar_mount_offset_local,
                sim=self.sim,
                sonar_mount_rpy_deg=getattr(self.cfg, "sonar_mount_rpy_deg", None),
                sonar_mount_visual=bool(getattr(self.cfg, "sonar_mount_visual", False)),
                sonar_housing_offset_local=getattr(self.cfg, "sonar_housing_offset_local", None),
            )
            self._apply_target_reflectivity()

    def _apply_target_reflectivity(self) -> None:
        """Tag Target meshes with OceanSim reflectivity semantic (boost small-sphere SNR)."""
        refl = float(getattr(self.cfg, "sonar_target_reflectivity", 0.0))
        if refl <= 0.0:
            return
        try:
            from isaacsim.core.utils.prims import get_prim_at_path
            from isaacsim.core.utils.semantics import add_update_semantics
            from pxr import Usd, UsdGeom
        except ImportError:
            return
        label = f"{refl:g}"
        for i in range(self.num_envs):
            root_path = f"/World/envs/env_{i}/Target"
            try:
                root = get_prim_at_path(root_path)
            except Exception:
                continue
            if root is None or not root.IsValid():
                continue
            mesh_prims = [
                prim
                for prim in Usd.PrimRange(root)
                if prim.IsA(UsdGeom.Mesh) or prim.GetTypeName() in ("Mesh", "Sphere")
            ]
            if not mesh_prims:
                mesh_prims = [root]
            for prim in mesh_prims:
                try:
                    add_update_semantics(prim=prim, type_label="reflectivity", semantic_label=label)
                except Exception:
                    pass

    def _sync_target_entity(self, env_ids: torch.Tensor | None = None) -> None:
        """Write LQR kinematic state onto the visible sphere (world frame)."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        n = len(env_ids)
        pose = torch.zeros(n, 7, device=self.device)
        pose[:, 0] = self._target_pos[env_ids, 0]
        pose[:, 1] = self._target_pos[env_ids, 1]
        pose[:, 2] = self.cfg.starting_depth
        pose[:, 3] = 1.0  # identity quat w
        vel = torch.zeros(n, 6, device=self.device)
        vel[:, 0] = self._target_state[env_ids, 2]
        vel[:, 1] = self._target_state[env_ids, 3]
        self._target.write_root_pose_to_sim(pose, env_ids)
        self._target.write_root_velocity_to_sim(vel, env_ids)

    def _agent_pose_2d(self) -> tuple[torch.Tensor, torch.Tensor]:
        agent_pos = self._robot.data.root_pos_w[:, :2]
        _, _, agent_yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        return agent_pos, agent_yaw

    def _sonar_pose_xy_yaw(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sonar mount XY + robot yaw (body frame of measurement)."""
        sonar_pw = sonar_pos_w(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._sonar_mount_offset_b,
        )
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        return sonar_pw[:, :2], yaw

    def _ensure_sonar(self) -> None:
        if not self._sonar_warmed:
            self.pipeline.warmup(self.oceansim_sonar, self.sim, n_frames=int(self.cfg.sonar_warmup_frames))
            self._sonar_warmed = True
        self.pipeline.ensure_updated(self.common_step_counter, self.oceansim_sonar, self.sim)

    def _detect_sonar_polar(
        self,
        *,
        prior_azimuth: torch.Tensor | None,
        prior_range: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ocfg = self.cfg.oceansim_sonar_cfg
        return detect_centroid_polar(
            self.pipeline.polar_image,
            self.gt,
            prior_azimuth=prior_azimuth,
            prior_range=prior_range,
            prior_gate_deg=float(self.cfg.sonar_prior_gate_deg),
            hori_fov_deg=float(ocfg.hori_fov),
            min_range=float(ocfg.min_range),
            max_range=float(ocfg.max_range),
            range_window_m=float(self.cfg.sonar_range_window_m),
            detect_max_range=float(self.cfg.sensor_range),
        )

    def _update_belief_from_camera(self, env_ids: torch.Tensor) -> None:
        raise NotImplementedError("kf_measurement='cam' is implemented on AUVTrackingV4CamEnv")

    def _update_belief(self, env_ids: torch.Tensor | None = None) -> None:
        """Measure (r,α) → KF update. No predict (Holo order)."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        if self._use_gt_meas():
            agent_pos, agent_yaw = self._agent_pose_2d()
            z_all, r_true, alpha_true = self.kf.get_observation(self._target_pos, agent_pos, agent_yaw)
            in_range = r_true <= float(self.cfg.sensor_range)
            in_fov = torch.abs(alpha_true) <= self._fov_half
            observed = in_range & in_fov
            self._observed[env_ids] = observed[env_ids]
            self._sonar_locked[env_ids] = observed[env_ids]
            obs_ids = env_ids[observed[env_ids]]
            if obs_ids.numel() > 0:
                self.kf.update(z_all[obs_ids], agent_pos[obs_ids], agent_yaw[obs_ids], env_ids=obs_ids)
            return

        if self._use_cam_meas():
            self._update_belief_from_camera(env_ids)
            return

        self._ensure_sonar()
        sonar_xy, yaw = self._sonar_pose_xy_yaw()

        # Gate from current KF state (prior for this step after last predict).
        az_pred, r_pred = body_polar_from_xy(self.kf.state[:, :2], sonar_xy, yaw)
        range_bias = float(self.cfg.sonar_range_bias_m)
        r_roi = torch.clamp(r_pred - range_bias, min=1.0e-3)

        # Cold start: belief t2b init can be far from agent view → gated ROI misses / latches clutter.
        # Until first accepted hit, use ungated detect (same idea as sonar_centroid debug).
        cold = ~self._sonar_locked
        if bool(self.cfg.sonar_cold_start_ungated) and cold.any():
            r_g, a_g, valid_g = self._detect_sonar_polar(prior_azimuth=None, prior_range=None)
        else:
            r_g = a_g = valid_g = None

        r_s, a_s, valid = self._detect_sonar_polar(prior_azimuth=az_pred, prior_range=r_roi)
        if r_g is not None:
            r_s = torch.where(cold, r_g, r_s)
            a_s = torch.where(cold, a_g, a_s)
            valid = torch.where(cold, valid_g, valid)

        r_corr = r_s + range_bias

        # After lock: drop peaks inconsistent with KF prior (max-range clutter).
        consist_r = float(self.cfg.sonar_consist_range_m)
        gate_rad = math.radians(float(self.cfg.sonar_prior_gate_deg) * 1.5)
        d_alpha = torch.atan2(torch.sin(a_s - az_pred), torch.cos(a_s - az_pred)).abs()
        consistent = (torch.abs(r_corr - r_pred) <= consist_r) & (d_alpha <= gate_rad)
        accept_geom = (valid > 0.5) & (r_corr <= self.cfg.sensor_range) & (torch.abs(a_s) <= self._fov_half)
        # Weak polar floor ≈ clutter latch (~0.37); true ball returns are stronger.
        min_peak = float(self.cfg.sonar_min_peak)
        if min_peak > 0.0 and hasattr(self.pipeline, "polar_image"):
            peak = self.pipeline.polar_image.amax(dim=(-2, -1))
            accept_geom = accept_geom & (peak >= min_peak)
        observed = accept_geom & (cold | consistent)

        self._observed[env_ids] = observed[env_ids]
        # Unlock on miss so a clutter-locked prior cannot keep accepting weak peaks.
        self._sonar_locked[env_ids] = observed[env_ids]

        obs_ids = env_ids[observed[env_ids]]
        if obs_ids.numel() > 0:
            z = torch.stack([r_corr[obs_ids], a_s[obs_ids]], dim=-1)
            self.kf.update(z, sonar_xy[obs_ids], yaw[obs_ids], env_ids=obs_ids)

    def _log_det_cov(self) -> torch.Tensor:
        # Symmetric jitter before det to avoid negative/NaN from roundoff
        cov = 0.5 * (self.kf.cov + self.kf.cov.transpose(1, 2))
        eye = torch.eye(4, device=self.device)
        det = torch.linalg.det(cov + 1e-6 * eye)
        return torch.log(det.clamp(min=1e-12))

    def _build_policy_obs(self) -> torch.Tensor:
        agent_pos, agent_yaw = self._agent_pose_2d()
        belief_pos = self.kf.state[:, :2]
        diff = belief_pos - agent_pos
        cos_yaw = torch.cos(agent_yaw)
        sin_yaw = torch.sin(agent_yaw)
        dx = cos_yaw * diff[:, 0] + sin_yaw * diff[:, 1]
        dy = -sin_yaw * diff[:, 0] + cos_yaw * diff[:, 1]
        r_b = torch.sqrt(dx**2 + dy**2 + 1e-10)
        alpha_b = torch.atan2(dy, dx)
        obs = torch.stack([r_b, alpha_b, self._log_det_cov(), self._observed.float()], dim=-1)
        return torch.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions[:] = self._actions
        self._actions[:] = torch.clip(actions, -1.0, 1.0)
        # Surge maps a_v in [-1, 1] onto [action_v_min, action_v_max].
        # Missing fields fall back to ±action_scale_v, which is v = a_v * action_scale_v (Holo v0).
        v_min = float(getattr(self.cfg, "action_v_min", -self.cfg.action_scale_v))
        v_max = float(getattr(self.cfg, "action_v_max", self.cfg.action_scale_v))
        a_v = self._actions[:, 0]
        self._cmd_vel[:, 0] = v_min + (a_v + 1.0) * 0.5 * (v_max - v_min)
        self._cmd_vel[:, 1] = self._actions[:, 1] * self.cfg.action_scale_w
        limit = float(getattr(self.cfg, "cmd_rate_limit", 0.0))
        if limit > 0.0:
            lo = self._prev_cmd_vel - limit
            hi = self._prev_cmd_vel + limit
            self._cmd_vel[:] = torch.min(torch.max(self._cmd_vel, lo), hi)
        self._prev_cmd_vel[:] = self._cmd_vel

    def _apply_action(self) -> None:
        t = None
        if self._ocean_current_field is not None:
            t = self.episode_length_buf.to(dtype=torch.float32) * float(self.sim.cfg.dt)
            v = self._ocean_current_field(self._robot.data.root_pos_w, t)
            self.set_ocean_currents(None, v)

        current_xy = None
        if self._ocean_current_field is not None and self.cfg.ocean_current.affect_target:
            tgt_pos = torch.zeros((self.num_envs, 3), device=self.device)
            tgt_pos[:, :2] = self._target_state[:, :2]
            tgt_pos[:, 2] = float(self.cfg.starting_depth)
            v_tgt = self._ocean_current_field(tgt_pos, t)
            current_xy = v_tgt[:, :2]

        # Target at physics rate (Holo: RRT+LQR every sim tick inside RL step)
        _, agent_yaw_now = self._agent_pose_2d()
        self._target_state[:] = self.target_motion.step(
            self._target_state,
            dt=float(self.cfg.sim.dt),
            agent_xy=self._robot.data.root_pos_w[:, :2],
            agent_yaw=agent_yaw_now,
            current_vel_xy=current_xy,
        )
        self._target_pos[:] = self._target_state[:, 0:2]
        self._sync_target_entity()
        # PID at physics rate (like HoloOcean ticks_per_rl_step inner loop)
        self._pwm[:] = self.pid.compute(
            self._cmd_vel,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_w,
            self._robot.data.root_ang_vel_w,
            dt=float(self.cfg.sim.dt),
        )
        self._thrust[:, 0, :], self._moment[:, 0, :] = self._compute_dynamics(self._pwm)
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

    def _compute_is_col(self) -> torch.Tensor:
        """Holo is_col without obstacles: OOB (margin2wall) or too close to target."""
        agent_xy = self._robot.data.root_pos_w[:, :2]
        dist_t = torch.norm(agent_xy - self._target_pos, dim=-1)
        too_close = dist_t <= self.cfg.proximity_margin
        out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.use_boundaries:
            # Holo in_bound: shrink half-extent by margin2wall
            half_x = float(self.cfg.max_auv_x) - float(self.cfg.margin2wall)
            half_y = float(self.cfg.max_auv_y) - float(self.cfg.margin2wall)
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > half_x)
                | (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > half_y)
                | (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        return out_of_bounds | too_close

    def _set_kf_limits(self, env_ids: torch.Tensor) -> None:
        """Holo target_limit: workspace XY ± (half - margin2wall), vel ± kf_vel_limit."""
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids, :2]
        half_x = float(self.cfg.max_auv_x) - float(self.cfg.margin2wall)
        half_y = float(self.cfg.max_auv_y) - float(self.cfg.margin2wall)
        vlim = float(self.cfg.kf_vel_limit)
        lo = torch.zeros(n, 4, device=self.device)
        hi = torch.zeros(n, 4, device=self.device)
        lo[:, 0] = origins[:, 0] - half_x
        lo[:, 1] = origins[:, 1] - half_y
        lo[:, 2] = -vlim
        lo[:, 3] = -vlim
        hi[:, 0] = origins[:, 0] + half_x
        hi[:, 1] = origins[:, 1] + half_y
        hi[:, 2] = vlim
        hi[:, 3] = vlim
        self.kf.set_limits(env_ids, lo, hi)

    def _compute_step_reward(self) -> torch.Tensor:
        """Holo get_reward core on posterior (belief already updated).

        Extra geometry terms (range / bearing / lag) default to scale 0 so Direct-v0 stays
        bit-identical to Holo. Direct-v4-cam turns them on: pure -logdet saturates once the
        target is merely inside the FOV, and headless diagnostics showed off-centre bearing,
        range drift to >4 m lock-loss, and ~24% time spent slower than the target.
        """
        r_uncertainty = -self._log_det_cov()
        # std term unused when rew_scale_uncertainty_std=0 (Holo c_std=0)
        action_diff = torch.norm(self._actions - self._prev_actions, dim=-1)
        r_smooth = torch.exp(-action_diff)
        auv_xy = self._robot.data.root_lin_vel_w[:, :2]
        tgt_xy = self._target_state[:, 2:4]
        speed_err = torch.sum((auv_xy - tgt_xy) ** 2, dim=-1)
        sigma = max(float(self.cfg.rew_speed_sigma), 1.0e-6)
        r_speed = torch.exp(-speed_err / (sigma * sigma))
        rew = (
            self.cfg.rew_scale_uncertainty_mean * r_uncertainty
            + self.cfg.rew_scale_smooth * r_smooth
            + self.cfg.rew_scale_speed * r_speed
        )

        # Geometry shaping uses GT polar (same quantities the camera / KF gate on). Zero-cost
        # when all three scales are 0 (v0 path): skip the pose work.
        scale_range = float(getattr(self.cfg, "rew_scale_range", 0.0))
        scale_bearing = float(getattr(self.cfg, "rew_scale_bearing", 0.0))
        scale_lag = float(getattr(self.cfg, "rew_scale_lag", 0.0))
        if scale_range != 0.0 or scale_bearing != 0.0 or scale_lag != 0.0:
            agent_pos, agent_yaw = self._agent_pose_2d()
            _, gt_r, gt_a = self.kf.get_observation(self._target_pos, agent_pos, agent_yaw)
            if scale_range != 0.0:
                r_star = float(self.cfg.rew_range_star)
                sig_r = max(float(self.cfg.rew_range_sigma), 1.0e-6)
                r_range = torch.exp(-((gt_r - r_star) / sig_r) ** 2)
                rew = rew + scale_range * r_range
            if scale_bearing != 0.0:
                # Normalise by fov_half so the same sigma_frac works across 45°/90°/100° sensors.
                fov_half = max(float(self._fov_half), 1.0e-6)
                sig_b = max(float(self.cfg.rew_bearing_sigma_frac) * fov_half, 1.0e-6)
                r_bearing = torch.exp(-(gt_a.abs() / sig_b) ** 2)
                rew = rew + scale_bearing * r_bearing
            if scale_lag != 0.0:
                # Speeds are scalars (XY norm): lag = how much slower the agent is than the target.
                # Being faster yields lag=0 → r_lag=1 (no penalty). Symmetric rew_scale_speed is
                # left at 0 whenever this term is used.
                v_auv = torch.linalg.norm(auv_xy, dim=-1)
                v_tgt = torch.linalg.norm(tgt_xy, dim=-1)
                lag = torch.relu(v_tgt - v_auv)
                sig_l = max(float(self.cfg.rew_lag_sigma), 1.0e-6)
                r_lag = torch.exp(-(lag / sig_l) ** 2)
                rew = rew + scale_lag * r_lag

        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=-50.0)
        # Holo: if is_col: reward = min(0, reward) - c_penalty
        if self._is_col.any():
            penalized = torch.minimum(rew, torch.zeros_like(rew)) - self.cfg.rew_c_penalty
            rew = torch.where(self._is_col, penalized, rew)
        return rew

    def _push_reward_history(self, rew: torch.Tensor) -> torch.Tensor:
        """Append rewards; return done_by_reward mask (Holo 100-step mean < -3.5)."""
        idx = self._rew_hist_idx
        self._rew_hist[self._robot._ALL_INDICES, idx] = rew
        self._rew_hist_idx = (idx + 1) % self._rew_window
        self._rew_hist_count = torch.clamp(self._rew_hist_count + 1, max=self._rew_window)
        full = self._rew_hist_count >= self._rew_window
        mean_rew = self._rew_hist.mean(dim=-1)
        return full & (mean_rew < self.cfg.rew_done_threshold)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # DirectRLEnv order: dones -> rewards -> reset -> obs
        # Holo order: belief update -> reward(is_col) -> done_by_reward (NOT is_col) -> predict in obs
        self._update_belief()
        self._needs_belief_refresh[:] = False

        self._is_col[:] = self._compute_is_col()
        self._cached_rew[:] = self._compute_step_reward()
        done_by_reward = self._push_reward_history(self._cached_rew)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Match Holo: proximity/OOB only penalize; terminate via bad rolling reward (or timeout)
        return done_by_reward, time_out

    def _get_rewards(self) -> torch.Tensor:
        # Computed in _get_dones so rolling done sees the current step (Holo get_reward)
        return self._cached_rew

    def _get_observations(self) -> dict:
        # After reset, KF was re-initialized — sense once before building obs.
        if self._needs_belief_refresh.any():
            refresh_ids = self._needs_belief_refresh.nonzero(as_tuple=False).squeeze(-1)
            self._update_belief(refresh_ids)
            self._needs_belief_refresh[refresh_ids] = False

        # v0: predict then expose RL state (prior for next step)
        self.kf.predict()
        return {"policy": self._build_policy_obs()}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        half_x = float(self.cfg.max_auv_x) - float(self.cfg.margin2wall)
        half_y = float(self.cfg.max_auv_y) - float(self.cfg.margin2wall)

        self._default_root_state[env_ids] = self._robot.data.default_root_state[env_ids]
        # Inset AUV spawn by init dist_max so the polar target sample stays in-bound
        # (plain clamp can rotate/shorten a2t and push the ball out of the sonar lobe).
        inset = float(self.cfg.target_init_dist_max)
        half_x_agent = max(half_x - inset, 1.0)
        half_y_agent = max(half_y - inset, 1.0)
        self._default_root_state[env_ids, 0] = origins[:, 0] + sample_uniform(-half_x_agent, half_x_agent, n, self.device)
        self._default_root_state[env_ids, 1] = origins[:, 1] + sample_uniform(-half_y_agent, half_y_agent, n, self.device)
        self._default_root_state[env_ids, 2] = self.cfg.starting_depth + sample_uniform(-0.5, 0.5, n, self.device)
        # Holo: agent_init_yaw ~ U[-π/2, π/2]
        agent_yaw = sample_uniform(
            self.cfg.agent_init_yaw_min, self.cfg.agent_init_yaw_max, n, self.device
        )
        zeros = torch.zeros(n, device=self.device)
        self._default_root_state[env_ids, 3:7] = quat_from_euler_xyz(zeros, zeros, agent_yaw)
        self._default_root_state[env_ids, 7:] = 0.0
        self._robot.write_root_pose_to_sim(self._default_root_state[env_ids, :7], env_ids)
        self._robot.write_root_velocity_to_sim(self._default_root_state[env_ids, 7:], env_ids)

        agent_pos = self._default_root_state[env_ids, :2]
        rel_ang = sample_uniform(
            self.cfg.target_init_ang_min, self.cfg.target_init_ang_max, n, self.device
        )
        angle = agent_yaw + rel_ang
        dist = sample_uniform(self.cfg.target_init_dist_min, self.cfg.target_init_dist_max, n, self.device)
        tx = agent_pos[:, 0] + dist * torch.cos(angle)
        ty = agent_pos[:, 1] + dist * torch.sin(angle)
        tx = torch.clamp(tx, origins[:, 0] - half_x, origins[:, 0] + half_x)
        ty = torch.clamp(ty, origins[:, 1] - half_y, origins[:, 1] + half_y)
        # If clamp still broke the lobe, snap to mid-range, dead ahead.
        rel_x = tx - agent_pos[:, 0]
        rel_y = ty - agent_pos[:, 1]
        r_now = torch.sqrt(rel_x * rel_x + rel_y * rel_y)
        bear = torch.atan2(rel_y, rel_x) - agent_yaw
        bear = torch.atan2(torch.sin(bear), torch.cos(bear))
        mid_r = 0.5 * (float(self.cfg.target_init_dist_min) + float(self.cfg.target_init_dist_max))
        bad = (r_now < float(self.cfg.target_init_dist_min) - 1.0e-3) | (
            r_now > float(self.cfg.target_init_dist_max) + 1.0e-3
        ) | (torch.abs(bear) > float(self.cfg.target_init_ang_max) + 1.0e-3)
        if bool(bad.any()):
            tx = torch.where(bad, agent_pos[:, 0] + mid_r * torch.cos(agent_yaw), tx)
            ty = torch.where(bad, agent_pos[:, 1] + mid_r * torch.sin(agent_yaw), ty)
            tx = torch.clamp(tx, origins[:, 0] - half_x, origins[:, 0] + half_x)
            ty = torch.clamp(ty, origins[:, 1] - half_y, origins[:, 1] + half_y)
        # Holo: target starts at rest; RRT then faces first waypoint
        self._target_state[env_ids, 0] = tx
        self._target_state[env_ids, 1] = ty
        self._target_state[env_ids, 2] = 0.0
        self._target_state[env_ids, 3] = 0.0
        self._target_pos[env_ids, 0] = tx
        self._target_pos[env_ids, 1] = ty

        centers = origins[:, :2]
        self.target_motion.reset(
            env_ids,
            self._target_state[env_ids],
            centers,
            agent_xy=agent_pos,
            agent_yaw=agent_yaw,
        )
        self._sync_target_entity(env_ids)

        # Holo belief init: polar offset from true target using target yaw (path heading)
        self._set_kf_limits(env_ids)
        b_r = sample_uniform(self.cfg.belief_init_dist_min, self.cfg.belief_init_dist_max, n, self.device)
        b_ang = sample_uniform(self.cfg.belief_init_ang_min, self.cfg.belief_init_ang_max, n, self.device)
        target_yaw = self.target_motion.yaw[env_ids]
        init_state = torch.zeros(n, 4, device=self.device)
        init_state[:, 0] = tx + b_r * torch.cos(target_yaw + b_ang)
        init_state[:, 1] = ty + b_r * torch.sin(target_yaw + b_ang)
        init_state[:, 0] = torch.clamp(init_state[:, 0], origins[:, 0] - half_x, origins[:, 0] + half_x)
        init_state[:, 1] = torch.clamp(init_state[:, 1], origins[:, 1] - half_y, origins[:, 1] + half_y)
        self.kf.reset(env_ids, init_state, self.cfg.target_init_cov)
        self.pid.reset(env_ids, depth_target=float(self.cfg.starting_depth))
        if hasattr(self, "oceansim_sonar") and self.oceansim_sonar is not None:
            env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device)
            self.oceansim_sonar.reset_envs(env_ids_t)
            self.pipeline.invalidate_cache()
        self._sonar_locked[env_ids] = False
        self._observed[env_ids] = False
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._cmd_vel[env_ids] = 0.0
        self._prev_cmd_vel[env_ids] = 0.0
        self._pwm[env_ids] = 0.0
        self._ocean_current_vel[env_ids] = 0.0
        self._needs_belief_refresh[env_ids] = True
        self._is_col[env_ids] = False
        self._cached_rew[env_ids] = 0.0
        self._rew_hist[env_ids] = 0.0
        self._rew_hist_count[env_ids] = 0
        self._rew_hist_idx[env_ids] = 0
        if getattr(self.cfg, "thruster_first_order_lag", False):
            # prevTime must restart with the per-env episode clock, otherwise dt < 0
            # on the first step of the new episode (clamped in update(), but the
            # stale state would also carry over).
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[torch.as_tensor(env_ids, device=self.device).long()] = True
            self.thruster_dyn.reset(reset_mask)

    def _sample_from_sphere(self, n, r):
        phi = sample_uniform(0, 2 * math.pi, n, self.device)
        costheta = sample_uniform(-1, 1, n, self.device)
        theta = torch.arccos(costheta)
        return torch.stack(
            [
                r * torch.sin(theta) * torch.cos(phi),
                r * torch.sin(theta) * torch.sin(phi),
                r * torch.cos(theta),
            ],
            dim=-1,
        )

    def set_ocean_currents(
        self,
        env_ids: torch.Tensor | Sequence[int] | None,
        velocity: torch.Tensor | Sequence[float],
    ) -> None:
        """Holo-like API: set world-frame water velocity [m/s] for selected envs."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        v = torch.as_tensor(velocity, device=self.device, dtype=torch.float32)
        if v.ndim == 1:
            v = v.view(1, 3).expand(env_ids.numel(), 3).clone()
        if v.shape != (env_ids.numel(), 3):
            raise ValueError(f"velocity shape {tuple(v.shape)} != ({env_ids.numel()}, 3)")
        v = clip_current_velocity(v, float(self.cfg.ocean_current.clip_speed))
        self._ocean_current_vel[env_ids] = v

    def get_ocean_currents(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> torch.Tensor:
        """Return world-frame water velocity buffer [m/s] (clone)."""
        if env_ids is None:
            return self._ocean_current_vel.clone()
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        return self._ocean_current_vel[env_ids].clone()

    def close(self) -> None:
        sonar = getattr(self, "oceansim_sonar", None)
        if sonar is not None:
            sonar.close()
        super().close()

    def _compute_dynamics(self, actions) -> tuple[torch.Tensor, torch.Tensor]:
        n_thrusters = self.num_thrusters
        thruster_forces = torch.zeros((self.num_envs, n_thrusters, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, n_thrusters, 3), device=self.device, dtype=torch.float)
        motor = torch.clone(actions)
        motor = torch.nan_to_num(motor, nan=0.0, posinf=0.0, neginf=0.0)
        # Absolute sim time per env (DynamicsFirstOrder expects shape [num_envs])
        t = self.episode_length_buf.to(dtype=torch.float32) * float(self.sim.cfg.dt)

        if self.thruster_layout == "gsrov8":
            # Normalised command -> first-order lag -> thrust. A propeller makes T ~ rpm^2
            # and PWM maps roughly linearly onto rpm, so T = T_max * delta * |delta|, with
            # the forward/reverse datasheet rating selected by sign (2.0 / 1.8 kgf at 24 V)
            # and T_max already scaled by (V/V_rated)^2 in __init__.
            motor = self.thruster_dyn.update(motor, t)
            deadband = float(getattr(self.cfg, "thruster_deadband", 0.05))
            motor = torch.where(motor.abs() < deadband, torch.zeros_like(motor), motor)
            thrust = torch.where(
                motor >= 0.0,
                torch.full_like(motor, self._thrust_fwd_n),
                torch.full_like(motor, self._thrust_rev_n),
            )
            thrust = thrust * motor * motor.abs()
        else:
            motor[torch.abs(motor) < 0.08] = 0
            mp, mn = motor >= 0.08, motor <= -0.08
            motor[mp] = -139.0 * (motor[mp] ** 2) + 500 * motor[mp] + 8.28
            motor[mn] = 161.0 * (motor[mn] ** 2) + 517.86 * motor[mn] - 5.72
            motor = self.thruster_dyn.update(motor, t)
            thrust = self.thruster_conv.convert(motor)

        thruster_forces[..., 0] = 1.0
        thruster_forces = quat_apply(self.thruster_quats, thruster_forces)
        thruster_forces = thruster_forces * thrust.unsqueeze(-1)
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)
        thruster_forces = torch.sum(thruster_forces, dim=-2)
        thruster_torques = torch.sum(thruster_torques, dim=-2)
        bf, bt = self.force_calc.calculate_buoyancy_forces(
            self._robot.data.root_quat_w,
            self.cfg.water_rho,
            self.volumes,
            abs(self._gravity_mag),
            self.com_to_cob_offsets,
        )
        lin_vel = self._robot.data.root_lin_vel_w
        if self._ocean_current_field is not None or torch.any(self._ocean_current_vel != 0):
            lin_vel = lin_vel - self._ocean_current_vel
        df, dt_, vf, vt_ = self.force_calc.calculate_density_and_viscosity_forces(
            self._robot.data.root_quat_w,
            lin_vel,
            self._robot.data.root_ang_vel_w,
            self.inertia_tensors,
            self.inertia_tensors.mean(dim=1, keepdim=True),
            self.cfg.water_beta,
            self.cfg.water_rho,
            self.masses,
        )
        return df + bf + vf + thruster_forces, dt_ + bt + vt_ + thruster_torques
