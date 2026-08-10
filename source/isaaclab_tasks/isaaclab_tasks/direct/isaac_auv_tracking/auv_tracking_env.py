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
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .sonar_meas import body_polar_from_xy, detect_centroid_polar
from .thruster_dynamics import DynamicsFirstOrder, ConversionFunctionBasic, get_thruster_com_and_orientations

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
class AUVTrackingEnvCfg(DirectRLEnvCfg):
    # Control period ≈ 0.2 s like HoloOcean v0 (dt=1/120, decimation=24)
    decimation = 24
    # Holo make()/examples commonly use t_steps=200 @ 0.2 s → 40 s
    episode_length_s = 40.0
    # HoloOcean v0 PID: action_dim=2 in [-1,1], then scale [1.5, 0.5].
    # Must be bounded Box — int→Box(±inf) can make (high-low)*u NaN in some RL wrappers.
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    observation_space = 4
    state_space = 0
    action_scale_v = 1.5
    action_scale_w = 0.5

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
    # Episode target-motion schedule (from reset):
    #   0–3s  冷启动: cruise 0→target_speed + leash
    #   3–5s  easy:   full cruise + leash
    #   5s+   正常:   free RRT, no leash
    target_soft_start_s = 3.0
    target_easy_s = 5.0
    target_leash_range = 3.2  # < sensor_range=4; leave margin for FOV / lag
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
    target_init_dist_min = 1.0
    target_init_dist_max = 4.0
    # Holo ang_dist_range_a2t: target bearing relative to agent yaw
    target_init_ang_min = -math.pi / 5.0
    target_init_ang_max = math.pi / 5.0
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
    # Holo is_col: proximity / OOB → min(0, r) - c_penalty (does NOT terminate)
    rew_c_penalty = 5.0
    proximity_margin = 1.0
    # Holo: rolling mean reward over 100 steps < -3.5 → done
    rew_done_window = 100
    rew_done_threshold = -3.5

    # WarpAUV dynamics
    com_to_cob_offset = [0.0, 0.0, 0.01]
    water_rho = 997.0
    water_beta = 0.001306
    rotor_constant = 0.1 / 100.0
    dyn_time_constant = 0.05
    volume = 0.022747843530591776
    mass = 2.2701e01


class AUVTrackingEnv(DirectRLEnv):
    cfg: AUVTrackingEnvCfg

    def __init__(self, cfg: AUVTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        n_envs = cfg.scene.num_envs
        # Device may be finalized inside DirectRLEnv; use cfg hint then refresh after super.
        dev = getattr(cfg.sim, "device", "cuda:0")
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
        self._sonar_warmed = False
        self._sonar_locked = torch.zeros(n_envs, dtype=torch.bool, device=dev)

        super().__init__(cfg, render_mode, **kwargs)
        self._sonar_locked = self._sonar_locked.to(self.device)

        # Align buffers with env device after AppLauncher / DirectRLEnv set it.
        self.pipeline.device = self.device
        self.pipeline.sonar_image = self.pipeline.sonar_image.to(self.device)
        self.pipeline.polar_image = self.pipeline.polar_image.to(self.device)
        self._sonar_mount_offset_b = self._sonar_mount_offset_b.to(self.device)
        self.pipeline.warmup(self.oceansim_sonar, self.sim, n_frames=int(self.cfg.sonar_warmup_frames))
        self._sonar_warmed = True

        # RL action [a_v, a_ω]; low-level PWM from PID
        self._actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._cmd_vel = torch.zeros(self.num_envs, 2, device=self.device)
        self._pwm = torch.zeros(self.num_envs, 6, device=self.device)
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

        self.thruster_com_offsets, self.thruster_quats = get_thruster_com_and_orientations(self.device)
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = self.thruster_quats.repeat(self.num_envs, 1)

        self._gravity_mag = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.inertia_tensors[:, 0] = 0.37
        self.inertia_tensors[:, 1] = 0.97
        self.inertia_tensors[:, 2] = 1.19
        self.masses = torch.full((self.num_envs, 1), self.cfg.mass, device=self.device)
        self.com_to_cob_offsets = torch.tensor(self.cfg.com_to_cob_offset, device=self.device).repeat(self.num_envs, 1)
        self.volumes = torch.full((self.num_envs, 1), self.cfg.volume, device=self.device)

        self.force_calc = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_dyn = DynamicsFirstOrder(self.num_envs, 6, self.cfg.dyn_time_constant, self.device)
        self.thruster_conv = ConversionFunctionBasic(self.cfg.rotor_constant)

        self.pid = BatchPIDController(
            num_envs=self.num_envs,
            device=self.device,
            dt=float(self.cfg.sim.dt),
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

    def _update_belief(self, env_ids: torch.Tensor | None = None) -> None:
        """Sonar measure (r,α) → KF update. No predict (Holo order)."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

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
        # HoloOcean v0: cmd_vel = action * action_range_scale
        self._cmd_vel[:, 0] = self._actions[:, 0] * self.cfg.action_scale_v
        self._cmd_vel[:, 1] = self._actions[:, 1] * self.cfg.action_scale_w

    def _apply_action(self) -> None:
        # Target at physics rate (Holo: RRT+LQR every sim tick inside RL step)
        self._target_state[:] = self.target_motion.step(
            self._target_state,
            dt=float(self.cfg.sim.dt),
            agent_xy=self._robot.data.root_pos_w[:, :2],
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
        """Holo get_reward core on posterior (belief already updated)."""
        r_uncertainty = -self._log_det_cov()
        # std term unused when rew_scale_uncertainty_std=0 (Holo c_std=0)
        action_diff = torch.norm(self._actions - self._prev_actions, dim=-1)
        r_smooth = torch.exp(-action_diff)
        rew = (
            self.cfg.rew_scale_uncertainty_mean * r_uncertainty
            + self.cfg.rew_scale_smooth * r_smooth
        )
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
        # Holo: agent XY uniform over map (here: uniform in in_bound workspace)
        self._default_root_state[env_ids, 0] = origins[:, 0] + sample_uniform(-half_x, half_x, n, self.device)
        self._default_root_state[env_ids, 1] = origins[:, 1] + sample_uniform(-half_y, half_y, n, self.device)
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
        # Holo: target bearing relative to agent yaw within ang_dist_range_a2t
        rel_ang = sample_uniform(
            self.cfg.target_init_ang_min, self.cfg.target_init_ang_max, n, self.device
        )
        angle = agent_yaw + rel_ang
        dist = sample_uniform(self.cfg.target_init_dist_min, self.cfg.target_init_dist_max, n, self.device)
        tx = agent_pos[:, 0] + dist * torch.cos(angle)
        ty = agent_pos[:, 1] + dist * torch.sin(angle)
        # Keep target inside workspace
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
        self._pwm[env_ids] = 0.0
        self._needs_belief_refresh[env_ids] = True
        self._is_col[env_ids] = False
        self._cached_rew[env_ids] = 0.0
        self._rew_hist[env_ids] = 0.0
        self._rew_hist_count[env_ids] = 0
        self._rew_hist_idx[env_ids] = 0

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

    def close(self) -> None:
        if hasattr(self, "oceansim_sonar"):
            self.oceansim_sonar.close()
        super().close()

    def _compute_dynamics(self, actions) -> tuple[torch.Tensor, torch.Tensor]:
        thruster_forces = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        motor = torch.clone(actions)
        motor = torch.nan_to_num(motor, nan=0.0, posinf=0.0, neginf=0.0)
        motor[torch.abs(motor) < 0.08] = 0
        mp, mn = motor >= 0.08, motor <= -0.08
        motor[mp] = -139.0 * (motor[mp] ** 2) + 500 * motor[mp] + 8.28
        motor[mn] = 161.0 * (motor[mn] ** 2) + 517.86 * motor[mn] - 5.72
        # Absolute sim time per env (DynamicsFirstOrder expects shape [num_envs])
        t = self.episode_length_buf.to(dtype=torch.float32) * float(self.sim.cfg.dt)
        motor = self.thruster_dyn.update(motor, t)
        motor = self.thruster_conv.convert(motor)
        thruster_forces[..., 0] = 1.0
        thruster_forces = quat_apply(self.thruster_quats, thruster_forces)
        thruster_forces = thruster_forces * motor.unsqueeze(-1)
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
        df, dt_, vf, vt_ = self.force_calc.calculate_density_and_viscosity_forces(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_w,
            self._robot.data.root_ang_vel_w,
            self.inertia_tensors,
            self.inertia_tensors.mean(dim=1, keepdim=True),
            self.cfg.water_beta,
            self.cfg.water_rho,
            self.masses,
        )
        return df + bf + vf + thruster_forces, dt_ + bt + vt_ + thruster_torques
