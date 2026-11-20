# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based configuration for the v7 flow-hover AUV task (GS-ROV, optical-flow nulling).

Behaviour-preserving port of the Direct ``AUVTrackingV7FlowHoverGsrovEnv`` to the manager-based
workflow. Observation/reward/termination/action logic is split into ``mdp`` terms; the shared
GS-ROV physics lives in :class:`..dynamics.GSROVPlant` (used by the ``VelocityPIDAction`` term).
Custom USD scene assets (textured plane, reference spheres, forward/view cameras) are still
spawned imperatively by the env (see ``v7_flow_hover_env.py``) per the refactor scope.
"""

from __future__ import annotations

import math

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_env import OceanCurrentCfg

from . import mdp

_CAM_HFOV_DEG = 81.65
_CAM_APERTURE_MM = 20.955


##
# Scene definition
##


@configclass
class V7FlowHoverSceneCfg(InteractiveSceneCfg):
    """Robot (camera-enabled cuboid proxy) + light. Cameras/plane/spheres are spawned by the env."""

    # GS-ROV lightweight cuboid proxy. Mass matches cfg.mass so buoyancy/weight are consistent.
    # Initial pos/rot (trim attitude) are set in the env cfg __post_init__.
    robot: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.56, 0.43, 0.30),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=13.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """No command generator for this task (station keeping)."""


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    velocity: mdp.actions.VelocityPIDActionCfg = mdp.actions.VelocityPIDActionCfg(
        asset_name="robot", scale=0.5, cmd_rate_limit=0.0
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Asymmetric policy observations: image (CNN actor) + state (MLP critic)."""

        image = ObsTerm(func=mdp.observations.image, clip=(-1.0, 1.0))
        state = ObsTerm(func=mdp.observations.ego_state)

        def __post_init__(self):
            # Keep image and state as separate Dict keys for the asymmetric SAC policy.
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms (un-weighted values; ``weight`` reproduces the Direct reward weights)."""

    flow = RewTerm(func=mdp.rewards.flow_nulling, weight=1.0, params={"sigma_flow": 0.05})
    alive = RewTerm(func=mdp.rewards.alive, weight=0.3)
    depth = RewTerm(func=mdp.rewards.depth_hold, weight=0.1, params={"sigma_depth": 1.5})
    attitude = RewTerm(func=mdp.rewards.attitude_hold, weight=0.1, params={"sigma_att": 0.3})
    action_rate = RewTerm(func=mdp.rewards.action_smoothness, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=env_mdp.time_out, time_out=True)
    texture_lost = DoneTerm(func=mdp.terminations.texture_lost)
    depth_exceeded = DoneTerm(func=mdp.terminations.depth_exceeded)
    attitude_exceeded = DoneTerm(func=mdp.terminations.attitude_exceeded)
    out_of_bounds = DoneTerm(func=mdp.terminations.out_of_bounds)


@configclass
class EventCfg:
    """Reset events (robot pose handled by scene default; buffers cleared here)."""

    reset_buffers = EventTerm(func=mdp.terminations.reset_auv_buffers, mode="reset")


##
# Environment configuration
##


@configclass
class V7FlowHoverEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based config for v7 station keeping via global optical-flow nulling (GS-ROV)."""

    # Scene settings
    scene: V7FlowHoverSceneCfg = V7FlowHoverSceneCfg(num_envs=64, env_spacing=45.0)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # ------------------------------------------------------------------ task fields
    starting_depth: float = 0.0
    use_real_usd: bool = False

    # Camera (upright, no rotation). 1280x720 render -> 64x64 policy input.
    cam_height: int = 64
    cam_width: int = 64
    render_height: int = 720
    render_width: int = 1280
    cam_hfov_deg: float = _CAM_HFOV_DEG
    cam_aperture_mm: float = _CAM_APERTURE_MM
    frame_stack: int = 1
    cam_sync_usd_pose: bool = True

    # Plain (non-sensor) viewport camera (recording / eyeballing only).
    spawn_view_camera: bool = True
    view_camera_name: str = "ViewCamera0"
    view_camera_offset: tuple = (0.28, 0.0, 0.0)
    view_camera_roll_deg: float = -90.0

    # Flow (Farneback @ 1280x720 -> 64x64, 20 Hz policy chain).
    flow_source: str = "farneback"
    min_texture_px: int = 20
    texture_grad_threshold: float = 8.0
    texture_lost_limit_steps: int = 50

    # Textured plane (pool cross-section) + reference spheres.
    plane_size: float = 4.0
    plane_aspect: float = 1.0
    plane_distance: float = 2.0
    plane_texture_tile: float = 0.8
    plane_texture_path: str = ""
    plane_texture_paths: tuple = ()
    randomize_plane_texture: bool = False
    plane_occluders: bool = True

    ref_sphere_radius: float = 0.1
    ref_sphere_color: tuple = (0.0, 1.0, 0.0)
    ref_sphere_offset_x: float = 1.5
    ref_sphere_half_side: float = 0.25

    # GS-ROV physics (from real-vehicle measurements).
    thruster_layout: str = "gsrov8"
    thruster_max_thrust_fwd_n: float = 19.613
    thruster_max_thrust_rev_n: float = 17.652
    thruster_neutral_us: float = 1500.0
    thruster_span_us: float = 500.0
    thruster_rated_voltage_v: float = 24.0
    thruster_battery_voltage_v: float = 24.0
    thruster_deadband: float = 0.05

    mass: float = 13.15
    volume: float = 0.01295
    com_to_cob_offset: list = (0.040, 0.0, 0.11)
    inertia_diag: tuple = (0.297, 0.440, 0.545)
    trim_roll_rad: float = 0.0
    trim_pitch_rad: float = float(-np.arctan2(0.040, 0.11))

    # Fluid / thruster-dynamics (match AUVTrackingEnvCfg defaults).
    water_rho: float = 997.0
    water_beta: float = 0.001306
    rotor_constant: float = 0.1 / 100.0
    dyn_time_constant: float = 0.05
    thruster_first_order_lag: bool = False
    ocean_current: OceanCurrentCfg = OceanCurrentCfg()

    # Reward weights / kernels.
    rew_flow: float = 1.0
    rew_alive: float = 0.3
    rew_depth: float = 0.1
    rew_att: float = 0.1
    rew_action: float = 0.01
    sigma_flow: float = 0.05
    sigma_depth: float = 1.5
    sigma_att: float = 0.3

    # Depth / attitude tolerances.
    depth_target: float = 0.0
    depth_limit: float = 1.5
    att_limit: float = 0.3

    # Workspace bounds.
    use_boundaries: bool = True
    max_auv_x: float = 20.0
    max_auv_y: float = 20.0
    max_auv_z: float = 7.0
    margin2wall: float = 1.0

    # Observation / action spaces are derived by the managers from the terms above.

    # Camera sensor (created by the env after the robot camera prim is spawned).
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ForwardCamera0",
        data_types=["rgb", "distance_to_image_plane"],
        width=1280,
        height=720,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        spawn=None,
    )

    def __post_init__(self):
        """Post initialization."""
        # Policy frequency: 20 Hz. decimation=6 at sim.dt=1/120 -> 1/(1/120*6) = 20 Hz.
        self.decimation = 6
        self.episode_length_s = 40.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 6
        self.sim.device = "cuda:0"
        # Re-render after resets so the flow camera reflects the reset pose (no stale frame).
        self.rerender_on_reset = True

        # Place the robot at the pool's vertical center with the free-floating trim attitude.
        self.scene.robot.init_state.pos = (0.0, 0.0, self.starting_depth)
        hp = self.trim_pitch_rad * 0.5
        # Quaternion (w, x, y, z) for a pure pitch (rotation about +Y).
        self.scene.robot.init_state.rot = (math.cos(hp), 0.0, math.sin(hp), 0.0)

        # RewardManager multiplies every term by ``dt`` before summing. The Direct env summed
        # ``term * weight`` directly (no dt), so divide the configured weights by the policy
        # step-dt to reproduce the exact same per-step reward magnitude.
        inv_dt = 1.0 / (self.decimation * self.sim.dt)
        for term_name in ("flow", "alive", "depth", "attitude", "action_rate"):
            term = getattr(self.rewards, term_name)
            term.weight *= inv_dt
