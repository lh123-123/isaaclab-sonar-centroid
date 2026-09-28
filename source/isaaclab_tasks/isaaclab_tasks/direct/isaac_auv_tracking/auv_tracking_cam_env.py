# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AUV tracking Direct-v3-cam: forward-looking camera + grayscale frame stacking.

Same dynamics / KF / reward as v0, with three changes:
  - no sonar: the KF measurement comes from the geometric GT polar (`kf_measurement="gt"`)
  - the target is a second WarpAUV instead of a sphere (Holo v0 parity)
  - the policy observation is a stack of K luminance frames from a camera that sits at the
    original sonar mount pose (64x64, HFOV 45 deg, K=4 -> 4 channels)

The camera renders RGB, but under this scene's single uniform DomeLight the channels are
~3x redundant (measured: 89.1% of pixels have R == G == B exactly, corr(R, G) = 0.9999864,
chroma residual = 0.0022% of total variance), so only Rec.601 luma is kept. That cuts the
per-transition replay cost by 3x, which is what caps the buffer.

Thruster first-order lag is enabled (`dyn_time_constant=0.2`), so the thrusters no
longer respond instantaneously to the PWM command.
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from .assets.warpauv import WARPAUV_CFG
from .assets.warpauv_cam import WARPAUV_CAM_CFG
from .auv_tracking_env import AUVTrackingEnv, AUVTrackingEnvCfg
from isaaclab_tasks.direct.sonar_centroid_track.auv_platform import AUV_SONAR_MOUNT_OFFSET_LOCAL

# The camera reuses the sonar mount pose so v0/v2 comparisons stay apples-to-apples.
CAM_MOUNT_OFFSET_LOCAL: tuple[float, float, float] = AUV_SONAR_MOUNT_OFFSET_LOCAL

# 20.955 mm aperture at 45 deg HFOV -> f = (h/2) / tan(HFOV/2) = 25.29 mm
_CAM_HFOV_DEG = 45.0
_CAM_APERTURE_MM = 20.955
# The camera is no longer spawned (tiled_camera.spawn is None), so these intrinsics now live in
# the asset itself: warpauv_base_link/LookFollowCamera0 authors focalLength=25.2949 mm and
# horizontalAperture=20.955 mm, i.e. exactly the 45 deg HFOV that sensor_fov_deg assumes. These
# two constants are kept as the reference value so the asset and the KF gate cannot silently
# drift apart.
_CAM_FOCAL_MM = float(_CAM_APERTURE_MM * 0.5 / np.tan(np.radians(_CAM_HFOV_DEG) * 0.5))


TARGET_AUV_CFG: RigidObjectCfg = WARPAUV_CFG.replace(
    prim_path="/World/envs/env_.*/Target",
    spawn=WARPAUV_CFG.spawn.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # The pose is overwritten every step by _sync_target_entity, so gravity would
            # only add drift between the write and the physics step.
            disable_gravity=True,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        # NOTE: the target keeps the asset's collisions -- deliberately, matching Holo v0, where
        # the target AUV is a normal collidable body. Do not re-add
        # `collision_props=CollisionPropertiesCfg(collision_enabled=False)`: it would be a silent
        # no-op. The AUV's collision geometry lives in `/warpauv/warpauv_base_link/collisions`,
        # which is an *instance* prim (prototype `/__Prototype_2/mesh_0`, collisionEnabled=True),
        # and `isaaclab.sim.utils.apply_nested` skips instance prims outright
        # (`if child_prim.IsInstance(): continue`), so the attribute is never written. Contact
        # stays rare anyway: `proximity_margin=1.0` already penalises the agent well before the
        # ~0.4-0.7 m at which the two hulls could touch, and if contact did happen
        # _sync_target_entity rewrites the target's pose on the very next step, so the target
        # cannot be pushed off its LQR trajectory for more than a single step.
    ),
)
"""Second WarpAUV used as the tracking target (Holo v0 parity: same vehicle as the agent)."""


@configclass
class AUVTrackingCamEnvCfg(AUVTrackingEnvCfg):
    """v0 dynamics/reward, camera-only observation, AUV target."""

    # --- no sonar: KF measurement is the geometric GT polar ---
    kf_measurement: str = "gt"
    sensor_range: float = 4.0
    # Must match the camera HFOV below. If it is wider, the GT `observed` gate reports a
    # valid measurement for targets the camera cannot see, so the uncertainty reward
    # would contradict the observation.
    sensor_fov_deg: float = _CAM_HFOV_DEG

    # The agent rides the editable asset *copy* (data/warpauv_cam), which carries the
    # LookFollowCamera0 prim. The target stays on the pristine asset: it needs no camera, and
    # sharing the copy would silently add a camera prim to every target as well.
    robot_cfg: RigidObjectCfg = WARPAUV_CAM_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- target: a second WarpAUV, Holo v0 a2t sampling range ---
    target_cfg: RigidObjectCfg = TARGET_AUV_CFG
    target_init_dist_min: float = 1.0
    target_init_dist_max: float = 4.0
    # Optional debug aid. The target is deliberately the *same* vehicle as the agent, and with a
    # single uniform DomeLight the two are nearly indistinguishable in RGB: measured target-vs-
    # background luminance contrast is ~12/255 locally and ~4/255 on average, over a ~14x12 px
    # silhouette. Setting this to a colour (e.g. the sonar beacon orange (0.85, 0.38, 0.12)) tints
    # only the Target prim so the figure/ground signal is unambiguous. Kept None by default so the
    # task stays faithful to "target is the same AUV"; the camera sits 0.35 m ahead of the vehicle
    # and cannot see it, so tinting the target introduces no self/other ambiguity.
    target_tint: tuple[float, float, float] | None = None

    # --- thrusters: first-order lag is now a real knob ---
    dyn_time_constant: float = 0.2
    thruster_first_order_lag: bool = True

    # --- reward: v2 sweep winner, suppresses orbiting ---
    rew_scale_speed: float = 0.2

    # --- camera ---
    cam_height: int = 64
    cam_width: int = 64
    cam_hfov_deg: float = _CAM_HFOV_DEG
    frame_stack: int = 4
    # Mirror the robot's simulated pose into the robot prim's USD xform every render step.
    # Required: with replicate_physics=True PhysX writes poses to Fabric and never updates USD,
    # so the camera (a child prim, and not a rigid body) would otherwise inherit a frozen
    # transform and the rendered view would never follow the vehicle. See _sync_robot_usd_pose.
    cam_sync_usd_pose: bool = True
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        # Wraps the LookFollowCamera0 prim that ships inside the asset copy. spawn=None is the
        # supported "adopt an existing prim" path (interactive_scene.py only spawns when
        # `asset_cfg.spawn is not None`).
        prim_path="/World/envs/env_.*/Robot/warpauv_base_link/LookFollowCamera0",
        data_types=["rgb"],
        width=64,
        height=64,
        # NOTE: left at the default (False) to avoid a redundant per-step XFormPrim read: nothing in
        # this env consumes Camera.data.pos_w. It would now be correct if enabled, because
        # _sync_robot_usd_pose keeps the USD xform fresh (before that fix it reported the spawn
        # pose forever, even with this flag True). Code that needs the camera pose should instead
        # derive it from the robot's simulated pose plus this mount offset.
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAM_MOUNT_OFFSET_LOCAL,
            rot=(1.0, 0.0, 0.0, 0.0),
            # "world": forward +X, up +Z in the Robot frame == the body convention the
            # WarpAUV USD and the sonar mount already use.
            convention="world",
        ),
        # No spawner: the mount pose (translate 0.35/0/0.05 + the "world" orientation, both baked
        # into the asset) and the intrinsics (focal 25.2949 mm on a 20.955 mm aperture = 45 deg
        # HFOV, clipping 0.05-30 m) come from the asset. This also means `offset` above is *inert*:
        # camera.py only applies it inside the spawn branch.
        spawn=None,
    )

    # uint8 keeps the replay buffer at 1 byte/pixel: 2 * frame_stack * 64 * 64 = 32 KB per
    # transition. Grayscale (see cam_encoder) rather than RGB: measured 89.1% of pixels have
    # R == G == B exactly, so the two dropped channels were pure redundancy. float32 would be
    # 4x this, and RGB would be 3x.
    observation_space = gym.spaces.Box(
        low=0, high=255, shape=(frame_stack, cam_height, cam_width), dtype=np.uint8
    )


class AUVTrackingCamEnv(AUVTrackingEnv):
    """v0 env with the sonar replaced by a forward camera and K-frame luma stacking."""

    cfg: AUVTrackingCamEnvCfg

    def _setup_scene(self):
        # Base builds robot + target, clones the envs and (because kf_measurement="gt")
        # skips the sonar. The camera must come after clone_environments: SensorBase
        # counts envs by resolving the ".../env_.*/Robot" parent path.
        self._apply_target_tint()
        super()._setup_scene()

        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Allocate the frame stack here: _setup_scene runs before the first _reset_idx,
        # and self.device / self.num_envs are already final.
        k = int(self.cfg.frame_stack)
        self._frame_buf = torch.zeros(
            (self.num_envs, k, self.cfg.cam_height, self.cfg.cam_width),
            dtype=torch.uint8,
            device=self.device,
        )
        # Set on reset, consumed by _get_observations once a *new* frame has been rendered.
        self._stack_pending_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Writable USD handle over the robot roots; see _sync_robot_usd_pose.
        self._robot_usd = None
        if bool(getattr(self.cfg, "cam_sync_usd_pose", True)):
            self._init_usd_pose_sync()

    def _apply_target_tint(self) -> None:
        """Bind a distinct visual material on the Target prim only (see cfg.target_tint).

        Applied to the *cfg* before the target is spawned, because the material is bound during
        spawning. Nothing happens when target_tint is None, which keeps the default behaviour
        (target identical to the agent).
        """
        tint = getattr(self.cfg, "target_tint", None)
        if tint is None:
            return
        spawn = self.cfg.target_cfg.spawn
        if spawn is None or not hasattr(spawn, "visual_material"):
            print("[cam] WARN: target_tint set but target spawn does not support visual_material", flush=True)
            return
        # PreviewSurfaceCfg is a VisualMaterialCfg, i.e. exactly the type the field expects.
        spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(c) for c in tint))

    def _read_camera_frame(self) -> torch.Tensor:
        """Camera RGB ``(N, H, W, 3)`` uint8 -> Rec.601 luma ``(N, 1, H, W)`` uint8.

        The camera renders colour but this scene has no usable chroma (see the module
        docstring), so the frames are collapsed to luminance. The channel axis is kept as
        a singleton so ``expand``/assignment against the ``(N, K, H, W)`` stack stay 4-D.
        """
        rgb = self._tiled_camera.data.output["rgb"]
        if rgb.dtype != torch.uint8:
            # TiledCamera "rgb" is uint8; anything else means the sensor config changed.
            rgb = rgb.to(torch.uint8)
        rgb = rgb[..., :3].to(torch.float32)
        # Round rather than truncate: a plain .to(uint8) would bias every pixel down by up to 1.
        luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return luma.round().clamp_(0.0, 255.0).to(torch.uint8).unsqueeze(1)

    def _init_usd_pose_sync(self) -> None:
        """Open a writable USD view over the robot roots (one prim per env)."""
        from isaacsim.core.prims import XFormPrim

        robot_paths = [f"{p}/Robot" for p in self.scene.env_prim_paths]
        try:
            # reset_xform_properties=False: keep the authored xform ops, we only write values.
            self._robot_usd = XFormPrim(robot_paths, reset_xform_properties=False)
        except Exception as exc:  # noqa: BLE001 - never let a USD nicety break the env
            self._robot_usd = None
            print(f"[cam] WARN: USD pose sync disabled, could not open {robot_paths[:1]}: {exc}", flush=True)

    def _sync_robot_usd_pose(self) -> None:
        """Mirror the robot's simulated world pose into its USD xform.

        Why this is needed: with ``replicate_physics=True`` PhysX keeps rigid-body poses in
        Fabric and never writes them back to USD, so every USD xform stays at its authored
        value forever (measured: teleporting the robot 8.9 m produced a 0.000 m change in its
        USD world transform). The camera is a *child* prim and is not a rigid body, so the
        renderer resolves it from USD only -- meaning it inherited a frozen parent transform and
        the rendered view never moved at all (the floor band stayed at image row 53 while the
        robot was dropped 6 m). Writing the robot's pose into USD makes the child camera (and any
        future child sensor) track the vehicle through normal USD composition, so no
        camera-convention math is duplicated here.

        `XFormPrim.set_world_poses(usd=True)` converts our world pose into the local transform of
        each prim's (frozen) parent, which is exactly the correction needed.
        """
        # NOTE: this view uses the torch backend, so pass tensors -- numpy arrays raise
        # "AttributeError: 'numpy.ndarray' object has no attribute 'detach'".
        self._robot_usd.set_world_poses(
            positions=self._robot.data.root_pos_w,
            orientations=self._robot.data.root_quat_w,
            usd=True,
        )

    def _apply_action(self) -> None:
        super()._apply_action()
        # `step()` renders inside the decimation loop as
        #   if _sim_step_counter % sim.render_interval == 0: sim.render()
        # so writing here, on exactly those iterations, lands the fresh pose in USD before the
        # renderer reads it. _sim_step_counter is already incremented for this iteration, so the
        # modulo matches the engine's condition. Doing it every substep would mean ~24x the USD
        # writes for no benefit.
        if self._robot_usd is None:
            return
        interval = max(int(self.cfg.sim.render_interval), 1)
        if self._sim_step_counter % interval == 0:
            self._sync_robot_usd_pose()

    def _get_observations(self) -> dict:
        # Keep the base KF bookkeeping (predict + post-reset belief sense); only the
        # "policy" entry is replaced by the frame stack.
        obs = super()._get_observations()
        k = int(self.cfg.frame_stack)
        frame = self._read_camera_frame()  # (N, 1, H, W)

        pending = self._stack_pending_reset
        if bool(pending.any()):
            # A reset episode must start with K copies of its *own* first frame, so the
            # previous episode's tail never leaks into the stack.
            self._frame_buf[pending] = frame[pending].expand(-1, k, -1, -1)
            self._stack_pending_reset[pending] = False

        live = ~pending
        if bool(live.any()):
            self._frame_buf[live, :-1] = self._frame_buf[live, 1:]
            self._frame_buf[live, -1] = frame[live, 0]

        # Already (N, K, H, W) uint8, i.e. exactly the declared observation_space.
        obs["policy"] = self._frame_buf
        return obs

    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        super()._reset_idx(env_ids)
        # Base __init__ may call _reset_idx through DirectRLEnv before _setup_scene has
        # run, so the buffer is not guaranteed to exist yet.
        if not hasattr(self, "_stack_pending_reset"):
            return
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        self._stack_pending_reset[ids] = True

    def _sync_target_entity(self, env_ids: torch.Tensor | None = None) -> None:
        """Same as base, but yaw the target AUV along its heading.

        Holo v0 teleports the target with a rotation derived from its velocity; without
        this the WarpAUV stays at identity quaternion and appears to drift sideways.
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        n = len(env_ids)
        pose = torch.zeros(n, 7, device=self.device)
        pose[:, 0] = self._target_pos[env_ids, 0]
        pose[:, 1] = self._target_pos[env_ids, 1]
        pose[:, 2] = self.cfg.starting_depth
        yaw = self.target_motion.yaw[env_ids]
        zeros = torch.zeros_like(yaw)
        pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        vel = torch.zeros(n, 6, device=self.device)
        vel[:, 0] = self._target_state[env_ids, 2]
        vel[:, 1] = self._target_state[env_ids, 3]
        self._target.write_root_pose_to_sim(pose, env_ids)
        self._target.write_root_velocity_to_sim(vel, env_ids)
