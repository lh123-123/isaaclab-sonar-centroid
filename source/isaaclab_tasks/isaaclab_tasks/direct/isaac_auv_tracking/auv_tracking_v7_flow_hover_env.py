# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AUV station keeping: global optical-flow nulling (Direct-v7-flow-hover).

No target object: the vehicle holds position by driving the whole-image optical flow
to zero. Camera is upright (no 180 deg rotation). First pass uses the WarpAUV proxy;
real dynamics parameters (mass/inertia/thruster coords/thrust curves) are swapped in later.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab_tasks.direct.isaac_auv_tracking.agents.raft_flow import estimate_flow
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_env import AUVTrackingEnv, AUVTrackingEnvCfg
from isaaclab_tasks.direct.isaac_auv_tracking.assets.warpauv_cam import WARPAUV_CAM_CFG

_CAM_HFOV_DEG = 90.0
_CAM_APERTURE_MM = 20.955


@configclass
class AUVTrackingV7FlowHoverEnvCfg(AUVTrackingEnvCfg):
    """Station keeping via global optical-flow nulling (WarpAUV proxy)."""

    # No sonar / no target belief.
    kf_measurement: str = "gt"
    # Fixed AUV starting depth at z=5 (overrides base class default of 8).
    starting_depth: float = 5.0

    # Agent rides the camera-enabled asset copy.
    robot_cfg: RigidObjectCfg = WARPAUV_CAM_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Camera (upright, no rotation).
    cam_height: int = 64
    cam_width: int = 64
    render_height: int = 256
    render_width: int = 256
    cam_hfov_deg: float = _CAM_HFOV_DEG
    cam_aperture_mm: float = _CAM_APERTURE_MM
    frame_stack: int = 2
    cam_sync_usd_pose: bool = True

    # Flow.
    flow_source: str = "raft"
    flow_mag_min_px: float = 0.5
    flow_mag_rel_frac: float = 0.4
    min_texture_px: int = 20
    texture_grad_threshold: float = 8.0
    texture_lost_limit_steps: int = 15

    # Textured plane in front of camera (4x4 by default): replaces the 4 dark spheres.
    # Four corner points define the plane; plane_size controls its extent (meters).
    plane_size: float = 4.0
    plane_aspect: float = 1.0
    plane_distance: float = 2.0
    plane_texture_tile: float = 2.0
    plane_texture_path: str = ""
    plane_texture_paths: tuple = ()
    randomize_plane_texture: bool = False
    plane_occluders: bool = True

    # Reward weights (lenient depth so agent learns flow-nulling first).
    rew_flow: float = 1.0
    rew_alive: float = 0.3
    rew_depth: float = 0.1
    rew_att: float = 0.1
    rew_action: float = 0.01
    sigma_flow: float = 0.05  # normalized flow rms
    sigma_depth: float = 1.5  # meters (was 0.5 → lenient)
    sigma_att: float = 0.3  # radians

    # Depth / attitude targets (wider tolerance).
    depth_target: float = 5.0  # set in reset to starting_depth (AUV at z=5)
    depth_limit: float = 3.0  # meters from target -> terminate (was 2.0)
    att_limit: float = 0.8  # radians (was 0.6)

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/warpauv_base_link/LookFollowCamera0",
        data_types=["rgb", "distance_to_image_plane"],
        width=256,
        height=256,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        spawn=None,
    )

    # Observation space: RGB + flow stack + flow mag + texture mask.
    observation_space = {
        "image": gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(3 + frame_stack * 2 + 1 + 1, 64, 64),
            dtype=np.float32,
        ),
        "state": gym.spaces.Box(
            low=np.array([-10.0, -np.pi, -np.pi, -np.pi, -3.0, -3.0, -3.0, -3.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, np.pi, np.pi, np.pi, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        ),
    }


class AUVTrackingV7FlowHoverEnv(AUVTrackingEnv):
    """Station keeping via global optical-flow nulling."""

    cfg: AUVTrackingV7FlowHoverEnvCfg

    def _setup_scene(self):
        # Base builds robot + target + ground + light + sonar. We override to drop target/sonar.
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self._robot = RigidObject(self.cfg.robot_cfg)
        # Use textured seabed ground (sand texture from OceanSim assets).
        self._build_textured_ground()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["robot"] = self._robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Camera (adopted prim inside asset).
        self._apply_camera_intrinsics()
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera

        # 4x4 textured plane ahead of AUV camera (replaces 4 dark spheres).
        self._spawn_textured_plane()

        # Flow / texture buffers.
        k = int(self.cfg.frame_stack)
        h, w = int(self.cfg.cam_height), int(self.cfg.cam_width)
        self._flow_buf = torch.zeros((self.num_envs, k * 2, h, w), dtype=torch.float32, device=self.device)
        self._flow_image = torch.zeros((self.num_envs, 2, h, w), dtype=torch.float32, device=self.device)
        self._texture_mask = torch.zeros((self.num_envs, 1, h, w), dtype=torch.float32, device=self.device)
        self._flow_rms = torch.zeros(self.num_envs, device=self.device)
        self._texture_count = torch.zeros(self.num_envs, device=self.device)
        self._prev_rgb = torch.zeros((self.num_envs, self.cfg.render_height, self.cfg.render_width, 3), dtype=torch.uint8, device=self.device)
        self._prev_rgb_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stack_pending_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._texture_lost_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # USD pose sync for camera.
        self._robot_usd = None
        if bool(getattr(self.cfg, "cam_sync_usd_pose", True)):
            self._init_usd_pose_sync()

        # Ground truth not needed; keep _target_pos as agent pos for compatibility.
        self._target_pos = torch.zeros(self.num_envs, 2, device=self.device)

    def _build_textured_ground(self):
        """Build a simple textured sand ground plane using low-res texture."""
        try:
            from pxr import UsdGeom, UsdShade, Sdf, Gf
            # Try low-res texture first
            tex_paths = [
                "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/ground_lite/sand_01_8k/textures/sand_01_diff_256.jpg",
                "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/textures/ground_diff_4096.png",
            ]
            diff_tex = None
            for p in tex_paths:
                if os.path.isfile(p):
                    diff_tex = p
                    break
            stage = self.sim.stage
            size = 30.0
            tile = 10.0
            half = size * 0.5
            plane = UsdGeom.Mesh.Define(stage, "/World/ground")
            plane.GetPointsAttr().Set([Gf.Vec3f(-half, -half, 0.0), Gf.Vec3f(half, -half, 0.0),
                                       Gf.Vec3f(half, half, 0.0), Gf.Vec3f(-half, half, 0.0)])
            plane.GetFaceVertexCountsAttr().Set([4])
            plane.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
            stp = UsdGeom.PrimvarsAPI(plane).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            stp.Set([Gf.Vec2f(0, 0), Gf.Vec2f(tile, 0), Gf.Vec2f(tile, tile), Gf.Vec2f(0, tile)])
            xf = UsdGeom.Xformable(plane.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.03))
            mp = "/World/Ground/Material"
            mat = UsdShade.Material.Define(stage, mp)
            sh = UsdShade.Shader.Define(stage, mp + "/Preview")
            sh.CreateIdAttr("UsdPreviewSurface")
            if diff_tex:
                rd = UsdShade.Shader.Define(stage, mp + "/stReader")
                rd.CreateIdAttr("UsdPrimvarReader_float2")
                rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
                tx = UsdShade.Shader.Define(stage, mp + "/Tex")
                tx.CreateIdAttr("UsdUVTexture")
                tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(diff_tex))
                tx.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(rd.ConnectableAPI(), "result")
                tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
            else:
                sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.76, 0.70, 0.55))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(plane.GetPrim()).Bind(mat)
            plane.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        except Exception as e:
            print(f"[v7] WARN textured ground failed, fallback to plain: {e}", flush=True)
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    def _default_plane_texture_candidates(self):
        """Discover a preset group of swappable textures (256px diffuse maps)."""
        cands = []
        base = "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/ground_lite"
        if os.path.isdir(base):
            try:
                for d in sorted(os.listdir(base)):
                    texdir = os.path.join(base, d, "textures")
                    if not os.path.isdir(texdir):
                        continue
                    for f in sorted(os.listdir(texdir)):
                        if f.endswith("_diff_256.jpg"):
                            cands.append(os.path.join(texdir, f))
            except OSError:
                pass
        # seabed-level textures
        for p in [
            "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/textures/ground_diff_4096.png",
            "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/textures/ground_detiled_2048.png",
        ]:
            if os.path.isfile(p):
                cands.append(p)
        # rock / collected extras
        for p in [
            "/home/lh/IsaacLab/data_downloads/OceanSim_assets/collected_rock/textures/cropped_rock.jpg",
            "/home/lh/IsaacLab/data_downloads/OceanSim_assets/collected_MHL/textures/mhl_3hz_high_smooth_scaled.jpg",
        ]:
            if os.path.isfile(p):
                cands.append(p)
        return cands

    def _spawn_plane_occluders(self, env_prim, ox, oy, oz, depth, size, dist, half, tile, tex, half_w=None, back=0.75):
        """Spawn side/top/bottom occluder walls forming an open box around the front plane.

        The walls run from ``back`` behind the env origin to the front plane and share the
        plane's cross-section, so the camera sits inside the box. They stop neighboring
        environments leaking into view when the vehicle yaws or drifts past the plane edge.
        Vertices are wound with inward-facing normals and the material is double-sided, so
        the walls are visible from inside the box.
        """
        try:
            from pxr import UsdGeom, UsdShade, Sdf, Gf
            if half_w is None:
                half_w = half
            half_h = half
            z_lo = depth - half_h + oz
            z_hi = depth + half_h + oz
            y_lo = -half_w + oy
            y_hi = half_w + oy
            x_lo = -float(back) + ox
            x_hi = dist + ox
            dark_color = Gf.Vec3f(0.25, 0.22, 0.20)
            # (name, ordered corners, inward normal)
            walls = [
                # left wall at y=y_lo, inward normal +Y
                ("OccLeft", [
                    Gf.Vec3f(x_lo, y_lo, z_lo),
                    Gf.Vec3f(x_lo, y_lo, z_hi),
                    Gf.Vec3f(x_hi, y_lo, z_hi),
                    Gf.Vec3f(x_hi, y_lo, z_lo),
                ], Gf.Vec3f(0.0, 1.0, 0.0)),
                # right wall at y=y_hi, inward normal -Y
                ("OccRight", [
                    Gf.Vec3f(x_lo, y_hi, z_lo),
                    Gf.Vec3f(x_hi, y_hi, z_lo),
                    Gf.Vec3f(x_hi, y_hi, z_hi),
                    Gf.Vec3f(x_lo, y_hi, z_hi),
                ], Gf.Vec3f(0.0, -1.0, 0.0)),
                # top wall at z=z_hi, inward normal -Z
                ("OccTop", [
                    Gf.Vec3f(x_lo, y_lo, z_hi),
                    Gf.Vec3f(x_lo, y_hi, z_hi),
                    Gf.Vec3f(x_hi, y_hi, z_hi),
                    Gf.Vec3f(x_hi, y_lo, z_hi),
                ], Gf.Vec3f(0.0, 0.0, -1.0)),
                # bottom wall at z=z_lo, inward normal +Z
                ("OccBottom", [
                    Gf.Vec3f(x_lo, y_lo, z_lo),
                    Gf.Vec3f(x_hi, y_lo, z_lo),
                    Gf.Vec3f(x_hi, y_hi, z_lo),
                    Gf.Vec3f(x_lo, y_hi, z_lo),
                ], Gf.Vec3f(0.0, 0.0, 1.0)),
            ]
            for name, pts, normal in walls:
                path = f"{env_prim}/{name}"
                mesh = UsdGeom.Mesh.Define(self.sim.stage, path)
                mesh.GetPointsAttr().Set(pts)
                mesh.GetFaceVertexCountsAttr().Set([4])
                mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
                mesh.GetNormalsAttr().Set([normal] * 4)
                stp = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                stp.Set([Gf.Vec2f(0, 0), Gf.Vec2f(tile, 0), Gf.Vec2f(tile, tile), Gf.Vec2f(0, tile)])
                mat_path = f"{env_prim}/{name}Mat"
                mat = UsdShade.Material.Define(self.sim.stage, mat_path)
                sh = UsdShade.Shader.Define(self.sim.stage, mat_path + "/Preview")
                sh.CreateIdAttr("UsdPreviewSurface")
                if tex:
                    rd = UsdShade.Shader.Define(self.sim.stage, mat_path + "/stReader")
                    rd.CreateIdAttr("UsdPrimvarReader_float2")
                    rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
                    tx = UsdShade.Shader.Define(self.sim.stage, mat_path + "/Tex")
                    tx.CreateIdAttr("UsdUVTexture")
                    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex))
                    tx.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(rd.ConnectableAPI(), "result")
                    tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                    tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
                else:
                    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(dark_color)
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
                sh.CreateInput("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
                mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)
                mesh.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        except Exception as e:
            print(f"[v7] WARN occluder spawn failed: {e}", flush=True)

    def _resolve_plane_texture(self, rng_choice=None):
        """Resolve a texture path for the plane."""
        import random
        p = getattr(self.cfg, "plane_texture_path", "") or ""
        if p and os.path.isfile(p):
            base = p
        else:
            envp = os.environ.get("V7_PLANE_TEXTURE")
            if envp and os.path.isfile(envp):
                base = envp
            else:
                cands = [c for c in (getattr(self.cfg, "plane_texture_paths", ()) or ()) if isinstance(c, str) and os.path.isfile(c)]
                if cands:
                    if getattr(self.cfg, "randomize_plane_texture", False):
                        base = random.choice(cands)
                    else:
                        base = cands[0]
                else:
                    # Prefer a textured default (sand) for stable gradients.
                    preferred = "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/ground_lite/sand_01_8k/textures/sand_01_diff_256.jpg"
                    if os.path.isfile(preferred):
                        base = preferred
                    else:
                        dc = self._default_plane_texture_candidates()
                        base = dc[0] if dc else None
        if getattr(self.cfg, "randomize_plane_texture", False):
            cands = [c for c in (getattr(self.cfg, "plane_texture_paths", ()) or ()) if isinstance(c, str) and os.path.isfile(c)]
            if not cands:
                cands = self._default_plane_texture_candidates()
            if cands:
                base = random.choice(cands)
        return base

    def _spawn_textured_plane(self):
        """Spawn a 4x4 (configurable) textured plane per env, in front of the AUV camera.

        The plane is defined by 4 corner points (4 spheres' positions conceptually) and
        blocks the camera's view of neighboring environments while providing a textured
        surface whose material can be swapped to test texture effects.
        """
        try:
            from pxr import UsdGeom, UsdShade, Sdf, Gf
            import random
            depth = self.cfg.starting_depth
            size = float(self.cfg.plane_size)
            aspect = float(getattr(self.cfg, "plane_aspect", 1.0)) or 1.0
            dist = float(self.cfg.plane_distance)
            half_w = size * 0.5 * aspect  # horizontal (Y) half-extent
            half_h = size * 0.5           # vertical (Z) half-extent
            tile = float(self.cfg.plane_texture_tile)
            dark_color = Gf.Vec3f(0.70, 0.10, 0.08)
            for ei in range(self.num_envs):
                origin = self.scene.env_origins[ei]
                ox = float(origin[0]); oy = float(origin[1]); oz = float(origin[2])
                env_prim = self.scene.env_prim_paths[ei]
                path = f"{env_prim}/FlowPlane"
                mesh = UsdGeom.Mesh.Define(self.sim.stage, path)
                pts = [
                    Gf.Vec3f(dist + ox, -half_w + oy, depth - half_h + oz),
                    Gf.Vec3f(dist + ox,  half_w + oy, depth - half_h + oz),
                    Gf.Vec3f(dist + ox,  half_w + oy, depth + half_h + oz),
                    Gf.Vec3f(dist + ox, -half_w + oy, depth + half_h + oz),
                ]
                mesh.GetPointsAttr().Set(pts)
                mesh.GetFaceVertexCountsAttr().Set([4])
                mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
                stp = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                # UVs: V flipped so image "up" matches world +Z; 180° rotated.
                stp.Set([Gf.Vec2f(tile, 0), Gf.Vec2f(0, 0), Gf.Vec2f(0, tile), Gf.Vec2f(tile, tile)])
                mesh.GetPrim().CreateAttribute("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
                # compute normals
                try:
                    mesh.GetNormalsAttr().Set([Gf.Vec3f(-1.0, 0.0, 0.0)] * 4)
                except Exception:
                    pass

                tex = self._resolve_plane_texture()
                mat_path = f"{env_prim}/FlowPlaneMat"
                mat = UsdShade.Material.Define(self.sim.stage, mat_path)
                sh = UsdShade.Shader.Define(self.sim.stage, mat_path + "/Preview")
                sh.CreateIdAttr("UsdPreviewSurface")
                if tex:
                    rd = UsdShade.Shader.Define(self.sim.stage, mat_path + "/stReader")
                    rd.CreateIdAttr("UsdPrimvarReader_float2")
                    rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
                    tx = UsdShade.Shader.Define(self.sim.stage, mat_path + "/Tex")
                    tx.CreateIdAttr("UsdUVTexture")
                    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex))
                    tx.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(rd.ConnectableAPI(), "result")
                    tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                    tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
                else:
                    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(dark_color)
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
                mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)
                mesh.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)

                # Side/top/bottom occluders to reduce neighboring envs entering camera view.
                if getattr(self.cfg, "plane_occluders", True):
                    self._spawn_plane_occluders(env_prim, ox, oy, oz, depth, size, dist, half_h, tile, tex, half_w)
        except Exception as e:
            print(f"[v7] WARN plane spawn failed: {e}", flush=True)

    def _apply_camera_intrinsics(self):
        from pxr import UsdGeom
        hfov = float(self.cfg.cam_hfov_deg)
        aperture = float(self.cfg.cam_aperture_mm)
        focal = float(aperture * 0.5 / np.tan(np.radians(hfov) * 0.5))
        for i in range(self.num_envs):
            path = f"{self.scene.env_prim_paths[i]}/Robot/warpauv_base_link/LookFollowCamera0"
            prim = self.sim.stage.GetPrimAtPath(path)
            if prim is None or not prim.IsValid():
                continue
            UsdGeom.Camera(prim).GetFocalLengthAttr().Set(focal)

    def _init_usd_pose_sync(self):
        from isaacsim.core.prims import XFormPrim
        robot_paths = [f"{p}/Robot" for p in self.scene.env_prim_paths]
        try:
            self._robot_usd = XFormPrim(robot_paths, reset_xform_properties=False)
        except Exception as exc:
            self._robot_usd = None
            print(f"[v7-hover] WARN: USD pose sync disabled: {exc}", flush=True)

    def _sync_robot_usd_pose(self):
        if self._robot_usd is None:
            return
        self._robot_usd.set_world_poses(
            positions=self._robot.data.root_pos_w,
            orientations=self._robot.data.root_quat_w,
            usd=True,
        )

    def _camera_rgb(self):
        rgb = self._tiled_camera.data.output["rgb"]
        if rgb.dtype != torch.uint8:
            rgb = rgb.clamp(0, 255).to(torch.uint8)
        return rgb[..., :3]

    def _read_camera_frame(self):
        rgb = self._camera_rgb().permute(0, 3, 1, 2).to(torch.float32)
        if rgb.shape[-2:] != (self.cfg.cam_height, self.cfg.cam_width):
            rgb = F.interpolate(rgb, size=(self.cfg.cam_height, self.cfg.cam_width), mode="area")
        return rgb.round().clamp_(0.0, 255.0).to(torch.uint8)

    def _texture_from_rgb(self, rgb_nhwc):
        # gray gradient magnitude -> texture mask
        gray = rgb_nhwc.float().mean(dim=-1)  # (N,Rh,Rw)
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[:, 1:-1, 1:-1] = gray[:, 1:-1, 2:] - gray[:, 1:-1, :-2]
        gy[:, 1:-1, 1:-1] = gray[:, 2:, 1:-1] - gray[:, :-2, 1:-1]
        mag = torch.sqrt(gx * gx + gy * gy)
        rh, rw = gray.shape[-2], gray.shape[-1]
        h, w = self.cfg.cam_height, self.cfg.cam_width
        # threshold at render res, then MAX-pool down to cam res so thin texture survives.
        hot = (mag > self.cfg.texture_grad_threshold).float().unsqueeze(1)  # (N,1,Rh,Rw)
        if (rh, rw) != (h, w):
            mask = F.adaptive_max_pool2d(hot, (h, w))
        else:
            mask = hot
        count = mask.flatten(1).sum(dim=-1)
        return mask, count

    def _update_flow_and_texture(self):
        curr = self._camera_rgb()  # (N,Rh,Rw,3) uint8
        h, w = self.cfg.cam_height, self.cfg.cam_width
        rh, rw = self.cfg.render_height, self.cfg.render_width

        # texture
        tex_mask, tex_count = self._texture_from_rgb(curr)
        self._texture_mask = tex_mask
        self._texture_count = tex_count

        fresh = ~self._prev_rgb_ok
        self._flow_image.zero_()
        self._flow_rms[:] = 0.0
        if bool((~fresh).any()):
            est = estimate_flow(self._prev_rgb[~fresh], curr[~fresh])  # (m,2,Rh,Rw)
            small = F.interpolate(est, size=(h, w), mode="area") * (h / rh)
            du, dv = small[:, 0], small[:, 1]
            du_n = (du / max(w - 1, 1)).clamp(-1.0, 1.0)
            dv_n = (dv / max(h - 1, 1)).clamp(-1.0, 1.0)
            self._flow_image[~fresh, 0] = du_n
            self._flow_image[~fresh, 1] = dv_n
            # flow rms (pixels) over textured pixels
            mag_px = torch.sqrt(du * du + dv * dv)
            # expand texture mask to match
            tex = tex_mask[~fresh].squeeze(1)  # (m,H,W)
            tex_flat = tex.reshape(tex.shape[0], -1)
            mag_flat = mag_px.reshape(mag_px.shape[0], -1)
            # rms over textured pixels if enough, else all
            has_tex = tex_flat.sum(dim=-1) >= 4
            sq = mag_flat * mag_flat
            rms_tex = torch.sqrt((sq * tex_flat).sum(dim=-1) / tex_flat.sum(dim=-1).clamp(min=1))
            rms_all = torch.sqrt(sq.mean(dim=-1))
            rms = torch.where(has_tex, rms_tex, rms_all)
            self._flow_rms[~fresh] = rms
        self._prev_rgb[:] = curr
        self._prev_rgb_ok[:] = True
        self._flow_rms[fresh] = 0.0

    def _apply_action(self):
        # Ocean current (no target).
        dt = float(self.cfg.sim.dt)
        t = None
        if getattr(self, "_ocean_current_field", None) is not None:
            t = self.episode_length_buf.to(dtype=torch.float32) * dt
            v = self._ocean_current_field(self._robot.data.root_pos_w, t)
            self.set_ocean_currents(None, v)

        # PID at physics rate.
        self._pwm[:] = self.pid.compute(
            self._cmd_vel,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_w,
            self._robot.data.root_ang_vel_w,
            dt=dt,
        )
        thrust, moment = self._compute_dynamics(self._pwm)
        self._thrust[:, 0, :] = thrust
        self._moment[:, 0, :] = moment
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

        # USD sync on render iterations.
        interval = max(int(self.cfg.sim.render_interval), 1)
        on_render = self._sim_step_counter % interval == 0
        if self._robot_usd is not None and on_render:
            self._sync_robot_usd_pose()

    def _compute_is_col(self):
        # OOB only (no target proximity).
        out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.use_boundaries:
            half_x = float(self.cfg.max_auv_x) - float(self.cfg.margin2wall)
            half_y = float(self.cfg.max_auv_y) - float(self.cfg.margin2wall)
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > half_x)
                | (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > half_y)
                | (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        return out_of_bounds

    def _ego_vel_state(self):
        _, agent_yaw = self._agent_pose_2d()
        vel_w = self._robot.data.root_lin_vel_w[:, :2]
        cy, sy = torch.cos(agent_yaw), torch.sin(agent_yaw)
        v_surge = cy * vel_w[:, 0] + sy * vel_w[:, 1]
        v_sway = -sy * vel_w[:, 0] + cy * vel_w[:, 1]
        v_heave = self._robot.data.root_lin_vel_w[:, 2]
        yaw_rate = self._robot.data.root_ang_vel_w[:, 2]
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        depth = self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth
        return torch.stack([
            depth.clamp(-10.0, 10.0),
            roll.clamp(-np.pi, np.pi),
            pitch.clamp(-np.pi, np.pi),
            yaw.clamp(-np.pi, np.pi),
            v_surge.clamp(-3.0, 3.0),
            v_sway.clamp(-3.0, 3.0),
            v_heave.clamp(-3.0, 3.0),
            yaw_rate.clamp(-3.0, 3.0),
            self._flow_rms.clamp(0.0, 1.0),
            (self._texture_count / max(self.cfg.cam_height * self.cfg.cam_width, 1)).clamp(0.0, 1.0),
        ], dim=-1)

    def _get_observations(self):
        k = int(self.cfg.frame_stack)
        h, w = self.cfg.cam_height, self.cfg.cam_width

        rgb = self._read_camera_frame().to(torch.float32) / 255.0

        pending = self._stack_pending_reset.clone()
        if bool(pending.any()):
            self._flow_buf[pending] = self._flow_image[pending, :2].repeat(1, k, 1, 1)
            self._stack_pending_reset[pending] = False
        live = ~pending
        if bool(live.any()):
            self._flow_buf[live, :-2] = self._flow_buf[live, 2:].clone()
            self._flow_buf[live, -2:] = self._flow_image[live, :2]

        flow_mag = torch.sqrt(self._flow_image[:, 0] ** 2 + self._flow_image[:, 1] ** 2).unsqueeze(1)

        image = torch.cat([
            rgb,
            self._flow_buf,
            flow_mag,
            self._texture_mask,
        ], dim=1)

        state = self._ego_vel_state()

        return {"policy": {"image": image, "state": state}}

    def _compute_step_reward(self):
        present = (self._texture_count >= self.cfg.min_texture_px).to(torch.float32)
        flow_rms_n = (self._flow_rms / max(float(self.cfg.cam_width), 1.0)).clamp(0.0, 1.0)
        flow_term = present * torch.exp(-(flow_rms_n ** 2) / (self.cfg.sigma_flow ** 2))

        depth = self._robot.data.root_pos_w[:, 2]
        depth_err = depth - self.cfg.starting_depth
        depth_term = torch.exp(-(depth_err ** 2) / (self.cfg.sigma_depth ** 2))

        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        att_err = torch.sqrt(roll * roll + pitch * pitch)
        att_term = torch.exp(-(att_err ** 2) / (self.cfg.sigma_att ** 2))

        alive = present
        smooth = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)

        rew = (
            self.cfg.rew_flow * flow_term
            + self.cfg.rew_alive * alive
            + self.cfg.rew_depth * depth_term
            + self.cfg.rew_att * att_term
            - self.cfg.rew_action * smooth
        )
        rew = torch.nan_to_num(rew, nan=0.0, posinf=5.0, neginf=0.0)

        # texture-lost streak
        self._texture_lost_streak = torch.where(
            present > 0.5,
            torch.zeros_like(self._texture_lost_streak),
            self._texture_lost_streak + 1,
        )
        return rew

    def _get_dones(self):
        # Update flow/texture (after physics+render) so reward uses current frame.
        self._update_flow_and_texture()
        self._cached_rew[:] = self._compute_step_reward()

        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminated = terminated | (self._texture_lost_streak >= self.cfg.texture_lost_limit_steps)
        depth_err = (self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth).abs()
        terminated = terminated | (depth_err > self.cfg.depth_limit)
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        att_err = torch.sqrt(roll * roll + pitch * pitch)
        terminated = terminated | (att_err > self.cfg.att_limit)
        terminated = terminated | self._compute_is_col()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self):
        return self._cached_rew

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        try:
            DirectRLEnv._reset_idx(self, env_ids)
        except Exception:
            pass
        if not hasattr(self, "_default_root_state"):
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        # robot spawn at env origin (keep multi-env placement correct)
        origins = self.scene.env_origins[ids]
        self._default_root_state[ids, 0] = origins[:, 0]
        self._default_root_state[ids, 1] = origins[:, 1]
        self._default_root_state[ids, 2] = self.cfg.starting_depth
        # level attitude, zero yaw (skeleton); randomize later
        zeros_q = torch.zeros(ids.numel(), device=self.device)
        self._default_root_state[ids, 3] = 1.0  # w
        self._default_root_state[ids, 4:7] = 0.0
        self._robot.write_root_pose_to_sim(self._default_root_state[ids, :7], ids)
        self._robot.write_root_velocity_to_sim(self._default_root_state[ids, 7:], ids)

        self._flow_buf[ids] = 0.0
        self._flow_image[ids] = 0.0
        self._texture_mask[ids] = 0.0
        self._flow_rms[ids] = 0.0
        self._texture_count[ids] = 0.0
        self._prev_rgb_ok[ids] = False
        self._stack_pending_reset[ids] = True
        self._texture_lost_streak[ids] = 0
        self._actions[ids] = 0.0
        self._prev_actions[ids] = 0.0
        self._cmd_vel[ids] = 0.0
        self._prev_cmd_vel[ids] = 0.0
        self._pwm[ids] = 0.0
        if hasattr(self, "_cached_rew"):
            self._cached_rew[ids] = 0.0
        if hasattr(self, "_ocean_current_vel"):
            self._ocean_current_vel[ids] = 0.0
        if hasattr(self, "pid"):
            self.pid.reset(ids, depth_target=float(self.cfg.starting_depth))
        if hasattr(self, "thruster_dyn") and getattr(self.cfg, "thruster_first_order_lag", False):
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[ids] = True
            self.thruster_dyn.reset(reset_mask)
        if getattr(self, "_robot_usd", None) is not None:
            self._sync_robot_usd_pose()

    def _pre_physics_step(self, actions):
        self._prev_actions[:] = self._actions
        self._actions[:] = torch.clip(actions, -1.0, 1.0)
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

    def close(self):
        sonar = getattr(self, "oceansim_sonar", None)
        if sonar is not None:
            sonar.close()
        super().close()
