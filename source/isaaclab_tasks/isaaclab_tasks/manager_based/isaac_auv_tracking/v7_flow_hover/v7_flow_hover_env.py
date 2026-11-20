# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based v7 flow-hover env: custom scene assets + stateful optical-flow buffers.

This subclasses :class:`ManagerBasedRLEnv` only to (1) spawn the imperative USD assets the
scene cfg cannot express declaratively (textured plane, occluders, reference spheres, the
per-env forward camera and the viewport camera) and (2) own the stateful Farneback-flow /
texture-mask buffers consumed by the ``mdp`` observation/reward/termination terms.

The camera is created *after* the managers in ``__init__`` so it can be spawned under the
already-cloned ``Robot`` prims. The observation ``image`` term returns a correctly-shaped
placeholder during the manager's dimension-probing call (camera not yet built), then real
frames afterwards.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import TiledCamera

from .v7_flow_hover_env_cfg import V7FlowHoverEnvCfg


def _farneback_flow(prev_rgb: torch.Tensor, curr_rgb: torch.Tensor) -> torch.Tensor:
    """Farneback optical flow on CPU, returning (N, 2, H, W) in pixels."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "cv2 (opencv-python) is required for Farneback flow. Install with: pip install opencv-python"
        )
    n = prev_rgb.shape[0]
    frames_prev = prev_rgb.cpu().numpy()  # (N, H, W, 3) uint8
    frames_curr = curr_rgb.cpu().numpy()
    flows = np.empty((n, 2, frames_prev.shape[1], frames_prev.shape[2]), dtype=np.float32)
    for i in range(n):
        g_prev = cv2.cvtColor(frames_prev[i], cv2.COLOR_RGB2GRAY)
        g_curr = cv2.cvtColor(frames_curr[i], cv2.COLOR_RGB2GRAY)
        flow_hwc = cv2.calcOpticalFlowFarneback(
            g_prev,
            g_curr,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=2,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )  # (H, W, 2) float32
        flows[i] = flow_hwc.transpose(2, 0, 1)  # -> (2, H, W) channel-first
    return torch.from_numpy(flows).to(prev_rgb.device).float()


class V7FlowHoverEnv(ManagerBasedRLEnv):
    """Station keeping via global optical-flow nulling (GS-ROV)."""

    cfg: V7FlowHoverEnvCfg

    def __init__(self, cfg: V7FlowHoverEnvCfg, render_mode: str | None = None, **kwargs):
        # Camera + USD handles are created after the managers (see _build_custom_scene).
        self._tiled_camera = None
        self._robot_usd = None
        self._flow_token = -1
        # Allocate the stateful flow buffers up front so any manager init-time probe that
        # touches them (only observation terms do) sees valid zero tensors.
        self._allocate_flow_buffers(cfg)

        super().__init__(cfg, render_mode, **kwargs)

        self._build_custom_scene()

    # ------------------------------------------------------------------ properties
    @property
    def _robot(self):
        return self.scene["robot"]

    # ------------------------------------------------------------------ buffers
    def _allocate_flow_buffers(self, cfg: V7FlowHoverEnvCfg) -> None:
        n = int(cfg.scene.num_envs)
        dev = cfg.sim.device
        rh, rw = int(cfg.render_height), int(cfg.render_width)
        h, w = int(cfg.cam_height), int(cfg.cam_width)
        self._flow_image = torch.zeros((n, 2, h, w), dtype=torch.float32, device=dev)
        self._texture_mask = torch.zeros((n, 1, h, w), dtype=torch.float32, device=dev)
        self._flow_rms = torch.zeros(n, device=dev)
        self._texture_count = torch.zeros(n, device=dev)
        self._prev_rgb = torch.zeros((n, rh, rw, 3), dtype=torch.uint8, device=dev)
        self._prev_rgb_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self._texture_lost_streak = torch.zeros(n, dtype=torch.long, device=dev)

    # ------------------------------------------------------------------ scene build
    def _build_custom_scene(self) -> None:
        # Camera (adopted prim inside the cuboid), intrinsics, viewport camera, tiled sensor.
        self._spawn_robot_camera()
        self._apply_camera_intrinsics()
        self._spawn_view_camera()

        # Resolve the env-token to the concrete pattern the TiledCamera expects.
        self.cfg.tiled_camera.prim_path = "/World/envs/env_.*/Robot/ForwardCamera0"
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # The simulator has already been reset by now, so the PHYSICS_READY callback that
        # normally initializes sensors will never fire for this late-added camera.
        if not self._tiled_camera.is_initialized:
            self._tiled_camera._initialize_impl()
            self._tiled_camera._is_initialized = True

        # Textured plane + occluders, then the reference spheres.
        self._spawn_textured_plane()
        self._spawn_reference_spheres()

        # USD pose sync for the adopted camera prim.
        if bool(getattr(self.cfg, "cam_sync_usd_pose", True)):
            self._init_usd_pose_sync()
            self._sync_robot_usd_pose()

        # Populate camera buffers once.
        self.scene.update(dt=self.physics_dt)
        if self.sim.has_rtx_sensors() or self.sim.has_gui():
            self.sim.render()

    # ------------------------------------------------------------------ flow update
    def _ensure_flow_updated(self) -> None:
        """Compute flow/texture once per env-step (lazy, shared across obs/reward/termination)."""
        if self._tiled_camera is None:
            return
        if self._flow_token == self.common_step_counter:
            return
        self._update_flow_and_texture()
        self._update_texture_streak()
        self._flow_token = self.common_step_counter

    def _camera_rgb(self) -> torch.Tensor:
        rgb = self._tiled_camera.data.output["rgb"]
        if rgb.dtype != torch.uint8:
            rgb = rgb.clamp(0, 255).to(torch.uint8)
        return rgb[..., :3]

    def _read_camera_frame(self) -> torch.Tensor:
        rgb = self._camera_rgb().permute(0, 3, 1, 2).to(torch.float32)
        if rgb.shape[-2:] != (self.cfg.cam_height, self.cfg.cam_width):
            rgb = F.interpolate(rgb, size=(self.cfg.cam_height, self.cfg.cam_width), mode="area")
        return rgb.round().clamp_(0.0, 255.0).to(torch.uint8)

    def _texture_from_rgb(self, rgb_nhwc: torch.Tensor):
        gray = rgb_nhwc.float().mean(dim=-1)  # (N,Rh,Rw)
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[:, 1:-1, 1:-1] = gray[:, 1:-1, 2:] - gray[:, 1:-1, :-2]
        gy[:, 1:-1, 1:-1] = gray[:, 2:, 1:-1] - gray[:, :-2, 1:-1]
        mag = torch.sqrt(gx * gx + gy * gy)
        rh, rw = gray.shape[-2], gray.shape[-1]
        h, w = self.cfg.cam_height, self.cfg.cam_width
        hot = (mag > self.cfg.texture_grad_threshold).float().unsqueeze(1)  # (N,1,Rh,Rw)
        if (rh, rw) != (h, w):
            mask = F.adaptive_max_pool2d(hot, (h, w))
        else:
            mask = hot
        count = mask.flatten(1).sum(dim=-1)
        return mask, count

    def _update_flow_and_texture(self) -> None:
        curr_raw = self._camera_rgb()  # (N, render_h, render_w, 3) uint8
        h, w = int(self.cfg.cam_height), int(self.cfg.cam_width)

        tex_mask, tex_count = self._texture_from_rgb(curr_raw)
        self._texture_mask = tex_mask
        self._texture_count = tex_count

        fresh = ~self._prev_rgb_ok
        self._flow_image.zero_()
        self._flow_rms[:] = 0.0
        if bool((~fresh).any()):
            est = _farneback_flow(self._prev_rgb[~fresh], curr_raw[~fresh])
            small = F.interpolate(est, size=(h, w), mode="area")
            du = small[:, 0, :, :] / 1280.0
            dv = small[:, 1, :, :] / 720.0
            self._flow_image[~fresh, 0] = du.clamp(-1.0, 1.0)
            self._flow_image[~fresh, 1] = dv.clamp(-1.0, 1.0)
            mag_px = torch.sqrt(small[:, 0, :, :] ** 2 + small[:, 1, :, :] ** 2)
            tex = tex_mask[~fresh].squeeze(1)  # (m, H, W)
            tex_flat = tex.reshape(tex.shape[0], -1)
            mag_flat = mag_px.reshape(mag_px.shape[0], -1)
            has_tex = tex_flat.sum(dim=-1) >= 4
            sq = mag_flat * mag_flat
            rms_tex = torch.sqrt((sq * tex_flat).sum(dim=-1) / tex_flat.sum(dim=-1).clamp(min=1))
            rms_all = torch.sqrt(sq.mean(dim=-1))
            rms = torch.where(has_tex, rms_tex, rms_all)
            self._flow_rms[~fresh] = rms
        self._prev_rgb[:] = curr_raw
        self._prev_rgb_ok[:] = True
        self._flow_rms[fresh] = 0.0

    def _update_texture_streak(self) -> None:
        present = (self._texture_count >= self.cfg.min_texture_px).to(torch.float32)
        self._texture_lost_streak = torch.where(
            present > 0.5,
            torch.zeros_like(self._texture_lost_streak),
            self._texture_lost_streak + 1,
        )

    # ------------------------------------------------------------------ camera spawn
    def _spawn_robot_camera(self):
        """Forward-facing camera at the front face center of the proxy cuboid, pointing +X."""
        try:
            from pxr import Gf, UsdGeom

            half_x = 0.56 / 2.0
            for i in range(self.num_envs):
                env_p = self.scene.env_prim_paths[i]
                fwd_path = f"{env_p}/Robot/ForwardCamera0"
                fwd = self.sim.stage.DefinePrim(fwd_path, "Camera")
                xf = UsdGeom.Xformable(fwd)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(half_x, 0.0, 0.0))
                q_fwd = Gf.Quatf(0.0, 0.70710678, 0.0, -0.70710678)
                q_roll = Gf.Quatf(0.70710678, -0.70710678, 0.0, 0.0)
                xf.AddOrientOp().Set(q_roll * q_fwd)
                xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
                UsdGeom.Camera(fwd).GetFocalLengthAttr().Set(25.294921875)
        except Exception as e:
            print(f"[v7-mb] WARN camera spawn failed: {e}", flush=True)

    def _apply_camera_intrinsics(self):
        from pxr import Gf, UsdGeom

        hfov = float(self.cfg.cam_hfov_deg)
        aperture = float(self.cfg.cam_aperture_mm)
        aspect = float(self.cfg.render_width) / max(float(self.cfg.render_height), 1.0)
        focal = float(aperture * 0.5 / np.tan(np.radians(hfov) * 0.5))
        v_aperture = float(aperture / aspect)
        for i in range(self.num_envs):
            path = f"{self.scene.env_prim_paths[i]}/Robot/ForwardCamera0"
            prim = self.sim.stage.GetPrimAtPath(path)
            if prim is None or not prim.IsValid():
                continue
            cam = UsdGeom.Camera(prim)
            cam.GetFocalLengthAttr().Set(focal)
            cam.GetHorizontalApertureAttr().Set(aperture)
            cam.GetVerticalApertureAttr().Set(v_aperture)
            cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 200.0))

    def _spawn_view_camera(self):
        if not bool(getattr(self.cfg, "spawn_view_camera", True)):
            return
        try:
            from pxr import Gf, UsdGeom

            hfov = float(self.cfg.cam_hfov_deg)
            aperture = float(self.cfg.cam_aperture_mm)
            focal = float(aperture * 0.5 / np.tan(np.radians(hfov) * 0.5))
            v_aperture = float(aperture * self.cfg.render_height / self.cfg.render_width)
            off = self.cfg.view_camera_offset
            roll = float(np.radians(float(getattr(self.cfg, "view_camera_roll_deg", -90.0))))
            q_fwd = Gf.Quatf(0.0, 0.70710678, 0.0, -0.70710678)
            q_roll = Gf.Quatf(float(np.cos(roll * 0.5)), float(np.sin(roll * 0.5)), 0.0, 0.0)
            name = str(self.cfg.view_camera_name)
            for i in range(self.num_envs):
                path = f"{self.scene.env_prim_paths[i]}/Robot/{name}"
                prim = self.sim.stage.DefinePrim(path, "Camera")
                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(float(off[0]), float(off[1]), float(off[2])))
                xf.AddOrientOp().Set(q_roll * q_fwd)
                xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
                cam = UsdGeom.Camera(prim)
                cam.GetFocalLengthAttr().Set(focal)
                cam.GetHorizontalApertureAttr().Set(aperture)
                cam.GetVerticalApertureAttr().Set(v_aperture)
                cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 200.0))
        except Exception as e:
            print(f"[v7-mb] WARN view camera spawn failed: {e}", flush=True)

    # ------------------------------------------------------------------ plane / spheres
    def _default_plane_texture_candidates(self):
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
        for p in [
            "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/textures/ground_diff_4096.png",
            "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/textures/ground_detiled_2048.png",
        ]:
            if os.path.isfile(p):
                cands.append(p)
        for p in [
            "/home/lh/IsaacLab/data_downloads/OceanSim_assets/collected_rock/textures/cropped_rock.jpg",
            "/home/lh/IsaacLab/data_downloads/OceanSim_assets/collected_MHL/textures/mhl_3hz_high_smooth_scaled.jpg",
        ]:
            if os.path.isfile(p):
                cands.append(p)
        return cands

    def _resolve_plane_texture(self, rng_choice=None):
        import random

        p = getattr(self.cfg, "plane_texture_path", "") or ""
        if p and os.path.isfile(p):
            base = p
        else:
            envp = os.environ.get("V7_PLANE_TEXTURE")
            if envp and os.path.isfile(envp):
                base = envp
            else:
                cands = [
                    c
                    for c in (getattr(self.cfg, "plane_texture_paths", ()) or ())
                    if isinstance(c, str) and os.path.isfile(c)
                ]
                if cands:
                    base = random.choice(cands) if getattr(self.cfg, "randomize_plane_texture", False) else cands[0]
                else:
                    preferred = (
                        "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/"
                        "ground_lite/sand_01_8k/textures/sand_01_diff_256.jpg"
                    )
                    if os.path.isfile(preferred):
                        base = preferred
                    else:
                        dc = self._default_plane_texture_candidates()
                        base = dc[0] if dc else None
        if getattr(self.cfg, "randomize_plane_texture", False):
            cands = [
                c
                for c in (getattr(self.cfg, "plane_texture_paths", ()) or ())
                if isinstance(c, str) and os.path.isfile(c)
            ]
            if not cands:
                cands = self._default_plane_texture_candidates()
            if cands:
                base = random.choice(cands)
        return base

    def _spawn_plane_occluders(self, env_prim, ox, oy, oz, depth, size, dist, half, tile, tex, half_w=None, back=0.75):
        try:
            from pxr import Gf, Sdf, UsdGeom, UsdShade

            if half_w is None:
                half_w = half
            half_h = half
            z_lo = depth - half_h
            z_hi = depth + half_h
            y_lo = -half_w
            y_hi = half_w
            x_lo = -float(back)
            x_hi = dist
            dark_color = Gf.Vec3f(0.15, 0.25, 0.65)
            bottom_color = Gf.Vec3f(0.05, 0.12, 0.45)
            walls = [
                ("OccLeft", [
                    Gf.Vec3f(x_lo, y_lo, z_lo), Gf.Vec3f(x_lo, y_lo, z_hi),
                    Gf.Vec3f(x_hi, y_lo, z_hi), Gf.Vec3f(x_hi, y_lo, z_lo),
                ], Gf.Vec3f(0.0, 1.0, 0.0), dark_color),
                ("OccRight", [
                    Gf.Vec3f(x_lo, y_hi, z_lo), Gf.Vec3f(x_hi, y_hi, z_lo),
                    Gf.Vec3f(x_hi, y_hi, z_hi), Gf.Vec3f(x_lo, y_hi, z_hi),
                ], Gf.Vec3f(0.0, -1.0, 0.0), dark_color),
                ("OccBottom", [
                    Gf.Vec3f(x_lo, y_lo, z_lo), Gf.Vec3f(x_hi, y_lo, z_lo),
                    Gf.Vec3f(x_hi, y_hi, z_lo), Gf.Vec3f(x_lo, y_hi, z_lo),
                ], Gf.Vec3f(0.0, 0.0, 1.0), bottom_color),
                ("OccBack", [
                    Gf.Vec3f(x_lo, y_lo, z_lo), Gf.Vec3f(x_lo, y_lo, z_hi),
                    Gf.Vec3f(x_lo, y_hi, z_hi), Gf.Vec3f(x_lo, y_hi, z_lo),
                ], Gf.Vec3f(1.0, 0.0, 0.0), dark_color),
            ]
            for name, pts, normal, wall_color in walls:
                path = f"{env_prim}/{name}"
                mesh = UsdGeom.Mesh.Define(self.sim.stage, path)
                mesh.GetPointsAttr().Set(pts)
                mesh.GetFaceVertexCountsAttr().Set([4])
                mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
                mesh.GetNormalsAttr().Set([normal] * 4)
                stp = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
                stp.Set([Gf.Vec2f(0, 0), Gf.Vec2f(tile, 0), Gf.Vec2f(tile, tile), Gf.Vec2f(0, tile)])
                mat_path = f"{env_prim}/{name}Mat"
                mat = UsdShade.Material.Define(self.sim.stage, mat_path)
                sh = UsdShade.Shader.Define(self.sim.stage, mat_path + "/Preview")
                sh.CreateIdAttr("UsdPreviewSurface")
                sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(wall_color)
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
                sh.CreateInput("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
                mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)
                mesh.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        except Exception as e:
            print(f"[v7-mb] WARN occluder spawn failed: {e}", flush=True)

    def _spawn_textured_plane(self):
        try:
            from pxr import Gf, Sdf, UsdGeom, UsdShade

            depth = self.cfg.starting_depth
            size = float(self.cfg.plane_size)
            aspect = float(getattr(self.cfg, "plane_aspect", 1.0)) or 1.0
            dist = float(self.cfg.plane_distance)
            half_w = size * 0.5 * aspect
            half_h = size * 0.5
            tile = float(self.cfg.plane_texture_tile)
            dark_color = Gf.Vec3f(0.15, 0.25, 0.65)
            for ei in range(self.num_envs):
                env_prim = self.scene.env_prim_paths[ei]
                path = f"{env_prim}/FlowPlane"
                mesh = UsdGeom.Mesh.Define(self.sim.stage, path)
                pts = [
                    Gf.Vec3f(dist, -half_w, depth - half_h),
                    Gf.Vec3f(dist, half_w, depth - half_h),
                    Gf.Vec3f(dist, half_w, depth + half_h),
                    Gf.Vec3f(dist, -half_w, depth + half_h),
                ]
                mesh.GetPointsAttr().Set(pts)
                mesh.GetFaceVertexCountsAttr().Set([4])
                mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
                stp = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
                stp.Set([Gf.Vec2f(tile, 0), Gf.Vec2f(0, 0), Gf.Vec2f(0, tile), Gf.Vec2f(tile, tile)])
                mesh.GetPrim().CreateAttribute("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
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

                if getattr(self.cfg, "plane_occluders", True):
                    self._spawn_plane_occluders(env_prim, 0.0, 0.0, 0.0, depth, size, dist, half_h, tile, None, half_w)
        except Exception as e:
            print(f"[v7-mb] WARN plane spawn failed: {e}", flush=True)

    def _spawn_reference_spheres(self):
        try:
            from pxr import Gf, Sdf, UsdGeom, UsdShade

            r = float(self.cfg.ref_sphere_radius)
            cx = float(self.cfg.ref_sphere_offset_x)
            half = float(self.cfg.ref_sphere_half_side)
            depth = float(self.cfg.starting_depth)
            color = Gf.Vec3f(*self.cfg.ref_sphere_color)
            offsets = [
                (cx, -half, depth - half),
                (cx, -half, depth + half),
                (cx, half, depth - half),
                (cx, half, depth + half),
            ]
            for ei in range(self.num_envs):
                env_prim = self.scene.env_prim_paths[ei]
                for idx, (dx, dy, dz) in enumerate(offsets):
                    sphere_path = f"{env_prim}/RefSphere{idx}"
                    sphere = UsdGeom.Sphere.Define(self.sim.stage, sphere_path)
                    sphere.GetRadiusAttr().Set(r)
                    xf = UsdGeom.Xformable(sphere.GetPrim())
                    xf.ClearXformOpOrder()
                    xf.AddTranslateOp().Set(Gf.Vec3d(dx, dy, dz))
                    mat_path = f"{sphere_path}/Mat"
                    mat = UsdShade.Material.Define(self.sim.stage, mat_path)
                    sh = UsdShade.Shader.Define(self.sim.stage, mat_path + "/Preview")
                    sh.CreateIdAttr("UsdPreviewSurface")
                    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
                    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(color)
                    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
                    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                    UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(mat)
        except Exception as e:
            print(f"[v7-mb] WARN ref sphere spawn failed: {e}", flush=True)

    # ------------------------------------------------------------------ USD pose sync
    def _init_usd_pose_sync(self) -> None:
        from isaacsim.core.prims import XFormPrim

        robot_paths = [f"{p}/Robot" for p in self.scene.env_prim_paths]
        try:
            self._robot_usd = XFormPrim(robot_paths, reset_xform_properties=False)
        except Exception as exc:
            self._robot_usd = None
            print(f"[v7-mb] WARN: USD pose sync disabled: {exc}", flush=True)

    def _sync_robot_usd_pose(self) -> None:
        if self._robot_usd is None:
            return
        self._robot_usd.set_world_poses(
            positions=self._robot.data.root_pos_w,
            orientations=self._robot.data.root_quat_w,
            usd=True,
        )
