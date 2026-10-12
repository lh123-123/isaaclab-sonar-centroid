# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-v4-cam runtime probe: does the camera follow the AUV, and does the target AUV move?

Two questions, answered from runtime data only:

A. **Camera** -- the camera is a *child* prim of the robot. Under ``replicate_physics=True``
   PhysX keeps rigid-body poses in Fabric and never writes them back to USD, so a child prim
   would inherit a frozen authored transform and the view would never move. Two independent
   checks here:
     1. the camera prim's USD world pose vs the robot's simulated pose + mount offset (the
        renderer resolves child prims from USD, so this is what it actually uses);
     2. the rendered ``distance_to_image_plane`` at the pixel where the *live* camera pose
        projects the target, vs that pose's geometric camera-to-target range. A frozen render
        pose disagrees by metres (measured 23.47 m vs 1.01 m in the pre-fix v3-cam run).

B. **Target AUV motion** -- compare the motion model's command (``_target_pos`` /
   ``_target_state``, i.e. what ``BatchHoloTargetMotion`` produced) against the pose the
   simulator actually holds (``_target.data.root_pos_w``) and the USD pose the renderer sees.

  ./scripts/debug_auv_v4_motion_probe.sh --headless --num_envs 2 --steps 20

To *watch* it instead, run with the GUI (needs a display):

  ./scripts/debug_auv_v4_motion_probe.sh --ui --num_envs 1 --steps 300

With ``--ui`` the viewport is pointed at the onboard ``LookFollowCamera0`` by default, so the
window shows exactly the frames the policy consumes. Pass ``--viewport_cam persp`` for the free
camera instead. The window stays live (stepping the same action) until it is closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# NDJSON trace file, opt-in: set AUV_PROBE_LOG to a path to record a machine-readable trace.
# Left unset (the default) the probe only prints, so there is no hard-coded session artifact.
LOG_PATH: str | None = os.environ.get("AUV_PROBE_LOG")

# Below this camera-to-target-centre range the hulls can interpenetrate (the WarpAUV AABB is
# ~0.68 m long), so a depth ray at the target's centre pixel hits the target's own near hull
# instead of its centre and the render-vs-geometry comparison is meaningless.
MIN_CLEAN_RANGE = 0.8

# Tag this run so successive traces are distinguishable without diffing timestamps.
RUN_ID = os.environ.get("AUV_DEBUG_RUN_ID", "probe")

# --- why the UI stutters, and the fix -------------------------------------------------------
# The task renders once per *control* step: decimation == render_interval == 24, so
# `_sim_step_counter % render_interval == 0` fires exactly once per 0.2 s of sim time. Measured
# consequences (headless, 1 env):
#   step p50 = 112 ms (physics), +8.3 ms per extra rendered frame,
#   => 1 render/step = ~8 fps AND the image advances 200 ms of sim time per frame.
# Both are perceived as stutter. The physics cost is the env's own per-substep action chain
# (PID @120 Hz + hydrodynamics; `_apply_action` is 69% of a step, 4.24 ms/substep), which is
# v0 task semantics and not the thing to change for smoothness. Rendering is cheap, so for a
# *viewing* run the fix is to render more often: 4 renders/step costs +25 ms/step and lifts the
# visible rate to ~28 fps with the image advancing 50 ms of sim time per frame.
# This only applies to --ui; headless keeps the task default so probe numbers stay comparable.
UI_RENDER_INTERVAL_DEFAULT = 6


def log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str | None = None) -> None:
    """Append one NDJSON record, when a trace path is configured. Never raises."""
    if LOG_PATH is None:
        return
    rec = {
        "runId": RUN_ID if run_id is None else run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(LOG_PATH, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass


def set_viewport_camera(choice: str, cam_prim_path: str) -> None:
    """Point the GUI viewport at the onboard camera, or at the free perspective camera.

    ``set_active_camera`` is the same call ``scripts/demos/h1_locomotion.py`` uses to switch
    between its third-person and perspective views. Never raises: a viewport nicety must not
    abort a probe run.
    """
    from omni.kit.viewport.utility import get_viewport_from_window_name

    vp = get_viewport_from_window_name("Viewport")
    if vp is None:
        print("[probe] WARN: no 'Viewport' window found; leaving the GUI camera untouched", flush=True)
        return
    target = "/OmniverseKit_Persp" if choice == "persp" else cam_prim_path
    try:
        vp.set_active_camera(target)
        print(f"[probe] GUI viewport camera -> {target}", flush=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(f"[probe] WARN: could not set viewport camera to {target}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct-v4-cam camera-follow / target-motion probe.")
    parser.add_argument("--num_envs", type=int, default=2)
    # The target motion model has a 3 s soft start + a 2 s leash phase (15 and 25 RL steps at
    # 0.2 s/step), so a run shorter than ~40 steps cannot answer "does the target AUV move".
    parser.add_argument("--steps", type=int, default=40)
    # Keep the agent *slower* than the target's 0.5 m/s cruise so the two never interpenetrate:
    # at 0.4 the agent overtakes and then rides at a ~0.26 m gap, where the rendered depth at the
    # target's centre pixel is the target's own near hull, not its centre -- unusable for the
    # render-vs-geometry check.
    parser.add_argument("--action_v", type=float, default=0.12, help="Surge action (moves the AUV).")
    parser.add_argument("--action_w", type=float, default=0.0, help="Yaw action.")
    parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v4-cam")
    parser.add_argument("--out_dir", type=str, default="logs/auv_tracking/v4_cam_probe")
    parser.add_argument("--hang_timeout", type=float, default=420.0)
    parser.add_argument("--save_png", type=int, default=1)
    # --- GUI (viewing only; the probe is headless by default) --------------------------------
    parser.add_argument("--ui", action="store_true", help="Run with the Isaac Sim GUI (needs a display).")
    parser.add_argument(
        "--viewport_cam",
        choices=["robot", "persp"],
        default=None,
        help="GUI viewport camera: 'robot' = the onboard LookFollowCamera0 (default with --ui), "
        "'persp' = the free perspective camera.",
    )
    parser.add_argument(
        "--hold_s",
        type=float,
        default=0.0,
        help="With --ui, keep stepping and rendering this long after the probe; 0 = until the window is closed.",
    )
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.0,
        help="Sleep this long between steps in --ui mode so the motion is watchable "
        "(0 = step as fast as the machine allows). Pacing only: it never affects the sim.",
    )
    # --- render-frequency / measurement knobs -------------------------------------------------
    parser.add_argument(
        "--render_interval",
        type=int,
        default=None,
        help="Override cfg.sim.render_interval (physics steps between renders). The task default "
        "is 24 == decimation, i.e. ONE rendered frame per 0.2 s control step, so the visible motion "
        "advances 0.2 s per frame. Lower = smoother motion. Does not affect physics or policy obs.",
    )
    parser.add_argument(
        "--with_depth",
        type=int,
        default=1,
        help="Request distance_to_image_plane alongside rgb. Only the H6 render-vs-geometry check "
        "needs it; 0 removes the extra annotator from a viewing run.",
    )
    parser.add_argument(
        "--profile",
        type=int,
        default=0,
        help="cProfile the first N stepping iterations and print the hottest functions. "
        "Profiling adds overhead, so read the relative split, not the absolute ms.",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=None,
        help="Override cfg.decimation (physics substeps per control step). DIAGNOSTIC ONLY: it "
        "changes how much sim time a control step covers, so it alters task semantics. Used to "
        "split the per-substep cost from any per-step constant cost.",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    # The probe is headless unless the GUI is explicitly requested: it is also the CI/debug path.
    args_cli.headless = not bool(args_cli.ui)
    args_cli.enable_cameras = True

    import faulthandler

    faulthandler.enable()
    # Arm the watchdog only headless. With --ui the process is *meant* to stay alive while the
    # window is open, so a wall-clock dump+exit would kill the very run the user is watching.
    if args_cli.headless and float(args_cli.hang_timeout) > 0:
        faulthandler.dump_traceback_later(float(args_cli.hang_timeout), exit=True)
        print(f"[probe] watchdog armed: stack dump + exit after {args_cli.hang_timeout:.0f}s", flush=True)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch
    from pxr import Usd, UsdGeom

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.math import convert_camera_frame_orientation_convention, matrix_from_quat, quat_apply

    if args_cli.task.endswith("v4-cam"):
        from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v4_cam_env import (
            AUVTrackingV4CamEnvCfg as Cfg,
        )
        from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v4_cam_env import (
            CAM_MOUNT_OFFSET_LOCAL as CAM_OFF,
        )
    else:
        from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_cam_env import AUVTrackingCamEnvCfg as Cfg
        from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_cam_env import (
            CAM_MOUNT_OFFSET_LOCAL as CAM_OFF,
        )

    log("META", "debug_auv_v4_motion_probe.py:main", "probe start",
        {"task": args_cli.task, "num_envs": args_cli.num_envs, "steps": args_cli.steps,
         "action": [args_cli.action_v, args_cli.action_w], "log_path": LOG_PATH, "pid": os.getpid()})

    cfg = Cfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.render_interval is None and int(args_cli.ui):
        # Smooth by default when watching. Task semantics are untouched: render_interval only gates
        # how often the scene is drawn, and the USD pose sync keys off the same value.
        args_cli.render_interval = UI_RENDER_INTERVAL_DEFAULT
        print(f"[probe] --ui: defaulting render_interval to {UI_RENDER_INTERVAL_DEFAULT} for smooth "
              f"motion (pass --render_interval 24 for the task default)", flush=True)
    if args_cli.decimation is not None:
        print(f"[probe] decimation: cfg default {cfg.decimation} -> {args_cli.decimation} (DIAGNOSTIC)",
              flush=True)
        cfg.decimation = int(args_cli.decimation)
    if args_cli.render_interval is not None:
        print(f"[probe] render_interval: cfg default {cfg.sim.render_interval} -> {args_cli.render_interval}",
              flush=True)
        cfg.sim.render_interval = int(args_cli.render_interval)
    # Depth is a probe-only addition: the env itself reads only "rgb". It is what makes the
    # camera-pose check decisive rather than a judgement call on a near-featureless image.
    cfg.tiled_camera.data_types = ["rgb"] + (["distance_to_image_plane"] if int(args_cli.with_depth) else [])
    print(f"[probe] building env {args_cli.task} num_envs={args_cli.num_envs}...", flush=True)
    env = gym.make(args_cli.task, cfg=cfg)
    raw = env.unwrapped
    print("[probe] env created", flush=True)

    # Decisive check that the FOV change reached the renderer: the intrinsics the sensor actually
    # uses, not the cfg value. HFOV = 2*atan(W / (2*fx)); a 45 deg view gives fx ~= W/2/tan(22.5).
    #
    # IMPORTANT: the intrinsics, the depth buffer and `project_target`'s u,v are all in RENDER
    # pixels, while cam_width/cam_height are the OBSERVATION size (the render is area-averaged down
    # in _read_camera_frame). Those coincided while both were 64; they must not be conflated now.
    # Take the render size from the sensor's own output so this keeps working whatever the cfg says.
    K0 = raw._tiled_camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
    fx, fy = float(K0[0, 0]), float(K0[1, 1])
    render_h, render_w = (int(x) for x in raw._tiled_camera.data.output["rgb"].shape[1:3])
    hfov_actual = float(np.degrees(2.0 * np.arctan(0.5 * render_w / fx)))
    vfov_actual = float(np.degrees(2.0 * np.arctan(0.5 * render_h / fy)))
    # 1 px at 1 m, in the pixels the policy actually sees: focal length scales with resolution.
    px_per_m_at_1m = fx * float(raw.cfg.cam_width) / render_w
    log("H7", "debug_auv_v4_motion_probe.py:main", "camera intrinsics as used by the renderer",
        {"cfg_cam_hfov_deg": float(raw.cfg.cam_hfov_deg), "cfg_sensor_fov_deg": float(raw.cfg.sensor_fov_deg),
         "K": K0.tolist(), "fx": fx, "fy": fy, "hfov_deg_actual": round(hfov_actual, 3),
         "vfov_deg_actual": round(vfov_actual, 3), "px_per_m_at_1m": round(px_per_m_at_1m, 3),
         "obs_W": int(raw.cfg.cam_width), "obs_H": int(raw.cfg.cam_height),
         "render_W": render_w, "render_H": render_h})
    print(f"[probe] camera intrinsics: fx={fx:.3f} fy={fy:.3f} -> HFOV={hfov_actual:.2f} deg "
          f"VFOV={vfov_actual:.2f} deg  (cfg cam_hfov={raw.cfg.cam_hfov_deg}, "
          f"sensor_fov={raw.cfg.sensor_fov_deg})", flush=True)
    print(f"[probe] render {render_w}x{render_h} -> obs {int(raw.cfg.cam_width)}x"
          f"{int(raw.cfg.cam_height)}  (u,v below are in RENDER pixels)", flush=True)

    if not args_cli.headless:
        # env_0's onboard camera: the same prim the render product samples, so the window shows
        # literally the frames the policy is fed. getattr keeps this working for the v3-cam
        # fallback task, whose env class predates _camera_prim_paths().
        cam_paths = getattr(raw, "_camera_prim_paths", None)
        cam_path = cam_paths()[0] if callable(cam_paths) else "/World/envs/env_0/Robot/warpauv_base_link/LookFollowCamera0"
        set_viewport_camera(args_cli.viewport_cam or "robot", cam_path)

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Read everything through raw.cfg: _setup_scene() rewrites target_cfg.init_state in place, so
    # the caller's cfg object is only authoritative for values set before gym.make().
    cfg = raw.cfg

    # --- target AABB in body frame, read once at the authored (spawn) pose -------------------
    stage = raw.sim.stage
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    def prim_local_extents(prim):
        """Rotation-free local-space AABB of a prim. Excludes the prim's *own* xform ops."""
        rng_ = bbox_cache.ComputeLocalBound(prim).ComputeAlignedRange()
        lo_ = np.array(rng_.GetMin(), dtype=np.float64)
        hi_ = np.array(rng_.GetMax(), dtype=np.float64)
        return lo_, hi_

    def prim_world_scale(prim) -> np.ndarray:
        """Per-axis magnitude of the prim's composed world transform (uniform here)."""
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array([np.linalg.norm(np.array(m.GetRow(i), dtype=np.float64)[:3]) for i in range(3)])

    tgt_prim = stage.GetPrimAtPath("/World/envs/env_0/Target")
    lo, hi = prim_local_extents(tgt_prim)
    spawn = np.array(cfg.target_cfg.init_state.pos, dtype=np.float64)
    # NOTE: measured, not assumed -- `ComputeLocalBound()` already folds the prim's *own* scale
    # into the returned range. Scaling it again here double-counts (first run of this check
    # reported ratio 0.2500 for a 0.5 scale, i.e. exactly 0.5^2). So take the bounds as-is.
    tgt_s = prim_world_scale(tgt_prim)
    tgt_half = 0.5 * (hi - lo)
    tgt_center_off = 0.5 * (lo + hi) - spawn
    # Agent extents, for the "is the target actually half size" ratio. The agent is unscaled.
    rob_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    rlo, rhi = prim_local_extents(rob_prim)
    rob_len = rhi - rlo
    tgt_len = hi - lo
    size_ratio = float(tgt_len[0] / rob_len[0]) if rob_len[0] > 1e-9 else float("nan")
    print(f"[probe] target half-extents={np.round(tgt_half, 4).tolist()} centre_off={np.round(tgt_center_off, 4).tolist()}",
          flush=True)
    print(f"[probe] world lengths: robot={np.round(rob_len, 4).tolist()} target={np.round(tgt_len, 4).tolist()}"
          f" -> target/robot={size_ratio:.4f}  (prim scale={np.round(tgt_s, 3).tolist()},"
          f" cfg.target_scale={getattr(cfg, 'target_scale', None)})", flush=True)
    log("H5,H8", "debug_auv_v4_motion_probe.py:main", "target AABB at spawn",
        {"half_extents": np.round(tgt_half, 4).tolist(), "center_offset": np.round(tgt_center_off, 4).tolist(),
         "spawn_pos": spawn.tolist(),
         "prim_world_scale": np.round(tgt_s, 5).tolist(),
         "robot_world_len": np.round(rob_len, 4).tolist(), "target_world_len": np.round(tgt_len, 4).tolist(),
         "size_ratio_x": round(size_ratio, 4),
         "cfg_target_scale": getattr(cfg, "target_scale", None)})

    cam_off_t = torch.tensor(CAM_OFF, dtype=torch.float32)

    obs = env.reset()
    if not isinstance(obs, dict):
        obs = {"policy": obs}

    def live_cam_pose(env_i: int = 0):
        """Camera world pose derived from the *simulated* robot pose + mount offset."""
        rb = raw._robot.data.root_pos_w[env_i].detach().cpu()
        rq = raw._robot.data.root_quat_w[env_i].detach().cpu()
        pos = rb + quat_apply(rq.unsqueeze(0), cam_off_t.unsqueeze(0)).squeeze(0)
        quat_ros = convert_camera_frame_orientation_convention(rq.unsqueeze(0), origin="world", target="ros").squeeze(0)
        return pos.numpy().astype(np.float64), quat_ros.numpy().astype(np.float64)

    def project_target(env_i: int = 0):
        """Project the *live* target pose through the *live* camera pose -> pixel + ranges."""
        tgt_pos = raw._target.data.root_pos_w[env_i].detach().cpu()
        tgt_quat = raw._target.data.root_quat_w[env_i].detach().cpu()
        pos_w, quat_ros = live_cam_pose(env_i)
        R = matrix_from_quat(torch.as_tensor(quat_ros).unsqueeze(0)).squeeze(0).numpy().astype(np.float64)
        K = raw._tiled_camera.data.intrinsic_matrices[env_i].detach().cpu().numpy().astype(np.float64)
        ctr_body = torch.as_tensor(tgt_center_off, dtype=torch.float32)
        ctr_w = (tgt_pos + quat_apply(tgt_quat.unsqueeze(0), ctr_body.unsqueeze(0)).squeeze(0)).numpy().astype(np.float64)
        p = R.T @ (ctr_w - pos_w)
        rng = float(np.linalg.norm(ctr_w - pos_w))
        if p[2] <= 1e-6:
            return float("nan"), float("nan"), rng, ctr_w, pos_w
        u = float(K[0, 0] * p[0] / p[2] + K[0, 2])
        v = float(K[1, 1] * p[1] / p[2] + K[1, 2])
        return u, v, rng, ctr_w, pos_w

    def depth_at(u: float, v: float, env_i: int = 0) -> float:
        """Rendered depth (m) at the given pixel, or NaN if unavailable.

        Returns NaN when the pixel is not *strictly* inside the image. Clamping instead (as an
        earlier revision did) silently samples the image border and reports metres of background,
        which reads as a bogus mismatch -- the target's centre pixel is frequently off-image here
        because the 45 deg HFOV is narrow relative to the leash geometry.
        """
        d = raw._tiled_camera.data.output.get("distance_to_image_plane")
        if d is None or not np.isfinite(u) or not np.isfinite(v):
            return float("nan")
        if not (0.0 <= u <= render_w - 1 and 0.0 <= v <= render_h - 1):
            return float("nan")
        d = d[env_i]
        d = d[..., 0] if d.ndim == 3 else d
        arr = d.detach().cpu().numpy().astype(np.float64)
        return float(arr[int(round(v)), int(round(u))])

    def luma(rgb: torch.Tensor) -> np.ndarray:
        """Collapse a (N, H, W, C) uint8 RGB frame to an (H, W) uint8 luma image on the host.

        ``rgb`` is the *render* buffer, so every luma statistic derived from this (``luma_mean``,
        ``luma_sha1_12``, the saved PNGs) is at RENDER resolution, not the observation's. That is
        deliberate for the frame-change checks: it sees the renderer's actual output, and it keeps
        the saved PNGs legible. Do not read these numbers as "what the policy sees" -- H9 compares
        the two resolutions explicitly.
        """
        f = rgb[0, ..., :3].to(torch.float32)
        y = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).round().clamp_(0, 255)
        return y.to(torch.uint8).cpu().numpy()

    def save_png(arr: np.ndarray, path: str, scale: int = 4) -> None:
        from PIL import Image

        img = Image.fromarray(arr)
        if scale > 1:
            img = img.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
        img.save(path)

    actions = torch.zeros((raw.num_envs, raw.action_space.shape[-1]), device=raw.device)
    actions[:, 0] = float(args_cli.action_v)
    actions[:, 1] = float(args_cli.action_w)

    def target_pixel_hits(lum_np: np.ndarray, u: float, v: float) -> tuple[int | None, list | None, float | None]:
        """Count target-like pixels in a +-40 render-pixel window around the projected centre.

        The orange albedo renders near luma 130-140 (vs ~238 for open water), so a plain threshold
        separates the vehicle from the background. The window keeps the count specific to the object
        being projected -- the seafloor is dark too, and a whole-frame count would include it.

        Returns ``(hits, bbox, window_min_luma)``; ``hits == 0`` with a window minimum equal to the
        background level means the object is genuinely absent from the frame rather than merely
        shifted, which is the distinction the reset-frame question turns on.
        """
        if not (np.isfinite(u) and np.isfinite(v)):
            return None, None, None
        x0, x1 = int(max(0, u - 40)), int(min(render_w, u + 41))
        y0, y1 = int(max(0, v - 40)), int(min(render_h, v + 41))
        patch = lum_np[y0:y1, x0:x1]
        if not patch.size:
            return None, None, None
        win_min = round(float(patch.min()), 1)
        m = patch < 200
        hits = int(m.sum())
        bbox = None
        if hits:
            ys, xs = np.nonzero(m)
            bbox = [int(x0 + xs.min()), int(x0 + xs.max()), int(y0 + ys.min()), int(y0 + ys.max())]
        return hits, bbox, win_min

    # --- what does the FIRST observation of an episode actually contain? ----------------------
    # reset() teleports the robot and the target but does not render, and _get_observations()
    # then reads whatever is in the camera buffer. So the policy's first observation is built
    # from a buffer that predates the teleport. Measuring it here, before any step, is exactly
    # measuring that buffer: this is the observation the actor is handed at t=0.
    _obuf = raw._tiled_camera.data.output["rgb"]
    if _obuf.dtype != torch.uint8:
        _obuf = _obuf.to(torch.uint8)
    _u, _v, _rng, _ctr, _ = project_target(0)
    _reset_luma = luma(_obuf)
    _hits, _bbox, _wmin = target_pixel_hits(_reset_luma, _u, _v)
    print(
        f"[probe] RESET BUFFER (first policy obs): target projects to px=[{_u:.1f}, {_v:.1f}]"
        f" rng={_rng:.2f}m -> hits={_hits} win_min_luma={_wmin}",
        flush=True,
    )

    # Did the first observation come from a frame that predates the teleport? reset() does not
    # render, so _get_observations() reads whatever happens to be in the camera buffer. Rendering
    # once here and re-reading says whether that buffer already described the post-reset world or
    # was a leftover: if the frame changes, the observation the actor was handed at t=0 was stale.
    raw.sim.render()
    _after = luma(raw._tiled_camera.data.output["rgb"].to(torch.uint8) if raw._tiled_camera.data.output["rgb"].dtype != torch.uint8 else raw._tiled_camera.data.output["rgb"])
    _h2, _b2, _w2 = target_pixel_hits(_after, _u, _v)
    _mean_shift = float(np.abs(_after.astype(np.int16) - _reset_luma.astype(np.int16)).mean())
    print(
        f"[probe] RESET BUFFER after explicit sim.render(): hits={_h2} win_min_luma={_w2}"
        f"  mean|Δ| vs the buffer the first obs used = {_mean_shift:.3f} luma levels",
        flush=True,
    )

    rows = []
    prev_luma = None
    prev_cam_usd = None
    prev_robot = None
    t_step_list: list[float] = []
    t_extra_list: list[float] = []
    _n_prof = int(args_cli.profile)
    _prof = None
    if _n_prof > 0:
        import cProfile

        _prof = cProfile.Profile()
        _prof.enable()
    for step in range(int(args_cli.steps)):
        if _prof is not None and step == _n_prof:
            _prof.disable()  # stop after N iterations so startup is excluded
        _t0 = time.perf_counter()
        obs, _rew, _term, _trunc, _info = env.step(actions)
        _t1 = time.perf_counter()
        if not isinstance(obs, dict):
            obs = {"policy": obs}
        # v4-cam's asymmetric split makes obs["policy"] a nested Dict ({"image", "state"}), while
        # v3-cam's is a bare tensor. The frame-stack questions (H2/H3) are about the image either
        # way, so dig out "image" instead of assuming a tensor.
        policy = obs["policy"]
        image_stack = (policy["image"] if isinstance(policy, dict) else policy).clone()
        stack_shifts = bool(
            (image_stack[0, : max(cfg.frame_stack - 1, 1)] != image_stack[0, -max(cfg.frame_stack - 1, 1) :]).any()
        )

        rgb = raw._tiled_camera.data.output["rgb"]
        if rgb.dtype != torch.uint8:
            rgb = rgb.to(torch.uint8)
        lum = luma(rgb)
        d_frame = np.abs(lum.astype(np.int16) - prev_luma.astype(np.int16)).mean() if prev_luma is not None else float("nan")

        u, v, rng, ctr_w, cam_pos = project_target(0)
        d_at = depth_at(u, v)

        # --- is the target in the frame at all? ----------------------------------------------
        # `luma()` already returns the (H, W) host array for env 0, so there is nothing to unwrap.
        tgt_hits, tgt_bbox, tgt_win_min = target_pixel_hits(lum, u, v)

        # USD xform / visibility of the Target prim, i.e. what the *stage* says versus what Fabric
        # says (`tgt_sim`, read from the rigid-object data). A divergence here would mean the
        # renderer is being handed two different poses.
        tgt_prim = stage.GetPrimAtPath("/World/envs/env_0/Target")
        tgt_usd, tgt_vis = None, None
        if tgt_prim is not None and tgt_prim.IsValid():
            tgt_usd = np.array(
                UsdGeom.Xformable(tgt_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation(),
                dtype=np.float64,
            )
            try:
                tgt_vis = str(UsdGeom.Imageable(tgt_prim).ComputeVisibility())
            except Exception:
                tgt_vis = "unreadable"
        # The depth-at-target-pixel check is only meaningful when the target's centre projects
        # strictly inside the image AND the hulls are not interpenetrating (2 x 0.34 m half-length
        # -> below ~0.8 m the ray hits the target's own near hull, not its centre).
        in_fov = bool(np.isfinite(u) and 0 <= u < render_w and 0 <= v < render_h)
        depth_valid = bool(in_fov and np.isfinite(d_at) and rng >= MIN_CLEAN_RANGE)
        robot_pos = raw._robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        tgt_cmd = raw._target_pos[0].detach().cpu().numpy().astype(np.float64)
        tgt_sim = raw._target.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        tgt_vel = raw._target_state[0, 2:4].detach().cpu().numpy().astype(np.float64)

        # Camera prim USD pose: exactly what the renderer inherits for the child camera.
        cam_usd = None
        cp = stage.GetPrimAtPath("/World/envs/env_0/Robot/warpauv_base_link/LookFollowCamera0")
        if cp is not None and cp.IsValid():
            m = UsdGeom.Xformable(cp).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            cam_usd = np.array(m.ExtractTranslation(), dtype=np.float64)
        cam_usd_step = float(np.linalg.norm(cam_usd - prev_cam_usd)) if (cam_usd is not None and prev_cam_usd is not None) else float("nan")
        robot_step = float(np.linalg.norm(robot_pos - prev_robot)) if prev_robot is not None else float("nan")

        row = {
            "step": step,
            "t_step_s": round(_t1 - _t0, 5),
            "robot_xy": [round(float(x), 3) for x in robot_pos[:2]],
            "robot_step_m": round(robot_step, 4),
            "cam_geo_pos": [round(float(x), 3) for x in cam_pos],
            "cam_usd_pos": None if cam_usd is None else [round(float(x), 3) for x in cam_usd],
            "cam_usd_err_m": None if cam_usd is None else round(float(np.linalg.norm(cam_usd - cam_pos)), 4),
            "cam_usd_step_m": round(cam_usd_step, 4),
            "tgt_cmd_xy": [round(float(x), 3) for x in tgt_cmd],
            "tgt_sim_xy": [round(float(x), 3) for x in tgt_sim],
            "tgt_cmd_vs_sim_m": round(float(np.linalg.norm(tgt_cmd - tgt_sim[:2])), 4),
            "tgt_speed_xy": round(float(np.linalg.norm(tgt_vel)), 4),
            "tgt_range_m": round(rng, 3),
            "tgt_px": [round(u, 1), round(v, 1)],
            "tgt_in_fov": in_fov,
            "tgt_usd_pos": None if tgt_usd is None else [round(float(x), 3) for x in tgt_usd],
            "tgt_usd_vs_sim_m": (
                None if tgt_usd is None else round(float(np.linalg.norm(tgt_usd - tgt_sim)), 4)
            ),
            "tgt_prim_vis": tgt_vis,
            "tgt_px_hits": tgt_hits,
            "tgt_px_bbox": tgt_bbox,
            "tgt_win_min_luma": tgt_win_min,
            "depth_at_tgt_px_m": None if not np.isfinite(d_at) else round(d_at, 3),
            "depth_check_valid": depth_valid,
            "depth_matches_geom": (None if not depth_valid else bool(0.5 * rng <= d_at <= 1.6 * rng)),
            "luma_mean": round(float(lum.mean()), 2),
            "luma_mean_abs_diff_vs_prev": None if not np.isfinite(d_frame) else round(float(d_frame), 4),
            "luma_sha1_12": hashlib.sha1(lum.tobytes()).hexdigest()[:12],
            "stack_oldest_vs_newest_differ": stack_shifts,
        }
        rows.append(row)
        print(f"[probe {step:03d}] robot_xy={row['robot_xy']} moved={row['robot_step_m']:.3f}m"
              f" cam_usd_err={row['cam_usd_err_m']} cam_usd_step={row['cam_usd_step_m']:.3f}m"
              f" tgt_cmd={row['tgt_cmd_xy']} tgt_sim={row['tgt_sim_xy']} |cmd-sim|={row['tgt_cmd_vs_sim_m']:.4f}"
              f" tgt_v={row['tgt_speed_xy']:.3f} rng={row['tgt_range_m']:.2f} px={row['tgt_px']} in_fov={row['tgt_in_fov']}"
              f" tgt_vis={row['tgt_prim_vis']} usd_sim={row['tgt_usd_vs_sim_m']} hits={row['tgt_px_hits']}"
              f" win_min={row['tgt_win_min_luma']}"
              f" depth={row['depth_at_tgt_px_m']} valid={row['depth_check_valid']} match={row['depth_matches_geom']}"
              f" luma={row['luma_mean']:.1f} d={row['luma_mean_abs_diff_vs_prev']}", flush=True)
        log("H1,H2,H4,H5,H6", "debug_auv_v4_motion_probe.py:step", "probe step", row)

        if int(args_cli.save_png) and (step <= 5 or step == int(args_cli.steps) - 1 or step % 5 == 0):
            save_png(lum, str(out_dir / f"luma_step{step:03d}.png"))
            d = raw._tiled_camera.data.output.get("distance_to_image_plane")
            if d is not None:
                dd = d[0]
                dd = dd[..., 0] if dd.ndim == 3 else dd
                dd = dd.detach().cpu().numpy()
                fin = np.isfinite(dd)
                if fin.any():
                    norm = np.zeros_like(dd, dtype=np.uint8)
                    lo_d, hi_d = float(np.percentile(dd[fin], 1)), float(np.percentile(dd[fin], 99))
                    norm[fin] = np.clip(255.0 * (dd[fin] - lo_d) / max(hi_d - lo_d, 1e-6), 0, 255).astype(np.uint8)
                    save_png(norm, str(out_dir / f"depth_step{step:03d}.png"))

        if not args_cli.headless and args_cli.step_delay > 0.0:
            time.sleep(float(args_cli.step_delay))  # pacing for the GUI view only

        _t2 = time.perf_counter()
        t_step_list.append(_t1 - _t0)
        t_extra_list.append(_t2 - _t1)

        prev_luma = lum
        prev_cam_usd = cam_usd
        prev_robot = robot_pos

    # --- verdicts ---------------------------------------------------------------------------
    fin = [r for r in rows if r["cam_usd_err_m"] is not None]
    max_err = max((r["cam_usd_err_m"] for r in fin), default=float("nan"))
    robot_moved = max((r["robot_step_m"] for r in rows if np.isfinite(r["robot_step_m"])), default=0.0)
    cam_moved = max((r["cam_usd_step_m"] for r in rows if np.isfinite(r["cam_usd_step_m"])), default=0.0)
    luma_diffs = [r["luma_mean_abs_diff_vs_prev"] for r in rows if r["luma_mean_abs_diff_vs_prev"] is not None]
    uniq = len({r["luma_sha1_12"] for r in rows})
    depth_checks = [r["depth_matches_geom"] for r in rows if r["depth_check_valid"]]
    n_fov = sum(1 for r in rows if r["tgt_in_fov"])
    tgt_spd = max((r["tgt_speed_xy"] for r in rows), default=0.0)
    tgt_cmd_err = max((r["tgt_cmd_vs_sim_m"] for r in rows), default=float("nan"))
    rng_min = min((r["tgt_range_m"] for r in rows), default=float("nan"))
    rng_max = max((r["tgt_range_m"] for r in rows), default=float("nan"))

    # H1's real test is the *error bound*: the camera prim's USD world pose vs the pose derived
    # from the robot's simulated pose must agree. The motion clause only exists to rule out the
    # degenerate "both are static" case, so it must be relative to how far the robot actually
    # moved -- an earlier absolute threshold (0.05 m/step) reported a false CONFIRMED whenever an
    # episode happened to start gently (measured: 0.0429 m/step, i.e. perfectly tracking).
    tracks_robot = (
        np.isfinite(max_err) and max_err < 0.05 and cam_moved > 0.01 and cam_moved >= 0.5 * robot_moved
    )
    v_h1 = ("REJECTED (camera pose tracks the robot)" if tracks_robot
            else "CONFIRMED (camera USD pose does not track the robot)")
    v_h2 = "REJECTED (frames do change)" if (luma_diffs and max(luma_diffs) > 0.5) else "CONFIRMED (frames static)"
    v_h3 = "CONFIRMED (duplicate frames)" if uniq < len(rows) else "REJECTED (all frames unique)"
    v_h4 = "REJECTED (target moves)" if tgt_spd > 0.05 else "CONFIRMED (target stays still)"
    # NOTE: _target_pos and _target.data.root_pos_w both come from the motion model (the env writes
    # the model pose into the sim every tick), so this comparison only proves the write reached the
    # sim view -- it is NOT independent evidence that the target moves. H4 carries that.
    v_h5 = "REJECTED (model pose present in sim view)" if (np.isfinite(tgt_cmd_err) and tgt_cmd_err < 0.05) else "CONFIRMED (command != simulated pose)"
    if not depth_checks:
        v_depth = "INCONCLUSIVE (no sample had the target strictly inside the image beyond " f"{MIN_CLEAN_RANGE} m)"
    elif all(depth_checks):
        v_depth = f"REJECTED (renderer agrees with the live camera pose in {len(depth_checks)}/{len(depth_checks)} clean samples)"
    else:
        v_depth = f"CONFIRMED ({sum(depth_checks)}/{len(depth_checks)} clean samples matched)"

    hfov_err = abs(hfov_actual - float(raw.cfg.cam_hfov_deg))
    v_h7 = ("REJECTED (renderer HFOV matches the configured value)" if hfov_err <= 2.0
            else f"CONFIRMED (renderer HFOV {hfov_actual:.2f} deg != configured {float(raw.cfg.cam_hfov_deg):.2f} deg)")

    want_scale = float(getattr(raw.cfg, "target_scale", 1.0))
    ratio_err = abs(size_ratio - want_scale)
    v_h8 = ("REJECTED (target world size matches cfg.target_scale)" if ratio_err <= 0.02
            else f"CONFIRMED (target/robot size ratio {size_ratio:.4f} != cfg.target_scale {want_scale:.4f})")

    # --- H9: the observation is exactly luma + area-downsample of the render -------------------
    # The invariant the 256x256 -> 64x64 path introduces. Recomputed independently from the live
    # sensor output rather than by calling raw._read_camera_frame() again, which would only agree
    # with itself. This pins down the documented recipe: Rec.601 weights, mode="area", a single
    # rounding, and downsampling BEFORE quantisation.
    h9_max, h9_mean_render, h9_mean_obs, h9_render_shape, h9_obs_shape = (None,) * 5
    v_h9 = "INCONCLUSIVE (no frame buffer to compare against)"
    _buf = getattr(raw, "_frame_buf", None)
    _rgb = raw._tiled_camera.data.output.get("rgb")
    if _buf is not None and _rgb is not None:
        _r = _rgb if _rgb.dtype == torch.uint8 else _rgb.to(torch.uint8)
        _r = _r[..., :3].to(torch.float32)
        _rl = 0.299 * _r[..., 0] + 0.587 * _r[..., 1] + 0.114 * _r[..., 2]  # (N, Rh, Rw) float
        _want = (
            torch.nn.functional.interpolate(
                _rl.unsqueeze(1), size=(int(raw.cfg.cam_height), int(raw.cfg.cam_width)), mode="area"
            )
            .round()
            .clamp_(0.0, 255.0)
            .to(torch.uint8)[:, 0]
        )  # (N, H, W)
        _got = _buf[:, -1]
        h9_render_shape = tuple(int(x) for x in _rl.shape[-2:])
        h9_obs_shape = tuple(int(x) for x in _got.shape[-2:])
        h9_max = int((_want.to(torch.int16) - _got.to(torch.int16)).abs().max())
        # Implementation-independent cross-check: area averaging is a weighted mean, so it must
        # preserve the image mean. Bilinear or nearest would drift away from this.
        h9_mean_render = float(_rl.mean())
        h9_mean_obs = float(_got.float().mean())
        v_h9 = (
            f"REJECTED (obs == documented luma+area transform of the {h9_render_shape[1]}x"
            f"{h9_render_shape[0]} render -> {h9_obs_shape[1]}x{h9_obs_shape[0]} obs;"
            f" max|diff|={h9_max} LSB; mean render {h9_mean_render:.3f} vs obs {h9_mean_obs:.3f})"
            if h9_max <= 1
            else f"CONFIRMED (obs != documented transform; max|diff|={h9_max} LSB)"
        )

    # --- performance summary (stutter investigation) -------------------------------------------
    def pctl(xs: list[float], q: float) -> float:
        return float(np.percentile(xs, q)) if xs else float("nan")

    decim = int(raw.cfg.decimation)
    rint = int(raw.cfg.sim.render_interval)
    renders_per_step = decim / max(rint, 1)
    control_dt = float(raw.cfg.sim.dt) * decim
    sim_s_per_frame = control_dt / renders_per_step
    sleep_total = float(args_cli.step_delay) * len(rows) if not args_cli.headless else 0.0
    wall_total = sum(t_step_list) + sum(t_extra_list) + sleep_total
    perf = {
        "decimation": decim,
        "render_interval": rint,
        "renders_per_control_step": round(renders_per_step, 3),
        "control_dt_s": round(control_dt, 4),
        "rendered_frames_per_sim_second": round(renders_per_step / control_dt, 2),
        "sim_seconds_per_rendered_frame": round(sim_s_per_frame, 4),
        "with_depth": int(args_cli.with_depth),
        "ui": (not args_cli.headless),
        "step_delay_s": float(args_cli.step_delay),
        "t_step_ms_p50": round(1000 * pctl(t_step_list, 50), 2),
        "t_step_ms_p90": round(1000 * pctl(t_step_list, 90), 2),
        "t_extra_ms_p50": round(1000 * pctl(t_extra_list, 50), 2),
        "t_extra_ms_p90": round(1000 * pctl(t_extra_list, 90), 2),
        "wall_fps_measured": round(len(rows) / wall_total, 2) if wall_total > 0 else None,
    }
    print("\n[probe] ===== PERF =====", flush=True)
    for k, v in perf.items():
        print(f"[probe]   {k} = {v}", flush=True)
    log("H-P1,H-P2,H-P3,H-P4,H-P5", "debug_auv_v4_motion_probe.py:perf", "performance summary", perf)

    if _prof is not None:
        import io
        import pstats

        _prof.disable()
        buf = io.StringIO()
        st = pstats.Stats(_prof, stream=buf).sort_stats("tottime")
        st.print_stats(18)
        top = buf.getvalue()
        print(f"\n[probe] ===== PROFILE (first {_n_prof} steps, by tottime) =====", flush=True)
        for line in top.splitlines():
            if line.strip():
                print(f"[prof] {line}", flush=True)
        # Also surface the cumulative view of the step entry point, which is what tells us
        # whether the time is in physics, sensors or the env's own bookkeeping.
        buf2 = io.StringIO()
        pstats.Stats(_prof, stream=buf2).sort_stats("cumulative").print_stats(12)
        for line in buf2.getvalue().splitlines():
            if line.strip():
                print(f"[prof-cum] {line}", flush=True)

    v_hp = (
        f"CONFIRMED (1 render per control step -> the view advances {sim_s_per_frame * 1000:.0f} ms of "
        f"sim time every frame = {renders_per_step / control_dt:.1f} frames/sim-second)"
        if renders_per_step <= 1.001
        else f"REJECTED ({renders_per_step:.1f} renders per control step -> "
        f"{renders_per_step / control_dt:.1f} frames/sim-second)"
    )

    summary = {
        "robot_moved_max_step_m": round(robot_moved, 4),
        "cam_usd_err_max_m": None if not np.isfinite(max_err) else round(max_err, 5),
        "cam_usd_moved_max_step_m": round(cam_moved, 4),
        "unique_frames": uniq, "n_steps": len(rows),
        "luma_diff_max": None if not luma_diffs else round(max(luma_diffs), 4),
        "stack_shifts": sum(1 for r in rows if r["stack_oldest_vs_newest_differ"]),
        "target_speed_max": round(tgt_spd, 4),
        "target_range_min": round(rng_min, 3), "target_range_max": round(rng_max, 3),
        "target_in_fov_steps": f"{n_fov}/{len(rows)}",
        "target_cmd_vs_sim_max_m": None if not np.isfinite(tgt_cmd_err) else round(tgt_cmd_err, 4),
        "depth_clean_samples": f"{len(depth_checks)}/{len(rows)}",
        "depth_checks_matched": f"{sum(depth_checks)}/{len(depth_checks)}",
        "cam_hfov_deg_actual": round(hfov_actual, 3),
        "cam_vfov_deg_actual": round(vfov_actual, 3),
        "cfg_cam_hfov_deg": float(raw.cfg.cam_hfov_deg),
        "cfg_sensor_fov_deg": float(raw.cfg.sensor_fov_deg),
        "px_per_m_at_1m": round(px_per_m_at_1m, 3),
        "target_world_len_m": round(float(tgt_len[0]), 4),
        "robot_world_len_m": round(float(rob_len[0]), 4),
        "target_vs_robot_size_ratio": round(size_ratio, 4),
        "cfg_target_scale": want_scale,
        "render_hw": None if h9_render_shape is None else list(h9_render_shape),
        "obs_hw": None if h9_obs_shape is None else list(h9_obs_shape),
        "render_vs_obs_max_diff_lsb": h9_max,
        "render_mean_luma": None if h9_mean_render is None else round(h9_mean_render, 3),
        "obs_mean_luma": None if h9_mean_obs is None else round(h9_mean_obs, 3),
    }
    print("\n[probe] ===== VERDICTS =====", flush=True)
    print(f"[probe] H1 frozen camera prim : {v_h1}", flush=True)
    print(f"[probe] H2 static frames      : {v_h2}", flush=True)
    print(f"[probe] H3 duplicate frames   : {v_h3}", flush=True)
    print(f"[probe] H4 target motion model: {v_h4}", flush=True)
    print(f"[probe] H5 model pose in sim  : {v_h5}", flush=True)
    print(f"[probe] H6 render-vs-geometry : {v_depth}", flush=True)
    print(f"[probe] H7 camera HFOV applied: {v_h7}", flush=True)
    print(f"[probe] H8 target scaled      : {v_h8}", flush=True)
    print(f"[probe] H9 render->obs        : {v_h9}", flush=True)
    print(f"[probe] HP render cadence     : {v_hp}", flush=True)
    print(f"[probe] summary: {json.dumps(summary)}", flush=True)
    log("VERDICT", "debug_auv_v4_motion_probe.py:verdict", "verdicts",
        {"H1": v_h1, "H2": v_h2, "H3": v_h3, "H4": v_h4, "H5": v_h5, "H6": v_depth, "H7": v_h7, "H8": v_h8,
         "H9": v_h9, "HP": v_hp, "perf": perf, "summary": summary})

    print(f"[probe] PNGs in {out_dir}", flush=True)

    if not args_cli.headless:
        hold = float(args_cli.hold_s)
        print(
            "[probe] GUI hold: "
            + ("stepping until the window is closed" if hold <= 0 else f"stepping for {hold:.0f}s")
            + " -- close the Isaac Sim window to exit",
            flush=True,
        )
        # Keep stepping the same action so the view stays live rather than showing a frozen frame.
        # Exits on window close (is_running() -> False) or after hold_s.
        t_hold = time.time()
        while simulation_app.is_running():
            env.step(actions)
            if args_cli.step_delay > 0.0:
                time.sleep(float(args_cli.step_delay))  # pacing for the GUI view only
            if hold > 0.0 and (time.time() - t_hold) >= hold:
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
