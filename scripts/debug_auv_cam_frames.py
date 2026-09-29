# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the Direct-v3-cam camera: orientation, framing, and target visibility.

Must pass before a training run. Three things are checked, in increasing strength:

1. Observation contract -- shape / dtype / frame-stack integrity.
2. Geometric framing -- project the target WarpAUV's simulated AABB through the camera
   intrinsics and check it lands inside the image at the expected pixel size. The target
   is now a second WarpAUV (same colours as the agent), so pixel-colour heuristics cannot
   tell us whether it is visible; geometry can.
3. Render pose -- DECISIVE. With ``replicate_physics=True`` PhysX writes poses into Fabric
   and the USD xforms stay frozen, so ``TiledCamera.data.pos_w`` reports the *spawn*
   pose forever. We therefore derive the camera pose from the robot and compare the
   rendered depth image against the geometric target range: if the renderer also used the
   frozen pose, the two would disagree by metres.

  ./scripts/debug_auv_cam_frames.sh --headless --num_envs 2 --steps 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _px_per_m(cfg, range_m: float) -> float:
    """Horizontal px per metre for a fronto-parallel object at ``range_m``."""
    hfov = np.radians(float(cfg.cam_hfov_deg))
    return float(cfg.cam_width) / (2.0 * float(range_m) * np.tan(hfov * 0.5))


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct-v3-cam camera / frame-stack check.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="logs/auv_tracking/v3_cam_debug")
    args, unknown = parser.parse_known_args()

    import sys

    import torch

    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]] + unknown
    live_parser = argparse.ArgumentParser()
    live_parser.add_argument("--num_envs", type=int, default=args.num_envs)
    live_parser.add_argument("--steps", type=int, default=args.steps)
    live_parser.add_argument("--save_every", type=int, default=args.save_every)
    live_parser.add_argument("--out_dir", type=str, default=args.out_dir)
    live_parser.add_argument("--hang_timeout", type=float, default=150.0)
    AppLauncher.add_app_launcher_args(live_parser)
    live_args = live_parser.parse_args()
    live_args.enable_cameras = True

    # If the process spins (e.g. the Replicator orchestrator stalls), dump every thread's
    # Python stack and exit. This works without ptrace permissions, unlike py-spy / gdb.
    import faulthandler

    faulthandler.enable()
    if float(live_args.hang_timeout) > 0:
        faulthandler.dump_traceback_later(float(live_args.hang_timeout), exit=True)
        print(f"[cam] watchdog armed: stack dump + exit after {live_args.hang_timeout:.0f}s", flush=True)

    app_launcher = AppLauncher(live_args)
    simulation_app = app_launcher.app
    print("[phase] AppLauncher done", flush=True)

    import gymnasium as gym
    from pxr import Usd, UsdGeom

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.math import (
        convert_camera_frame_orientation_convention,
        matrix_from_quat,
        quat_apply,
    )
    from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_cam_env import AUVTrackingCamEnvCfg

    def save_png(rgb_nhw3: torch.Tensor, path: str) -> None:
        """Write the exact uint8 pixels (no rescaling) so visibility can be judged by eye."""
        from PIL import Image

        arr = rgb_nhw3.detach().cpu().numpy()
        if arr.shape[0] == 1:
            Image.fromarray(arr[0]).save(path)
            return
        strip = np.concatenate([arr[i] for i in range(arr.shape[0])], axis=1)
        Image.fromarray(strip).save(path)

    print("[phase] imports done", flush=True)
    cfg = AUVTrackingCamEnvCfg()
    cfg.scene.num_envs = int(live_args.num_envs)
    # Depth is a debug-only addition: the decisive check that the renderer uses the live
    # camera pose rather than the frozen USD one. The env itself only reads "rgb".
    cfg.tiled_camera.data_types = ["rgb", "distance_to_image_plane"]
    print("[phase] building env (scene + camera + first reset)...", flush=True)
    env = gym.make("Isaac-AUV-Tracking-Direct-v3-cam", cfg=cfg)
    raw = env.unwrapped
    print("[phase] env created", flush=True)

    print(f"[cam] kf_measurement={raw.cfg.kf_measurement}  sonar={'yes' if raw.pipeline else 'no'}", flush=True)
    print(
        f"[cam] prim={cfg.tiled_camera.prim_path}  offset={cfg.tiled_camera.offset.pos}"
        f"  convention={cfg.tiled_camera.offset.convention}  {cfg.cam_width}x{cfg.cam_height}"
        f"  hfov={cfg.cam_hfov_deg}  K={cfg.frame_stack}",
        flush=True,
    )

    out_dir = Path(live_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = raw.sim.stage
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    # The WarpAUV geometry never moves, so read its AABB once at the authored spawn pose and
    # convert it into body-frame half-extents. We must NOT read world bounds per step: with
    # replicate_physics=True the USD xforms are frozen at the spawn pose.
    _tgt_prim = stage.GetPrimAtPath("/World/envs/env_0/Target")
    _lr = bbox_cache.ComputeLocalBound(_tgt_prim).ComputeAlignedRange()
    _lo = np.array(_lr.GetMin(), dtype=np.float64)
    _hi = np.array(_lr.GetMax(), dtype=np.float64)
    # The bound is expressed in the frame the prim was authored in, which is the spawn pose.
    _spawn = np.array(cfg.target_cfg.init_state.pos, dtype=np.float64)
    TGT_HALF = 0.5 * (_hi - _lo)
    TGT_CENTER_OFF = 0.5 * (_lo + _hi) - _spawn
    print(
        f"[cam] target body half-extents={np.round(TGT_HALF, 4).tolist()}"
        f"  (size={np.round(2 * TGT_HALF, 4).tolist()})"
        f"  centre offset from origin={np.round(TGT_CENTER_OFF, 4).tolist()}",
        flush=True,
    )

    _cam_off = torch.tensor(cfg.tiled_camera.offset.pos, dtype=torch.float32)

    def cam_pose(env_i: int = 0):
        """Camera world pose derived from the *simulated* robot pose + mount offset.

        `TiledCamera.data.pos_w` / `quat_w_ros` cannot be used: they stay at the spawn pose
        because XFormPrim reads USD, which PhysX does not update under replicate_physics.
        convention="world" with an identity offset rotation means the camera axes coincide
        with the body axes.
        """
        rb = raw._robot.data.root_pos_w[env_i].detach().cpu()
        rq = raw._robot.data.root_quat_w[env_i].detach().cpu()
        pos = rb + quat_apply(rq.unsqueeze(0), _cam_off.unsqueeze(0)).squeeze(0)
        quat_ros = convert_camera_frame_orientation_convention(
            rq.unsqueeze(0), origin="world", target="ros"
        ).squeeze(0)
        return pos.numpy().astype(np.float64), quat_ros.numpy().astype(np.float64)

    def target_pixel_stats(env_i: int = 0):
        """Project the simulated target AUV -> pixel centre / bbox / width."""
        tgt_pos = raw._target.data.root_pos_w[env_i].detach().cpu()
        tgt_quat = raw._target.data.root_quat_w[env_i].detach().cpu()
        signs = np.array(
            [[1 if (c & 1) else -1, 1 if (c & 2) else -1, 1 if (c & 4) else -1] for c in range(8)],
            dtype=np.float64,
        )
        corners_body = TGT_CENTER_OFF[None, :] + signs * TGT_HALF[None, :]
        corners_w = quat_apply(
            tgt_quat.unsqueeze(0), torch.as_tensor(corners_body, dtype=torch.float32)
        ).numpy().astype(np.float64) + tgt_pos.numpy()[None, :]

        K = raw._tiled_camera.data.intrinsic_matrices[env_i].detach().cpu().numpy().astype(np.float64)
        pos_w, quat_ros = cam_pose(env_i)
        R = matrix_from_quat(torch.as_tensor(quat_ros).unsqueeze(0)).squeeze(0).numpy().astype(np.float64)

        p_cam = (corners_w - pos_w[None, :]) @ R  # == R^T @ (p - pos_w)
        z = p_cam[:, 2]
        in_front = z > 1e-6
        u = np.full(len(corners_w), np.nan)
        v = np.full(len(corners_w), np.nan)
        u[in_front] = K[0, 0] * p_cam[in_front, 0] / z[in_front] + K[0, 2]
        v[in_front] = K[1, 1] * p_cam[in_front, 1] / z[in_front] + K[1, 2]
        range_m = float(np.linalg.norm(tgt_pos.numpy() - pos_w))
        return u, v, int(in_front.sum()), range_m

    def true_bearing_range(env_i: int = 0):
        robot_p = raw._robot.data.root_pos_w[env_i, :2].detach().cpu().numpy()
        robot_q = raw._robot.data.root_quat_w[env_i].detach().cpu().numpy()
        tgt_p = raw._target.data.root_pos_w[env_i, :2].detach().cpu().numpy()
        w, x, y, z = robot_q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        diff = tgt_p - robot_p
        c, s = np.cos(-yaw), np.sin(-yaw)
        dx = c * diff[0] - s * diff[1]
        dy = s * diff[0] + c * diff[1]
        return float(np.hypot(dx, dy)), float(np.degrees(np.arctan2(dy, dx)))

    def depth_at(u: float, v: float, env_i: int = 0) -> float:
        """Rendered depth (m) at the given pixel, or NaN if unavailable."""
        d = raw._tiled_camera.data.output.get("distance_to_image_plane")
        if d is None:
            return float("nan")
        d = d[env_i]
        d = d[..., 0] if d.ndim == 3 else d
        d = d.detach().cpu().numpy().astype(np.float64)
        iu = int(round(min(max(u, 0), cfg.cam_width - 1)))
        iv = int(round(min(max(v, 0), cfg.cam_height - 1)))
        return float(d[iv, iu])

    actions = torch.zeros((raw.num_envs, raw.action_space.shape[-1]), device=raw.device)
    print("[phase] calling env.reset()...", flush=True)
    obs = env.reset()
    print("[phase] reset done", flush=True)
    if not isinstance(obs, dict):
        obs = {"policy": obs}

    ok = True
    n_inside = 0
    n_sampled = 0
    for step in range(int(live_args.steps)):
        print(f"[phase] step {step}...", flush=True)
        # Small non-zero action so the vehicle actually moves (reveals stale-frame bugs).
        a = actions.clone()
        a[:, 0] = 0.3
        obs, _rew, _term, _trunc, _info = env.step(a)
        if not isinstance(obs, dict):
            obs = {"policy": obs}
        # NOTE: env._get_observations returns a *view* into the internal frame ring buffer,
        # which keeps mutating in place. Clone so comparisons below are meaningful.
        policy = obs["policy"].clone()
        if step == 0:
            shape, dtype = tuple(policy.shape), policy.dtype
            print(f"[cam] obs shape={shape} dtype={dtype} (expect (N, {cfg.frame_stack * 3}, "
                  f"{cfg.cam_height}, {cfg.cam_width}))", flush=True)
            if shape[1:] != (cfg.frame_stack * 3, cfg.cam_height, cfg.cam_width) or dtype != torch.uint8:
                ok = False
                print("[cam] FAIL: observation shape/dtype mismatch", flush=True)

        if step % max(int(live_args.save_every), 1) == 0:
            rgb = raw._tiled_camera.data.output["rgb"]
            if rgb.dtype != torch.uint8:
                rgb = rgb.to(torch.uint8)
            save_png(rgb[..., :3].contiguous(), str(out_dir / f"frame_step{step:03d}.png"))

        u, v, n_front, cam_range = target_pixel_stats(0)
        r_m, bearing_deg = true_bearing_range(0)
        exp_px = _px_per_m(cfg, r_m)
        rb = raw._robot.data.root_pos_w[0, :3].detach().cpu().numpy()
        print(
            f"[phys {step:02d}] t={int(raw.episode_length_buf[0])}"
            f"  act=[{float(raw._actions[0, 0]):+.2f},{float(raw._actions[0, 1]):+.2f}]"
            f"  robot=({rb[0]:7.3f},{rb[1]:7.3f},{rb[2]:6.3f})"
            f"  r_robot2tgt={r_m:5.2f} m  bearing={bearing_deg:+7.2f} deg"
            f"  r_cam2tgt={cam_range:5.2f} m  frame_mean={float(policy[0].float().mean()):6.2f}",
            flush=True,
        )
        finite = np.isfinite(u) & np.isfinite(v)
        if not finite.any():
            ok = False
            print(f"[step {step:02d}] FAIL: whole target behind the camera plane", flush=True)
            continue
        u0, u1 = float(u[finite].min()), float(u[finite].max())
        v0, v1 = float(v[finite].min()), float(v[finite].max())
        cx_img, cy_img = 0.5 * (cfg.cam_width - 1), 0.5 * (cfg.cam_height - 1)
        n_sampled += 1
        inside = (u1 > 0) and (u0 < cfg.cam_width - 1) and (v1 > 0) and (v0 < cfg.cam_height - 1)
        n_inside += int(inside)
        # Depth at the geometric target centre: this is what ties the rendered image to the
        # derived camera pose.
        du = 0.5 * (u0 + u1)
        dv = 0.5 * (v0 + v1)
        d_target = depth_at(du, dv, 0)
        d_img = raw._tiled_camera.data.output["distance_to_image_plane"][0]
        d_img = (d_img[..., 0] if d_img.ndim == 3 else d_img).detach().cpu().numpy()
        print(
            f"[step {step:02d}] proj centre=({du:6.1f},{dv:6.1f}) bbox w={u1-u0:5.1f} h={v1-v0:5.1f} px"
            f"  img={cfg.cam_width}x{cfg.cam_height} centre=({cx_img:.1f},{cy_img:.1f})"
            f"  expect_w~{exp_px*0.40:.1f} px  inside={inside}"
            f"  | depth: at_target={d_target:6.2f} m  img_min={d_img.min():6.2f}"
            f"  img_p5={np.percentile(d_img, 5):6.2f}  img_med={np.median(d_img):6.2f}",
            flush=True,
        )
        # Decisive: the rendered depth at the target's geometric pixel must match the
        # geometric camera-to-target range. A frozen render pose would differ by metres.
        if np.isfinite(d_target) and not (0.5 * cam_range <= d_target <= 1.6 * cam_range):
            ok = False
            print(
                f"[step {step:02d}] FAIL: rendered depth at target pixel ({d_target:.2f} m) does not"
                f" match geometric camera-to-target range ({cam_range:.2f} m) -> the renderer is"
                f" probably using the frozen USD camera pose",
                flush=True,
            )

    if n_inside < n_sampled:
        ok = False
        print(f"[cam] FAIL: target inside image in only {n_inside}/{n_sampled} sampled steps", flush=True)
    else:
        print(f"[cam] target inside image in {n_inside}/{n_sampled} sampled steps", flush=True)

    # Frame-stack integrity (use the cloned obs from the last step, NOT a live view):
    # right after a reset all K frames must be identical, and after stepping with a
    # moving scene the oldest and newest frames must differ.
    raw._reset_idx(None)
    filled_after_reset = raw._get_observations()["policy"].clone()
    frames_reset = filled_after_reset.reshape(raw.num_envs, cfg.frame_stack, 3, cfg.cam_height, cfg.cam_width)
    same_after_reset = bool((frames_reset[0, 0] == frames_reset[0, -1]).all())
    shifted = bool((policy[0, :3] != policy[0, -3:]).any())
    print(
        f"[cam] stack: identical_across_K_after_reset={same_after_reset} (expect True),"
        f" oldest_vs_newest_differ_after_{live_args.steps}_steps={shifted} (expect True)",
        flush=True,
    )
    if not same_after_reset or not shifted:
        ok = False

    nz = float((policy.to(torch.float32) > 0).float().mean())
    print(f"[cam] obs non-zero fraction={nz:.4f} (expect > 0.05; ~0 means a black frame)", flush=True)
    if nz <= 0.05:
        ok = False
    print(f"[cam] PNGs in {out_dir}", flush=True)
    print(f"[cam] RESULT: {'PASS' if ok else 'FAIL'}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
