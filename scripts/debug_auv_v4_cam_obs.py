# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Dump what Direct-v4-cam's ACTOR actually receives, as inspectable PNGs.

The v4-cam actor input is a dict:

    image       (N, 12, 64, 64) uint8   = 4 stacked RGB frames (frame_stack=4 x 3 ch)
    flow_image  (N,  2, 64, 64) float32 = 1 GT flow raster (du/W, dv/H) of the target pixel motion
    state       (N,  4)         float32 = KF belief [r, alpha, log det Sigma, observed]  (critic only)

so the CNN's first conv sees 12 + 2 = 14 channels. This script renders each of those pieces
separately (no rescaling of the RGB pixels) plus a contact sheet, and writes an npz of the raw
arrays, so the input can be judged by eye instead of inferred from the encoder code.

  ./scripts/debug_auv_v4_cam_obs.sh --headless --num_envs 1 --steps 8 --save_last 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _to_u8(a: np.ndarray) -> np.ndarray:
    """float array in [0, 1] -> uint8, no cleverness."""
    return np.clip(a * 255.0, 0.0, 255.0).round().astype(np.uint8)


def _autoscale(a: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale an almost-all-zero map so its own maximum becomes 1.0. Returns (map, gain)."""
    m = float(np.abs(a).max())
    if m <= 0.0:
        return np.zeros_like(a), 0.0
    return a / m, 1.0 / m


def _up(arr: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour upscale so a 64x64 panel is actually legible."""
    if scale <= 1:
        return arr
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump Direct-v4-cam actor observation as PNGs.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--save_last", type=int, default=3, help="How many trailing steps to render.")
    parser.add_argument("--action_v", type=float, default=0.3, help="Non-zero surge so the scene moves.")
    parser.add_argument("--out_dir", type=str, default="logs/auv_tracking/v4_cam_obs_dump")
    args, unknown = parser.parse_known_args()

    import sys

    import torch

    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]] + unknown
    live_parser = argparse.ArgumentParser()
    live_parser.add_argument("--num_envs", type=int, default=args.num_envs)
    live_parser.add_argument("--steps", type=int, default=args.steps)
    live_parser.add_argument("--save_last", type=int, default=args.save_last)
    live_parser.add_argument("--action_v", type=float, default=args.action_v)
    live_parser.add_argument("--out_dir", type=str, default=args.out_dir)
    AppLauncher.add_app_launcher_args(live_parser)
    live_args = live_parser.parse_args()
    live_args.enable_cameras = True

    app_launcher = AppLauncher(live_args)
    simulation_app = app_launcher.app
    print("[phase] AppLauncher done", flush=True)

    import gymnasium as gym
    from PIL import Image

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v4_cam_env import (
        AUVTrackingV4CamEnvCfg,
    )

    print("[phase] imports done", flush=True)
    cfg = AUVTrackingV4CamEnvCfg()
    cfg.scene.num_envs = int(live_args.num_envs)
    env = gym.make("Isaac-AUV-Tracking-Direct-v4-cam", cfg=cfg)
    raw = env.unwrapped
    print(
        f"[v4obs] K={cfg.frame_stack} obs={cfg.cam_width}x{cfg.cam_height}"
        f"  hfov={cfg.cam_hfov_deg}  render="
        f"{getattr(cfg, 'render_width', cfg.cam_width)}x{getattr(cfg, 'render_height', cfg.cam_height)}",
        flush=True,
    )

    # H6: the GT flow disk is stamped using cfg.cam_hfov_deg, so the rendered camera's real
    # HFOV must match it, or the disk lands somewhere the target is not.
    try:
        import math as _m

        _K = raw._tiled_camera.data.intrinsic_matrices[0].detach().cpu().numpy()
        _rgb_shape = raw._tiled_camera.data.output["rgb"].shape
        _rw, _rh = int(_rgb_shape[2]), int(_rgb_shape[1])
        _fx, _fy = float(_K[0, 0]), float(_K[1, 1])
        _hfov = 2.0 * _m.degrees(_m.atan(_rw / (2.0 * _fx)))
        _vfov = 2.0 * _m.degrees(_m.atan(_rh / (2.0 * _fy)))
        _fx_obs = _fx * float(cfg.cam_width) / _rw  # same lens, expressed in obs pixels
        print(
            f"[v4obs] camera intrinsics: fx_render={_fx:.2f} fy_render={_fy:.2f} on {_rw}x{_rh}"
            f"  -> actual HFOV={_hfov:.2f} deg vFOV={_vfov:.2f} deg"
            f"  | cfg.cam_hfov_deg={cfg.cam_hfov_deg}"
            f"  | fx in obs px={_fx_obs:.2f}"
            f"  | MATCH={abs(_hfov - float(cfg.cam_hfov_deg)) < 1.0}",
            flush=True,
        )
    except Exception as _exc:  # pragma: no cover - diagnostic only
        print(f"[v4obs] intrinsics probe failed: {_exc}", flush=True)

    # H7: the GT flow disk is stamped from the robot ROOT pose (`_agent_pose_2d` + root z), but
    # the RGB is rendered from the camera prim's actual (asset-authored) pose. Read that pose so
    # any systematic disk-vs-target offset can be attributed to the mount, not guessed at.
    try:
        from pxr import UsdGeom

        _cam_path = "/World/envs/env_0/Robot/warpauv_base_link/LookFollowCamera0"
        _prim = raw.sim.stage.GetPrimAtPath(_cam_path)
        _m = np.array(UsdGeom.Xformable(_prim).GetLocalTransformation(), dtype=np.float64)
        _t = _m[:3, 3]
        from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v4_cam_env import (
            CAM_MOUNT_OFFSET_LOCAL,
        )

        print(
            f"[v4obs] camera prim local transform vs warpauv_base_link:"
            f" translate={np.round(_t, 4).tolist()}"
            f"  | CAM_MOUNT_OFFSET_LOCAL(cfg, inert)={CAM_MOUNT_OFFSET_LOCAL}"
            f"  | same={np.allclose(_t, np.array(CAM_MOUNT_OFFSET_LOCAL), atol=1e-3)}",
            flush=True,
        )
        # Angular offset the ignored mount would produce at the current target range.
        try:
            _r = float(np.linalg.norm((raw._target_pos[0, :2] - raw._robot.data.root_pos_w[0, :2]).cpu()))
            _fxo = _fx * float(cfg.cam_width) / _rw
            _du = _fxo * float(_t[1]) / max(_r, 1e-3)
            _dv = -_fxo * float(_t[2]) / max(_r, 1e-3)
            print(
                f"[v4obs] mount-induced pixel offset at r={_r:.2f} m: du={_du:+.2f} px"
                f"  dv={_dv:+.2f} px  (positive dv = target renders LOWER than the disk)",
                flush=True,
            )
        except Exception as _e2:
            print(f"[v4obs] mount px-offset calc failed: {_e2}", flush=True)
    except Exception as _exc:  # pragma: no cover - diagnostic only
        print(f"[v4obs] camera mount probe failed: {_exc}", flush=True)

    def clone_policy(p):
        if isinstance(p, dict):
            return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in p.items()}
        return p.clone()

    def unwrap(obs):
        p = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
        return clone_policy(p)

    actions = torch.zeros((raw.num_envs, raw.action_space.shape[-1]), device=raw.device)
    actions[:, 0] = float(live_args.action_v)
    obs = env.reset()
    snaps: list[tuple[int, dict, dict]] = []
    n_steps = max(int(live_args.steps), 1)
    for step in range(n_steps):
        obs, _rew, _term, _trunc, _info = env.step(actions)
        pol = unwrap(obs)
        # Depth is the unbiased way to segment the target: it is the only thing inside the
        # ~4 m sonar/camera band, so `depth < 3 m` isolates it without relying on its tint.
        _dp = raw._tiled_camera.data.output.get("distance_to_image_plane")
        _depth = None
        if _dp is not None:
            _d = _dp[0].detach().cpu().float()
            if _d.dim() == 3:
                _d = _d[..., 0]
            _rh_, _rw_ = _d.shape
            _depth = _d.reshape(
                raw.cfg.cam_height, _rh_ // raw.cfg.cam_height,
                raw.cfg.cam_width, _rw_ // raw.cfg.cam_width,
            ).mean(dim=(1, 3)).numpy()
        snap = {
            "robot_xy": raw._robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy(),
            "tgt_xy": raw._target_pos[0, :2].detach().cpu().numpy().copy(),
            "gt_px": raw._gt_px[0].detach().cpu().numpy().copy(),
            "target_px_ok": bool(raw._gt_px_ok[0]),
            "stack_pending_reset": bool(raw._stack_pending_reset[0]),
            "depth": _depth,
        }
        snaps.append((step, pol, snap))
        img = pol["image"][0]
        fl = pol["flow_image"][0]
        st = pol["state"][0]
        print(
            f"[step {step:02d}] image={tuple(img.shape)}/{img.dtype}"
            f"  flow={tuple(fl.shape)}/{fl.dtype} range=[{float(fl.min()):+.4f},{float(fl.max()):+.4f}]"
            f"  state=[{float(st[0]):.2f},{float(st[1]):+.3f},{float(st[2]):.2f},{float(st[3]):.1f}]"
            f"  r={float(np.linalg.norm(snap['tgt_xy'] - snap['robot_xy'])):.2f}m",
            flush=True,
        )

    out_dir = Path(live_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k = int(cfg.frame_stack)
    n_save = min(int(live_args.save_last), len(snaps))
    upscale = 6
    for step, pol, snap in snaps[-n_save:]:
        img = pol["image"][0].detach().cpu().numpy()  # (12, 64, 64) uint8
        flow = pol["flow_image"][0].detach().cpu().numpy()  # (2, 64, 64) float32
        st = pol["state"][0].detach().cpu().numpy()
        h, w = img.shape[-2:]
        frames = img.reshape(k, 3, h, w)

        # Save the raw arrays FIRST so an assembly error below cannot lose the run.
        np.savez(
            out_dir / f"raw_obs_step{step:03d}.npz",
            image=img,
            flow_image=flow,
            state=st,
            gt_px=snap["gt_px"],
            robot_xy=snap["robot_xy"],
            tgt_xy=snap["tgt_xy"],
            depth=snap["depth"] if snap["depth"] is not None else np.zeros((h, w), np.float32),
        )

        # --- 1. the K stacked frames, exactly as the CNN receives them (no rescale) ---
        tiles = [np.transpose(frames[j], (1, 2, 0)) for j in range(k)]
        Image.fromarray(_up(np.concatenate(tiles, axis=1), upscale)).save(
            out_dir / f"K4_frames_step{step:03d}.png"
        )

        # --- 2. is one "RGB" frame actually colour, or 3 copies of a luma image? ---
        f0 = frames[0].astype(np.int16)
        diff_rg = np.abs(f0[0] - f0[1]).astype(np.float32)
        diff_gb = np.abs(f0[1] - f0[2]).astype(np.float32)
        # Autoscale each difference panel by its own max: a 1-LSB dither would be invisible
        # against the /255 scale but lights up here, so "real colour" vs "rounding noise" is
        # distinguishable by eye. Gains are printed below.
        rg_auto, rg_gain = _autoscale(diff_rg)
        gb_auto, gb_gain = _autoscale(diff_gb)
        ch_panels = [
            f0[0].astype(np.float32) / 255.0,
            f0[1].astype(np.float32) / 255.0,
            f0[2].astype(np.float32) / 255.0,
            rg_auto,
            gb_auto,
        ]
        Image.fromarray(_up(np.concatenate([_to_u8(p) for p in ch_panels], axis=1), upscale)).save(
            out_dir / f"frame0_channels_R_G_B_absRG_absGB_step{step:03d}.png"
        )

        # --- 3. the flow raster. "faithful" assumes the declared [-1, 1] box: it looks black
        #        precisely because the values are tiny. "autoscale" shows the disk's SHAPE. ---
        du, dv = flow[0], flow[1]
        du_auto, gain_u = _autoscale(du)
        dv_auto, gain_v = _autoscale(dv)
        flow_panels = [
            _to_u8((du + 1.0) * 0.5),
            _to_u8((du_auto + 1.0) * 0.5),
            _to_u8((dv + 1.0) * 0.5),
            _to_u8((dv_auto + 1.0) * 0.5),
            _to_u8((np.abs(flow).sum(axis=0) > 0).astype(np.float32)),
        ]
        Image.fromarray(_up(np.concatenate(flow_panels, axis=1), upscale)).save(
            out_dir / f"flow_du_faithful_du_auto_dv_faithful_dv_auto_mask_step{step:03d}.png"
        )

        # --- 4. ALIGNMENT: is the flow disk stamped where the target actually renders? The
        #        target is the only orange thing, so a warm-pixel mask finds it independently
        #        of the geometry that placed the disk. ---
        r_, g_, b_ = f0[0].astype(np.int32), f0[1].astype(np.int32), f0[2].astype(np.int32)
        warm = (r_ - b_ > 20) & (r_ - g_ > 8)
        n_warm = int(warm.sum())
        cy, cx = np.array(np.nonzero(warm)).mean(axis=1) if n_warm else (np.nan, np.nan)
        mask = np.abs(flow).sum(axis=0) > 0
        n_mask = int(mask.sum())
        my, mx = np.array(np.nonzero(mask)).mean(axis=1) if n_mask else (np.nan, np.nan)
        overlay = np.transpose(frames[0], (1, 2, 0)).copy()
        if n_warm:
            overlay[warm] = [0, 255, 0]
        Image.fromarray(_up(overlay, upscale)).save(out_dir / f"target_mask_overlay_step{step:03d}.png")
        pair = np.concatenate(
            [
                _to_u8((du_auto + 1.0) * 0.5)[..., None].repeat(3, axis=2),
                overlay,
            ],
            axis=1,
        )
        Image.fromarray(_up(pair, upscale)).save(out_dir / f"flow_disk_vs_target_step{step:03d}.png")

        # --- 5. contact sheet: the 14 conv channels, one row per group ---
        top = _to_u8(np.concatenate([np.clip(t / 255.0, 0, 1) for t in tiles], axis=1))
        bottom = _to_u8(
            np.concatenate(
                [
                    (du + 1.0) * 0.5,
                    (du_auto + 1.0) * 0.5,
                    (dv + 1.0) * 0.5,
                    (dv_auto + 1.0) * 0.5,
                ],
                axis=1,
            )
        )
        sheet = np.concatenate([top, np.stack([bottom] * 3, axis=-1)], axis=0)
        Image.fromarray(_up(sheet, upscale)).save(out_dir / f"contact_sheet_step{step:03d}.png")

        n_gt = int(np.isfinite(snap["gt_px"]).all())
        _dep = snap["depth"]
        _dline = ""
        if _dep is not None:
            _td = (_dep > 0.05) & (_dep < 3.0) & np.isfinite(_dep)
            _ntd = int(_td.sum())
            if _ntd:
                _ty, _tx = np.nonzero(_td)
                _dline = (
                    f"  depth<3m: n={_ntd} centroid=({_tx.mean():.1f},{_ty.mean():.1f})"
                    f" bbox_x=[{_tx.min()},{_tx.max()}] bbox_y=[{_ty.min()},{_ty.max()}]"
                    f"  disk->target dpx=({_tx.mean()-mx:+.1f},{_ty.mean()-my:+.1f})"
                )
            else:
                _dline = "  depth<3m: none"
        print(
            f"[dump step {step:03d}] gray_frac={float(((f0[0]==f0[1])&(f0[1]==f0[2])).mean()):.4f}"
            f"  |R-G| mean/max={diff_rg.mean():.3f}/{diff_rg.max():.0f}"
            f"  |G-B| mean/max={diff_gb.mean():.3f}/{diff_gb.max():.0f}"
            f"  px with |R-G|>2: {int((diff_rg > 2).sum())}"
            f"  flow_nz={n_mask}/{h * w} du_max={float(np.abs(du).max()):.5f} dv_max={float(np.abs(dv).max()):.5f}"
            f"  disk_centroid=({mx:.1f},{my:.1f}) gt_px=({snap['gt_px'][0]:.1f},{snap['gt_px'][1]:.1f}) ok={n_gt}"
            f"{_dline}",
            flush=True,
        )

    print(f"[v4obs] PNGs + npz in {out_dir}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # An uncaught error used to leave the Isaac Sim process (and its GPU memory) alive
        # because simulation_app.close() was never reached. Hard-exit so the next run is clean.
        import os
        import traceback

        traceback.print_exc()
        os._exit(1)
