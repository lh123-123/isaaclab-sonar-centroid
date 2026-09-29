# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Identify which pose the RTX renderer actually uses for the Direct-v3-cam camera.

Motivation
----------
``debug_auv_cam_frames.py`` proved the rendered image contains no target AUV even when the
target is ~1 m in front of the camera and would cover most of the frame. Two facts point at
USD transforms that never move:

* The WarpAUV asset is authored at ``init_state.pos = (0, 0, 5)`` (``WARPAUV_CFG``), while the
  robot is authored at ``(0, 0, 8)``. Live poses are written into PhysX, not into USD.
* ``TiledCamera.data.pos_w`` stays frozen at the spawn pose even with
  ``update_latest_camera_pose=True``.

So if the renderer reads frozen USD instead of Fabric, the camera renders at
``(0.35, 0, 8.05)`` and the target renders at ``(0, 0, 5)`` -- i.e. 0.35 m *behind* the camera
and 3 m below it, hence invisible.

This probe distinguishes the cases by teleporting prims to unambiguous places and measuring
cheap image signatures:

* ``band_row`` -- first image row whose mean luminance drops below a threshold. The seafloor is
  8 m below and the far clip is 30 m, so the floor only becomes visible where its distance is
  < 30 m. That row is therefore a direct read-out of the *render-time* camera height:
  z=8 -> row ~53, z=2 -> row ~37. Teleporting the robot to z=2 and watching this row is a
  decisive test of whether the render camera follows the robot.
* ``dark%`` / ``near_px`` -- fraction of dark pixels and count of pixels closer than 10 m. A
  visible 2 m-away AUV makes both jump.

Usage::

    ./scripts/debug_auv_cam_probe.sh --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _row_band(lum: np.ndarray, thresh: float = 150.0) -> int:
    """First image row whose mean luminance falls below ``thresh`` (the seafloor line)."""
    rows = lum.mean(axis=1)
    below = np.nonzero(rows < thresh)[0]
    return int(below[0]) if below.size else -1


def _summarize(rgb: np.ndarray, depth: np.ndarray | None) -> dict:
    lum = rgb.astype(np.float32).mean(axis=2)
    out = {
        "mean": float(lum.mean()),
        "std": float(lum.std()),
        "dark_pct": 100.0 * float((lum < 100).mean()),
        "band_row": _row_band(lum),
    }
    if depth is not None and depth.size:
        d = depth.astype(np.float32)
        finite = np.isfinite(d)
        out["d_min"] = float(d[finite].min()) if finite.any() else float("inf")
        out["d_inf_pct"] = 100.0 * float((~finite).mean())
        out["near_px"] = int((d < 10.0).sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the renderer's effective poses.")
    parser.add_argument("--out_dir", type=str, default="logs/auv_tracking/v3_cam_probe")
    args, unknown = parser.parse_known_args()

    import torch

    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]] + unknown
    live_parser = argparse.ArgumentParser()
    live_parser.add_argument("--out_dir", type=str, default=args.out_dir)
    live_parser.add_argument("--hang_timeout", type=float, default=150.0)
    AppLauncher.add_app_launcher_args(live_parser)
    live_args = live_parser.parse_args()
    live_args.enable_cameras = True

    import faulthandler

    faulthandler.enable()
    if float(live_args.hang_timeout) > 0:
        faulthandler.dump_traceback_later(float(live_args.hang_timeout), exit=True)

    app_launcher = AppLauncher(live_args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from PIL import Image

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_cam_env import AUVTrackingCamEnvCfg

    cfg = AUVTrackingCamEnvCfg()
    cfg.scene.num_envs = 1
    cfg.tiled_camera.data_types = ["rgb", "distance_to_image_plane"]
    env = gym.make("Isaac-AUV-Tracking-Direct-v3-cam", cfg=cfg)
    raw = env.unwrapped

    dev = raw.device
    e0 = torch.tensor([0], device=dev, dtype=torch.long)

    # Reset BEFORE patching: the initial reset places the robot/target via _sync_target_entity,
    # which we then disable so our teleports stick.
    print("[probe] resetting...", flush=True)
    env.reset()
    print("[probe] reset done", flush=True)

    # Freeze episode control: never reset, and stop the env from re-writing the target pose so
    # our teleports stick.
    raw._get_dones = lambda: (
        torch.zeros(raw.num_envs, dtype=torch.bool, device=dev),
        torch.zeros(raw.num_envs, dtype=torch.bool, device=dev),
    )
    raw._sync_target_entity = lambda *a, **k: None

    out_dir = Path(live_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def set_pose(entity, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        pose = torch.tensor([[*pos, *quat]], device=dev, dtype=torch.float32)
        vel = torch.zeros(1, 6, device=dev, dtype=torch.float32)
        entity.write_root_pose_to_sim(pose, e0)
        entity.write_root_velocity_to_sim(vel, e0)

    def settle_and_measure(tag: str, robot_pos=None, target_pos=None, n_steps: int = 3) -> dict:
        """Step a few times so the render products refresh, then measure the frame.

        The teleported poses are re-applied every step: otherwise gravity/thrust drift over the
        steps would blur the comparison we are trying to make.
        """
        zero = torch.zeros((raw.num_envs, raw.action_space.shape[-1]), device=dev)
        for _ in range(n_steps):
            if robot_pos is not None:
                set_pose(raw._robot, robot_pos)
            if target_pos is not None:
                set_pose(raw._target, target_pos)
            env.step(zero)
        rgb = raw._tiled_camera.data.output["rgb"]
        rgb = (rgb if rgb.dtype == torch.uint8 else rgb.to(torch.uint8))[0, ..., :3]
        arr = rgb.detach().cpu().numpy()
        Image.fromarray(arr).save(str(out_dir / f"{tag}.png"))

        d = raw._tiled_camera.data.output.get("distance_to_image_plane")
        depth = None
        if d is not None:
            d = d[0]
            depth = (d[..., 0] if d.ndim == 3 else d).detach().cpu().numpy()
        s = _summarize(arr, depth)
        rb = raw._robot.data.root_pos_w[0, :3].detach().cpu().numpy()
        tb = raw._target.data.root_pos_w[0, :3].detach().cpu().numpy()
        s["tag"] = tag
        s["robot"] = f"({rb[0]:6.2f},{rb[1]:6.2f},{rb[2]:6.2f})"
        s["target"] = f"({tb[0]:6.2f},{tb[1]:6.2f},{tb[2]:6.2f})"
        print(
            f"[probe] {tag:24s} robot={s['robot']} target={s['target']}"
            f"  mean={s['mean']:6.1f} std={s['std']:6.1f} dark={s['dark_pct']:5.1f}%"
            f"  band_row={s['band_row']:3d}"
            + (
                f"  d_min={s['d_min']:6.2f} d_inf={s['d_inf_pct']:5.1f}% near_px={s['near_px']}"
                if "d_min" in s
                else ""
            ),
            flush=True,
        )
        return s

    def teleport(robot_pos=None, target_pos=None, n: int = 3) -> None:
        """Move the prims and let the renderer catch up, re-applying the pose every step."""
        zero = torch.zeros((raw.num_envs, raw.action_space.shape[-1]), device=dev)
        for _ in range(n):
            if robot_pos is not None:
                set_pose(raw._robot, robot_pos)
            if target_pos is not None:
                set_pose(raw._target, target_pos)
            env.step(zero)

    print("\n[probe] === baseline (as-reset) ===", flush=True)
    results = {}
    results["base"] = settle_and_measure("base")

    print(
        "\n[probe] === A: is the RENDER camera live? move robot z 8 -> 2 ==="
        "\n[probe]     band_row should move 53 -> ~37 if the render camera follows the robot",
        flush=True,
    )
    results["A_robot_z2"] = settle_and_measure("A_robot_at_z2", robot_pos=(0.0, 0.0, 2.0))
    results["A_back"] = settle_and_measure("A_robot_back_z8", robot_pos=(0.0, 0.0, 8.0))

    print(
        "\n[probe] === B: robot at authored spawn (0,0,8); target 1.5 m dead ahead ==="
        "\n[probe]     camera would be (0.35,0,8.05) either way, so the target MUST be visible"
        "\n[probe]     if the target's render pose is live. Watch dark% / near_px.",
        flush=True,
    )
    results["B_tgt_ahead"] = settle_and_measure(
        "B_target_ahead_live", robot_pos=(0.0, 0.0, 8.0), target_pos=(1.85, 0.0, 8.0)
    )

    print(
        "\n[probe] === C: target at its USD authored spawn (0,0,5) ==="
        "\n[probe]     visible only if the renderer uses the frozen authored pose.",
        flush=True,
    )
    results["C_tgt_authored"] = settle_and_measure(
        "C_target_at_authored_spawn", robot_pos=(0.0, 0.0, 8.0), target_pos=(0.0, 0.0, 5.0)
    )

    print("\n[probe] === summary ===", flush=True)
    base = results["base"]
    live_cam = abs(results["A_robot_z2"]["band_row"] - base["band_row"]) > 5
    print(
        f"[probe] A render camera follows the robot: {live_cam}"
        f"  (band_row {base['band_row']} -> {results['A_robot_z2']['band_row']};"
        f" unchanged means the renderer uses the frozen USD camera pose)",
        flush=True,
    )
    # Judge visibility by depth, not by dark%: the target AUV is dark but the surrounding water
    # is bright, so its pixel-share barely moves the global mean luminance. `near_px` (pixels
    # closer than 10 m) and `d_min` change by orders of magnitude instead.
    def _visible(tag: str, expect_range: float) -> bool:
        s = results[tag]
        return s.get("near_px", 0) > 50 and s.get("d_min", float("inf")) < expect_range

    vis_b = _visible("B_tgt_ahead", 2.0)
    vis_c = _visible("C_tgt_authored", 20.0)
    print(
        f"[probe] B live target visible 1.5 m ahead: {vis_b}"
        f"  (near_px {base['near_px']} -> {results['B_tgt_ahead']['near_px']},"
        f" d_min {base['d_min']:.2f} -> {results['B_tgt_ahead']['d_min']:.2f})",
        flush=True,
    )
    print(
        f"[probe] C target visible at the frozen authored spawn (0,0,5): {vis_c}"
        f"  (near_px {base['near_px']} -> {results['C_tgt_authored']['near_px']},"
        f" d_min {base['d_min']:.2f} -> {results['C_tgt_authored']['d_min']:.2f};"
        f" True would mean the renderer still uses authored poses)",
        flush=True,
    )
    print(f"[probe] PNGs in {out_dir}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    import os
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        # SimulationApp.shutdown() can block for minutes after a mid-run exception, which
        # wastes a whole GPU slot. Exit hard instead so failures surface immediately.
        os._exit(1)
    os._exit(0)
