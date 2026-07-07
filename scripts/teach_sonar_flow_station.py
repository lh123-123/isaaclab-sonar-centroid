#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Teach phase: record sonar image + sparse IBVS feature points at station pose."""

from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Capture taught sonar + sparse features at station pose.")
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Lateral1D-Debug-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--sonar_ui",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Open OceanSim sonar viewport (default: on).",
)
parser.add_argument(
    "--output",
    type=str,
    default="logs/station_ref/station_ref.npz",
    help="Path to save reference snapshot (.npz).",
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=80,
    help="Max zero-action steps while waiting for stable sonar before capture.",
)
parser.add_argument(
    "--min_feature_streak",
    type=int,
    default=10,
    help="Require this many consecutive steps with >= min_features before capture.",
)
parser.add_argument(
    "--min_features",
    type=int,
    default=5,
    help="Minimum tracked features to consider sonar stable.",
)
parser.add_argument(
    "--offset_dx",
    type=float,
    default=0.0,
    help="Optional world X offset from hover for teach pose (debug only).",
)
parser.add_argument(
    "--offset_dy",
    type=float,
    default=0.0,
    help="Optional world Y offset from hover for teach pose (debug only).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _maybe_save_png(env, path: str, *, image_attr: str = "cart_image") -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    tensor = getattr(env, image_attr, None)
    if tensor is None:
        return
    img = tensor[0].detach().cpu().numpy()
    if img.size == 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = img
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = (arr * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.freeze_robot_pose = True
    env_cfg.eval_spawn_radius = 0.0
    env_cfg.spawn_radius = 0.0
    env_cfg.reference_mode = "capture"
    env_cfg.print_ibvs_debug = True
    env_cfg.ibvs_debug_interval = 10
    if args_cli.sonar_ui and hasattr(env_cfg, "oceansim_sonar_cfg"):
        env_cfg.oceansim_sonar_cfg.sonar_initialize_viewport = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    dev = unwrapped.device
    zero = torch.zeros(env.action_space.shape, device=dev)
    dt = getattr(env, "step_dt", unwrapped.step_dt)

    print("\n======== Sonar IBVS-DRL Teach ========")
    print(f"  task              : {args_cli.task}")
    print(f"  sonar_ui          : {args_cli.sonar_ui}")
    print(f"  output            : {args_cli.output}")
    print(f"  settle_steps      : {args_cli.settle_steps}")
    print(f"  min_feature_streak: {args_cli.min_feature_streak}")
    print(f"  min_features      : {args_cli.min_features}")
    print(f"  offset            : dx={args_cli.offset_dx:+.2f} dy={args_cli.offset_dy:+.2f}")
    print("  Robot frozen; confirm sonar target in viewport, then capture runs automatically.")
    print("=======================================\n")

    env.reset()
    if args_cli.offset_dx != 0.0 or args_cli.offset_dy != 0.0:
        unwrapped.set_displacement_offset(args_cli.offset_dx, args_cli.offset_dy)

    unwrapped._ensure_pipeline()
    valid_n = int(unwrapped.feature_tracker.valid_count()[0].item())
    if valid_n >= args_cli.min_features:
        print(f"[teach] features ready after reset ({valid_n}); capturing.", flush=True)
        snapshot = unwrapped.capture_reference_snapshot(args_cli.output)
        base = os.path.splitext(args_cli.output)[0]
        frame = str(getattr(unwrapped.cfg, "perception_frame", "cart"))
        track_png = f"{base}_track.png"
        _maybe_save_png(unwrapped, track_png, image_attr="track_image")
        if frame == "cart":
            _maybe_save_png(unwrapped, f"{base}_cart.png", image_attr="cart_image")
        print("\n========== Teach Summary ==========")
        print(f"  snapshot   : {os.path.abspath(args_cli.output)}")
        print(f"  track preview: {os.path.abspath(track_png)}")
        print(f"  n_features : {len(snapshot.ref_points)}")
        print("  Next:")
        print("    train: Isaac-SonarIbvsDrl-Stage1-Direct-v0")
        print("    eval : Isaac-SonarIbvsDrl-Eval-Direct-v0")
        print("===================================\n")
        env.close()
        simulation_app.close()
        return

    feature_streak = 0
    captured = False

    for step in range(args_cli.settle_steps):
        t0 = time.time()
        env.step(zero)
        unwrapped._ensure_pipeline()
        valid_n = int(unwrapped.feature_tracker.valid_count()[0].item())

        if valid_n >= args_cli.min_features:
            feature_streak += 1
        else:
            feature_streak = 0

        if feature_streak >= args_cli.min_feature_streak:
            captured = True
            print(f"[teach] stable features ({valid_n}) at step={step}; capturing.", flush=True)
            break

        if step % 10 == 0:
            print(
                f"[teach] step={step} valid_features={valid_n} streak={feature_streak}/"
                f"{args_cli.min_feature_streak}",
                flush=True,
            )

        sleep_t = dt - (time.time() - t0)
        if sleep_t > 0 and simulation_app.is_running():
            time.sleep(sleep_t)

    if not captured:
        print("[teach] WARN: feature stability not reached; capturing anyway.", flush=True)

    snapshot = unwrapped.capture_reference_snapshot(args_cli.output)
    base = os.path.splitext(args_cli.output)[0]
    frame = str(getattr(unwrapped.cfg, "perception_frame", "cart"))
    track_png = f"{base}_track.png"
    _maybe_save_png(unwrapped, track_png, image_attr="track_image")
    if frame == "cart":
        _maybe_save_png(unwrapped, f"{base}_cart.png", image_attr="cart_image")

    print("\n========== Teach Summary ==========")
    print(f"  snapshot   : {os.path.abspath(args_cli.output)}")
    print(f"  track preview: {os.path.abspath(track_png)}")
    print(f"  n_features : {len(snapshot.ref_points)}")
    print("  Next:")
    print("    train: Isaac-SonarIbvsDrl-Stage1-Direct-v0")
    print("    eval : Isaac-SonarIbvsDrl-Eval-Direct-v0")
    print("===================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
