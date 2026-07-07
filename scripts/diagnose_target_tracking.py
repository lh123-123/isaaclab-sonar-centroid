#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless: diagnose IBVS tracking vs target motion (discrete offsets + smooth circle)."""

from __future__ import annotations

import argparse
import math

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose sonar feature tracking under target motion.")
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Debug-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--smooth_steps", type=int, default=60)
parser.add_argument("--target_radius", type=float, default=0.15)
parser.add_argument("--target_period_s", type=float, default=30.0)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import euler_xyz_from_quat
from isaaclab_tasks.utils import parse_env_cfg

TARGET_OFFSETS = [
    (0.0, 0.0, "origin"),
    (0.15, 0.0, "+x0.15"),
    (0.0, 0.15, "+y0.15"),
    (-0.15, 0.0, "-x0.15"),
    (0.0, -0.15, "-y0.15"),
]


def _blob_centroid(cart: np.ndarray) -> tuple[float, float] | None:
    u8 = (np.clip(cart, 0.0, 1.0) * 255.0).astype(np.uint8)
    peak = float(u8.max())
    if peak <= 8.0:
        return None
    thresh = max(int(peak * 0.25), 8)
    mask = (u8 >= thresh).astype(np.uint8)
    if mask.sum() < 4:
        return None
    ys, xs = np.where(mask > 0)
    return float(xs.mean()), float(ys.mean())


def _metrics(unwrapped, env_id: int = 0) -> dict:
    m = unwrapped.get_ibvs_metrics(env_id=env_id)
    cart = unwrapped.cart_image[env_id].detach().cpu().numpy()
    centroid = _blob_centroid(cart)
    _, _, yaw = euler_xyz_from_quat(unwrapped.robot.data.root_quat_w[env_id : env_id + 1])
    taught = int((torch.linalg.norm(unwrapped.feature_tracker.ref_points[env_id], dim=1) > 1.0e-3).sum().item())
    return {
        "valid": int(m["valid_pts"]),
        "taught": taught,
        "pixel_err": float(m["pixel_err"]),
        "yaw_deg": float(torch.rad2deg(yaw[0]).item()),
        "centroid": centroid,
    }


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.reference_mode = "file"
    env_cfg.freeze_robot_pose = True
    env_cfg.print_ibvs_debug = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    zero = torch.zeros(env.action_space.shape, device=unwrapped.device)
    dt = float(getattr(env, "step_dt", unwrapped.step_dt))
    omega = 2.0 * math.pi / max(args_cli.target_period_s, 1.0e-3)

    env.reset()
    unwrapped.set_displacement_offset(0.0, 0.0)
    unwrapped.set_target_velocity(0.0, 0.0)

    print("\n=== Discrete target offsets (AUV pose+yaw locked) ===")
    print(f"{'tag':<10} {'valid':>8} {'yaw°':>7} {'centroid':>16} {'pixel_err':>10}")
    origin_centroid = None
    for dx, dy, tag in TARGET_OFFSETS:
        unwrapped.set_target_offset(dx, dy)
        for _ in range(5):
            env.step(zero)
        unwrapped._ensure_pipeline()
        row = _metrics(unwrapped)
        if tag == "origin" and row["centroid"] is not None:
            origin_centroid = row["centroid"]
        cent = "n/a" if row["centroid"] is None else f"({row['centroid'][0]:.1f},{row['centroid'][1]:.1f})"
        print(
            f"{tag:<10} {row['valid']}/{row['taught']:>2} "
            f"{row['yaw_deg']:7.2f} {cent:>16} {row['pixel_err']:10.4f}"
        )

    if origin_centroid is not None:
        print(f"\nTeach blob centroid ~ ({origin_centroid[0]:.1f}, {origin_centroid[1]:.1f}) px")

    print(f"\n=== Smooth circle R={args_cli.target_radius}m period={args_cli.target_period_s}s ===")
    unwrapped.set_target_offset(0.0, 0.0)
    vals: list[int] = []
    for step in range(args_cli.smooth_steps):
        phase = step * dt * omega
        dx = args_cli.target_radius * math.cos(phase)
        dy = args_cli.target_radius * math.sin(phase)
        unwrapped.set_smooth_target_offset(dx, dy)
        env.step(zero)
        if step % 10 == 0:
            unwrapped._ensure_pipeline()
            row = _metrics(unwrapped)
            vals.append(row["valid"])
            cent = "n/a" if row["centroid"] is None else f"({row['centroid'][0]:.1f},{row['centroid'][1]:.1f})"
            print(
                f"step={step:3d} valid={row['valid']}/{row['taught']} "
                f"yaw={row['yaw_deg']:.2f}° centroid={cent} err={row['pixel_err']:.4f}"
            )

    if vals:
        print(f"smooth valid_pts: min={min(vals)} max={max(vals)} mean={sum(vals)/len(vals):.1f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
