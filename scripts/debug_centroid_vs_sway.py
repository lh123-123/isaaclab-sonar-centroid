#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless: map centroid pixel error vs sway tracking error (robot_sway - target_dy)."""

from __future__ import annotations

import argparse
import math

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug centroid vs actual sway tracking error.")
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Lateral1D-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--sweep_min", type=float, default=-1.5)
parser.add_argument("--sweep_max", type=float, default=1.5)
parser.add_argument("--sweep_n", type=int, default=13)
parser.add_argument("--settle_steps", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.sonar_flow_hover.modules.debug import collect_metrics
from isaaclab_tasks.utils import parse_env_cfg


def _sample(
    env,
    unwrapped,
    *,
    robot_dy: float,
    target_dy: float,
    settle_steps: int,
) -> dict[str, float]:
    zero = torch.zeros(env.action_space.shape, device=unwrapped.device)
    unwrapped.set_target_offset(0.0, target_dy)
    unwrapped.set_displacement_offset(0.0, robot_dy)
    for _ in range(settle_steps):
        env.step(zero)
    unwrapped._ensure_pipeline()
    geom_err = unwrapped._geometric_centroid_errors_px()
    m = collect_metrics(
        pipeline=unwrapped.pipeline,
        robot=unwrapped.robot,
        station_origin_w=unwrapped._station_origin_w,
        actions=unwrapped.actions,
        reference_mode=unwrapped.ref_mgr.mode,
        kinematic_offset_w=unwrapped._kinematic_offset_w,
        target_kinematic_offset_b=unwrapped._target_kinematic_offset_b,
        centroid_err_px=geom_err,
        lateral_1d=True,
        env_id=0,
    )
    return {
        "robot_sway_m": float(m["robot_sway_m"]),
        "target_dy_m": float(m["target_dy_m"]),
        "sway_error_m": float(m["sway_error_m"]),
        "centroid_daz_px": float(m["centroid_daz_px"]),
        "centroid_dr_px": float(m["centroid_dr_px"]),
        "centroid_lat_px": float(m["centroid_lat_px"]),
        "centroid_dx_px": float(m["centroid_dx"]),
        "valid_pts": int(m["valid_pts"]),
    }


def _corr(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2 or float(np.std(xs)) < 1.0e-9 or float(np.std(ys)) < 1.0e-9:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
    )
    env_cfg.lateral_target_moving = False
    env_cfg.freeze_robot_pose = True
    env_cfg.print_ibvs_debug = False
    env_cfg.scene.num_envs = 1

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    zero = torch.zeros(env.action_space.shape, device=unwrapped.device)
    env.reset()
    unwrapped.set_target_velocity(0.0, 0.0)

    grid = np.linspace(args_cli.sweep_min, args_cli.sweep_max, args_cli.sweep_n)

    print("\n=== A: target_dy=0, sweep robot sway (m) ===", flush=True)
    print(f"{'robot_dy':>8} {'sway_err':>8} {'cent_px':>9} {'dx_px':>8} {'valid':>5}", flush=True)
    rows_a: list[dict] = []
    for dy in grid:
        row = _sample(env, unwrapped, robot_dy=float(dy), target_dy=0.0, settle_steps=args_cli.settle_steps)
        rows_a.append(row)
        print(
            f"{dy:+8.2f} {row['sway_error_m']:+8.3f} {row['centroid_lat_px']:+9.2f} "
            f"{row['centroid_dx_px']:+8.2f} {row['valid_pts']:5d}",
            flush=True,
        )

    print("\n=== B: robot_sway=0, sweep target_dy (m) ===", flush=True)
    print(f"{'target_dy':>8} {'sway_err':>8} {'cent_px':>9} {'dx_px':>8} {'valid':>5}", flush=True)
    rows_b: list[dict] = []
    for dy in grid:
        row = _sample(env, unwrapped, robot_dy=0.0, target_dy=float(dy), settle_steps=args_cli.settle_steps)
        rows_b.append(row)
        print(
            f"{dy:+8.2f} {row['sway_error_m']:+8.3f} {row['centroid_lat_px']:+9.2f} "
            f"{row['centroid_dx_px']:+8.2f} {row['valid_pts']:5d}",
            flush=True,
        )

    print("\n=== C: diagonal robot_sway = target_dy (ideal track) ===", flush=True)
    print(f"{'dy':>8} {'sway_err':>8} {'cent_px':>9} {'dx_px':>8} {'valid':>5}", flush=True)
    rows_c: list[dict] = []
    for dy in grid:
        row = _sample(env, unwrapped, robot_dy=float(dy), target_dy=float(dy), settle_steps=args_cli.settle_steps)
        rows_c.append(row)
        print(
            f"{dy:+8.2f} {row['sway_error_m']:+8.3f} {row['centroid_lat_px']:+9.2f} "
            f"{row['centroid_dx_px']:+8.2f} {row['valid_pts']:5d}",
            flush=True,
        )

    # Runtime-like: robot fixed at spawn, target moves (zero-action case)
    print("\n=== D: robot at 0, target_dy sweep (zero-action mimic) ===", flush=True)
    rows_d = rows_b

    # Correlation summary
    def _pack(rows: list[dict], x_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.array([r[x_key] for r in rows], dtype=np.float64)
        sway_err = np.array([abs(r["sway_error_m"]) for r in rows], dtype=np.float64)
        cent = np.array([r["centroid_lat_px"] for r in rows], dtype=np.float64)
        return xs, sway_err, cent

    xa, err_a, cent_a = _pack(rows_a, "robot_sway_m")
    xb, err_b, cent_b = _pack(rows_b, "target_dy_m")
    xc, err_c, cent_c = _pack(rows_c, "robot_sway_m")

    # Linear fit scale: cent ≈ k * |sway_err| near origin
    mask_a = err_a < 0.8
    k_a = float(np.median(cent_a[mask_a] / np.maximum(err_a[mask_a], 1.0e-6))) if mask_a.any() else float("nan")
    mask_b = err_b < 0.8
    k_b = float(np.median(cent_b[mask_b] / np.maximum(err_b[mask_b], 1.0e-6))) if mask_b.any() else float("nan")

    print("\n=== Correlation (|sway_err| vs |centroid_daz_px|) ===", flush=True)
    print(f"  A robot sweep,  target=0:  corr={_corr(err_a, cent_a):+.3f}  k≈{k_a:.2f} px/m", flush=True)
    print(f"  B target sweep, robot=0:  corr={_corr(err_b, cent_b):+.3f}  k≈{k_b:.2f} px/m", flush=True)
    print(f"  C diagonal (ideal):       corr={_corr(err_c, cent_c):+.3f}  cent_mean={cent_c.mean():+.2f}", flush=True)
    print(f"  C diagonal cent_px max={cent_c.max():+.2f} (should be ~0 if linear & teach correct)", flush=True)

    # Geometric prediction: azimuth shift ≈ lateral / range * (61/120°)
    range_m = 3.0
    fov_deg = 120.0
    cols = 61.0
    px_per_m_small = (cols / fov_deg) * (180.0 / math.pi) / range_m
    norm_per_m = px_per_m_small / cols
    print("\n=== Small-angle geometric estimate (range≈3m, FOV 120°, 61 cols) ===", flush=True)
    print(f"  expected |centroid_lat| / |lateral_err| ≈ {norm_per_m:.4f} norm/m", flush=True)
    print(f"  measured A: {k_a:.4f}   B: {k_b:.4f}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
