#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare sparse centroid vs geometric beacon projection vs sway_error (linearity check)."""

from __future__ import annotations

import argparse
import math

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug centroid linearity vs sway error.")
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Lateral1D-Direct-v0")
parser.add_argument("--sweep_min", type=float, default=-1.2)
parser.add_argument("--sweep_max", type=float, default=1.2)
parser.add_argument("--sweep_n", type=int, default=13)
parser.add_argument("--settle_steps", type=int, default=6)
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
from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import (
    geometric_beacon_col_px,
    sonar_pos_w,
)
from isaaclab_tasks.utils import parse_env_cfg


def _corr(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2 or float(np.std(xs)) < 1.0e-9 or float(np.std(ys)) < 1.0e-9:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _fit_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2 or float(np.std(xs)) < 1.0e-9:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.lateral_target_moving = False
    env_cfg.freeze_robot_pose = True
    env_cfg.print_ibvs_debug = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    zero = torch.zeros(env.action_space.shape, device=unwrapped.device)
    env.reset()
    unwrapped.set_target_velocity(0.0, 0.0)

    ocfg = env_cfg.oceansim_sonar_cfg
    ref_col = float(unwrapped.feature_tracker.ref_centroid()[0, 0].item())
    print(f"\nTeach ref centroid col = {ref_col:.2f} px", flush=True)

    grid = np.linspace(args_cli.sweep_min, args_cli.sweep_max, args_cli.sweep_n)
    rows: list[dict] = []

    print("\n=== Joint sweep: robot_sway = s, target_dy = 0 ===", flush=True)
    print(
        f"{'s':>6} {'sway_err':>8} {'sparse_dx':>10} {'geom_dcol':>10} "
        f"{'obs_px':>9} {'valid':>5}",
        flush=True,
    )

    for s in grid:
        unwrapped.set_target_offset(0.0, 0.0)
        unwrapped.set_displacement_offset(0.0, float(s))
        for _ in range(args_cli.settle_steps):
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
        sonar_pw = sonar_pos_w(
            unwrapped.robot.data.root_pos_w,
            unwrapped.robot.data.root_quat_w,
            unwrapped._sonar_mount_offset_b,
        )
        _col, geom_dcol = geometric_beacon_col_px(
            beacon_pos_w=unwrapped._beacon_pos_w,
            sonar_pos_w=sonar_pw,
            robot_quat_w=unwrapped.robot.data.root_quat_w,
            ref_col_px=ref_col,
            min_range=float(ocfg.min_range),
            max_range=float(ocfg.max_range),
            polar_width=int(ocfg.width),
            hori_fov_deg=float(ocfg.hori_fov),
        )
        obs_cent_lat = float(geom_err[0, 0].abs().item())
        row = {
            "s": float(s),
            "sway_err": float(m["sway_error_m"]),
            "sparse_dx": float(m["centroid_dx"]),
            "geom_dcol": geom_dcol,
            "obs_cent_lat": obs_cent_lat,
            "cent_lat": float(m["centroid_lat_px"]),
            "valid": int(m["valid_pts"]),
        }
        rows.append(row)
        print(
            f"{s:+6.2f} {row['sway_err']:+8.3f} {row['sparse_dx']:+10.2f} {row['geom_dcol']:+10.2f} "
            f"{row['obs_cent_lat']:9.2f} {row['valid']:5d}",
            flush=True,
        )

    sway_err = np.array([r["sway_err"] for r in rows])
    sparse_dx = np.array([r["sparse_dx"] for r in rows])
    geom_dcol = np.array([r["geom_dcol"] for r in rows])
    sparse_cent = np.array([r["cent_lat"] for r in rows])
    obs_cent = np.array([r["obs_cent_lat"] for r in rows])

    fov_rad = math.radians(float(ocfg.hori_fov))
    px_per_m = (float(ocfg.width) / fov_rad) / 3.0

    print("\n=== Linearity: pixel shift vs sway_error (target=0, robot=s) ===", flush=True)
    print(f"  theory slope (small-angle): {px_per_m:.3f} px/m", flush=True)
    print(f"  geometric dcol vs sway_err:  slope={_fit_slope(sway_err, geom_dcol):+.3f} px/m  corr={_corr(sway_err, geom_dcol):+.4f}", flush=True)
    print(f"  sparse dx vs sway_err:       slope={_fit_slope(sway_err, sparse_dx):+.3f} px/m  corr={_corr(sway_err, sparse_dx):+.4f}", flush=True)
    print(f"  obs geom cent_lat vs |sway|: slope={_fit_slope(np.abs(sway_err), obs_cent):+.3f} px/m  corr={_corr(np.abs(sway_err), obs_cent):+.4f}", flush=True)
    print(f"  sparse cent_lat vs |sway|:   slope={_fit_slope(np.abs(sway_err), sparse_cent):+.3f} px/m  corr={_corr(np.abs(sway_err), sparse_cent):+.4f}", flush=True)

    print("\n=== Joint: target_dy=t, robot_sway=t (ideal track) ===", flush=True)
    ideal_cent = []
    for t in grid:
        unwrapped.set_target_offset(0.0, float(t))
        unwrapped.set_displacement_offset(0.0, float(t))
        for _ in range(args_cli.settle_steps):
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
        ideal_cent.append(float(m["centroid_lat_px"]))
    print(f"  centroid_daz_px mean={np.mean(ideal_cent):+.2f} max={np.max(ideal_cent):+.2f}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
