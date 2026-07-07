#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Tracking: perceived centroid on polar sonar vs actual ball bright-spot centroid (no teach)."""

from __future__ import annotations

import argparse

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Compare perceived vs actual ball centroid on polar sonar (no teach)."
)
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Lateral1D-Direct-v0")
parser.add_argument("--sweep_min", type=float, default=-1.5)
parser.add_argument("--sweep_max", type=float, default=1.5)
parser.add_argument("--sweep_n", type=int, default=21)
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
from isaaclab_tasks.direct.sonar_flow_hover.modules.beacon_cart import detect_beacon_peak_polar_global
from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import (
    beacon_bearing_range,
    beacon_polar_rois,
    euler_xyz_from_quat,
    robot_lateral_sway_signed,
    sonar_pos_w,
)
from isaaclab_tasks.utils import parse_env_cfg


def _sonar_ball_centroid_independent(polar_np: np.ndarray, *, threshold_ratio: float) -> tuple[float, float] | None:
    """Actual ball center from sonar image only (global bright-spot + azimuth mirror)."""
    peak = detect_beacon_peak_polar_global(polar_np, threshold_ratio=threshold_ratio)
    if peak is None:
        return None
    gc, gr = peak
    col = float(polar_np.shape[1] - 1) - gc
    return col, float(gr)


def _perceived_geom_col_row(unwrapped, env_id: int = 0) -> tuple[float, float] | None:
    """Geometry-projected beacon center (Lateral1D obs source)."""
    ocfg = unwrapped.cfg.oceansim_sonar_cfg
    sonar_pw = sonar_pos_w(
        unwrapped.robot.data.root_pos_w,
        unwrapped.robot.data.root_quat_w,
        unwrapped._sonar_mount_offset_b,
    )
    bearing, horiz_range = beacon_bearing_range(
        unwrapped._beacon_pos_w,
        sonar_pw,
        unwrapped.robot.data.root_quat_w,
    )
    rois = beacon_polar_rois(
        bearing,
        horiz_range,
        torch.tensor([env_id], device=unwrapped.device),
        min_range=float(ocfg.min_range),
        max_range=float(ocfg.max_range),
        polar_height=int(ocfg.height),
        polar_width=int(ocfg.width),
        hori_fov_deg=float(ocfg.hori_fov),
    )
    if rois[0] is None:
        return None
    return float(rois[0][0]), float(rois[0][1])


def _sample(env, unwrapped, *, robot_dy: float, target_dy: float, settle_steps: int) -> dict:
    zero = torch.zeros(env.action_space.shape, device=unwrapped.device)
    unwrapped.set_target_offset(0.0, target_dy)
    unwrapped.set_displacement_offset(0.0, robot_dy)
    for _ in range(settle_steps):
        env.step(zero)
    unwrapped._ensure_pipeline()

    env_id = 0
    ft = unwrapped.feature_tracker
    polar_np = unwrapped.pipeline.track_image[env_id].detach().cpu().numpy()
    thr = float(unwrapped.cfg.feature_polar_min_intensity_ratio)

    sonar = _sonar_ball_centroid_independent(polar_np, threshold_ratio=thr)
    geom = _perceived_geom_col_row(unwrapped, env_id)
    sparse_col = float(ft.tracked_centroid()[env_id, 0].item())
    sparse_row = float(ft.tracked_centroid()[env_id, 1].item())
    valid_pts = int(ft.valid_count()[env_id].item())

    _, _, yaw = euler_xyz_from_quat(unwrapped.robot.data.root_quat_w[env_id : env_id + 1])
    robot_sway = float(
        robot_lateral_sway_signed(
            unwrapped._station_origin_w, unwrapped.robot.data.root_pos_w, yaw
        )[env_id].item()
    )

    out: dict = {
        "robot_sway_m": robot_sway,
        "target_dy_m": target_dy,
        "sway_err_m": robot_sway - target_dy,
        "valid_pts": valid_pts,
        "sonar_found": sonar is not None,
    }
    if sonar is None or geom is None:
        out.update(
            {
                "geom_col": float("nan"),
                "geom_row": float("nan"),
                "sparse_col": sparse_col,
                "sparse_row": sparse_row,
                "sonar_col": float("nan"),
                "sonar_row": float("nan"),
                "geom_sonar_dcol": float("nan"),
                "geom_sonar_drow": float("nan"),
                "geom_sonar_dist": float("nan"),
                "sparse_sonar_dcol": float("nan"),
                "sparse_sonar_drow": float("nan"),
                "sparse_sonar_dist": float("nan"),
            }
        )
        return out

    sonar_col, sonar_row = sonar
    geom_col, geom_row = geom
    out.update(
        {
            "geom_col": geom_col,
            "geom_row": geom_row,
            "sparse_col": sparse_col,
            "sparse_row": sparse_row,
            "sonar_col": sonar_col,
            "sonar_row": sonar_row,
            "geom_sonar_dcol": geom_col - sonar_col,
            "geom_sonar_drow": geom_row - sonar_row,
            "geom_sonar_dist": float(np.hypot(geom_col - sonar_col, geom_row - sonar_row)),
            "sparse_sonar_dcol": sparse_col - sonar_col,
            "sparse_sonar_drow": sparse_row - sonar_row,
            "sparse_sonar_dist": float(np.hypot(sparse_col - sonar_col, sparse_row - sonar_row)),
        }
    )
    return out


def _stats(xs: np.ndarray) -> str:
    return f"mean={np.mean(xs):.3f} p90={np.percentile(xs, 90):.3f} max={np.max(xs):.3f}"


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.lateral_target_moving = False
    env_cfg.freeze_robot_pose = True
    env_cfg.print_ibvs_debug = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset()
    unwrapped.set_target_velocity(0.0, 0.0)

    print("\n=== Perceived centroid vs sonar-image ball centroid (NO teach) ===", flush=True)
    print("  sonar_gt: global bright-spot on polar image (azimuth mirrored to geom frame)", flush=True)
    print("  geom:     geometry-projected beacon center (policy Lateral1D perception)", flush=True)
    print("  sparse:   sparse feature tracker tracked_centroid()", flush=True)

    grid = np.linspace(args_cli.sweep_min, args_cli.sweep_max, args_cli.sweep_n)
    rows: list[dict] = []

    print(
        f"\n{'sway':>7} {'geom_c':>7} {'sparse_c':>8} {'sonar_c':>7} "
        f"{'d_geom':>7} {'d_sparse':>8} {'valid':>5}",
        flush=True,
    )
    for dy in grid:
        row = _sample(env, unwrapped, robot_dy=float(dy), target_dy=0.0, settle_steps=args_cli.settle_steps)
        rows.append(row)
        if row["sonar_found"]:
            print(
                f"{row['sway_err_m']:+7.3f} {row['geom_col']:7.2f} {row['sparse_col']:8.2f} "
                f"{row['sonar_col']:7.2f} {row['geom_sonar_dcol']:+7.2f} "
                f"{row['sparse_sonar_dcol']:+8.2f} {row['valid_pts']:5d}",
                flush=True,
            )
        else:
            print(f"{row['sway_err_m']:+7.3f}   sonar ball not detected", flush=True)

    ok = [r for r in rows if r["sonar_found"] and not np.isnan(r["geom_sonar_dist"])]
    if not ok:
        print("\nNo valid samples.", flush=True)
        env.close()
        simulation_app.close()
        return

    gcol = np.abs(np.array([r["geom_sonar_dcol"] for r in ok]))
    grow = np.abs(np.array([r["geom_sonar_drow"] for r in ok]))
    gdist = np.array([r["geom_sonar_dist"] for r in ok])
    scol = np.abs(np.array([r["sparse_sonar_dcol"] for r in ok]))
    sdist = np.array([r["sparse_sonar_dist"] for r in ok])

    print("\n=== Gap: perceived − sonar_image (absolute px) ===", flush=True)
    print(f"  geom   |d_col|  {_stats(gcol)}", flush=True)
    print(f"  geom   |d_row|  {_stats(grow)}", flush=True)
    print(f"  geom   2D dist {_stats(gdist)}", flush=True)
    print(f"  sparse |d_col|  {_stats(scol)}", flush=True)
    print(f"  sparse 2D dist {_stats(sdist)}", flush=True)

    print("\n======== VERDICT ========", flush=True)
    if np.mean(gcol) < 2.5 and np.percentile(gcol, 90) < 3.0:
        print("  Geometry perception ≈ sonar ball center on azimuth (tracking usable).", flush=True)
    else:
        print("  Geometry vs sonar ball: azimuth gap large — perception may not match image.", flush=True)
    if np.mean(scol) > np.mean(gcol) * 1.5:
        print("  Sparse tracker centroid drifts further from sonar ball than geometry.", flush=True)
    print("=========================\n", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
