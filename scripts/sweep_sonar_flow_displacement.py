#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sweep small world-frame displacements and log sonar optical-flow response."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sonar flow vs small displacement sweep.")
parser.add_argument("--task", type=str, default="Isaac-SonarFlowHover-Sweep-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=25,
    help="Sim steps to hold at each offset before sampling flow.",
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

# World-frame offsets (m) relative to hover point; beacon is +x from hover.
DISPLACEMENTS: list[tuple[float, float, str]] = [
    (0.00, 0.00, "origin"),
    (0.10, 0.00, "+x 0.10"),
    (0.25, 0.00, "+x 0.25"),
    (0.50, 0.00, "+x 0.50"),
    (-0.10, 0.00, "-x 0.10"),
    (-0.25, 0.00, "-x 0.25"),
    (0.00, 0.10, "+y 0.10"),
    (0.00, 0.25, "+y 0.25"),
    (0.00, -0.10, "-y 0.10"),
    (0.00, -0.25, "-y 0.25"),
    (0.15, 0.15, "diag 0.15"),
]


def _sample_row(unwrapped) -> dict[str, float | bool | str]:
    return unwrapped.get_ibvs_metrics(env_id=0)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    dev = unwrapped.device
    zero_actions = torch.zeros(env.action_space.shape, device=dev)

    print(f"[sweep] task={args_cli.task} settle_steps={args_cli.settle_steps}")
    env.reset()

    ref = _sample_row(unwrapped)
    print(
        f"[sweep] reference (hover geometry): "
        f"ref_theta={ref['ref_theta']:+.4f} rad ref_range={ref['ref_range']:.3f} m "
        f"detected={ref['detected']}",
        flush=True,
    )

    rows: list[dict] = []
    for dx, dy, label in DISPLACEMENTS:
        unwrapped.set_displacement_offset(dx, dy)
        for _ in range(args_cli.settle_steps):
            env.step(zero_actions)

        row = _sample_row(unwrapped)
        row["label"] = label
        row["dx"] = dx
        row["dy"] = dy
        row["d_geo_theta"] = row["geo_theta"] - ref["geo_theta"]
        row["d_geo_range"] = row["geo_r"] - ref["geo_r"]
        rows.append(row)

        print(
            f"[sweep] {label:12s} dx={dx:+.2f} dy={dy:+.2f} "
            f"pos={row['pos_err']:.3f}m "
            f"e_th={row['e_theta']:+.4f} e_r={row['e_range']:+.3f} "
            f"beam_th={row['beam_theta']:+.4f} beam_r={row['beam_r']:.2f} "
            f"d_geo_th={row['d_geo_theta']:+.4f} d_geo_r={row['d_geo_range']:+.3f} "
            f"flow_err={row['flow_err']:.3f} det={row['detected']}",
            flush=True,
        )

    print("\n========== Sonar Flow Displacement Sweep ==========")
    print(
        f"{'label':<14} {'dx':>5} {'dy':>5} {'pos':>5} "
        f"{'e_θ':>7} {'e_r':>6} {'Δgeo_θ':>7} {'Δgeo_r':>7} "
        f"{'flow_err':>8} {'det':>5}"
    )
    print("-" * 78)
    for row in rows:
        print(
            f"{row['label']:<14} {row['dx']:+.2f} {row['dy']:+.2f} "
            f"{row['pos_err']:5.3f} "
            f"{row['e_theta']:+7.4f} {row['e_range']:+6.3f} "
            f"{row['d_geo_theta']:+7.4f} {row['d_geo_range']:+7.3f} "
            f"{row['flow_err']:8.3f} {str(row['detected']):>5}"
        )

    detected_rows = [r for r in rows if r["detected"]]
    if len(detected_rows) >= 2:
        origin = rows[0]
        max_e_theta = max(abs(r["e_theta"]) for r in detected_rows)
        max_e_range = max(abs(r["e_range"]) for r in detected_rows)
        print(f"\n  ref at origin: θ={origin['ref_theta']:+.4f} rad, r={origin['ref_range']:.3f} m")
        print(f"  max |e_theta| over sweep: {max_e_theta:.4f} rad ({math.degrees(max_e_theta):.2f}°)")
        print(f"  max |e_range| over sweep: {max_e_range:.3f} m")
        if max_e_theta > 0.02 or max_e_range > 0.05:
            print("  PASS: flow displacement error responds to small offsets.")
        else:
            print("  WARN: flow error barely changes — check reference mode or peak threshold.")
    else:
        print("\n  FAIL: sonar did not detect beacon during sweep.")
    print("===================================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
