#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Map centroid pixel error vs sway tracking error; quantify linear / nonlinear correspondence."""

from __future__ import annotations

import argparse
import math

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug centroid px vs sway error correspondence.")
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Lateral1D-Direct-v0")
parser.add_argument("--sweep_min", type=float, default=-1.5)
parser.add_argument("--sweep_max", type=float, default=1.5)
parser.add_argument("--sweep_n", type=int, default=31)
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
    beacon_bearing_range,
    sonar_pos_w,
)
from isaaclab_tasks.utils import parse_env_cfg


def _corr(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2 or float(np.std(xs)) < 1.0e-9 or float(np.std(ys)) < 1.0e-9:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _poly_fit(xs: np.ndarray, ys: np.ndarray, deg: int) -> tuple[np.ndarray, float, float]:
    """Return (coeffs high-to-low), R², RMSE."""
    if xs.size < deg + 1:
        return np.zeros(deg + 1), float("nan"), float("nan")
    coeffs = np.polyfit(xs, ys, deg)
    pred = np.polyval(coeffs, xs)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1.0e-12 else float("nan")
    rmse = float(np.sqrt(ss_res / xs.size))
    return coeffs, r2, rmse


def _fmt_poly(coeffs: np.ndarray) -> str:
    deg = len(coeffs) - 1
    parts: list[str] = []
    for i, c in enumerate(coeffs):
        power = deg - i
        if power == 0:
            parts.append(f"{c:+.4f}")
        elif power == 1:
            parts.append(f"{c:+.4f}*x")
        else:
            parts.append(f"{c:+.4f}*x^{power}")
    return " ".join(parts)


def _sample_row(
    env,
    unwrapped,
    *,
    robot_dy: float,
    target_dy: float,
    settle_steps: int,
    ocfg,
) -> dict[str, float]:
    zero = torch.zeros(env.action_space.shape, device=unwrapped.device)
    unwrapped.set_target_offset(0.0, target_dy)
    unwrapped.set_displacement_offset(0.0, robot_dy)
    for _ in range(settle_steps):
        env.step(zero)
    unwrapped._ensure_pipeline()

    geom_corr = unwrapped._geometric_centroid_errors_px(corrected=True)
    m = collect_metrics(
        pipeline=unwrapped.pipeline,
        robot=unwrapped.robot,
        station_origin_w=unwrapped._station_origin_w,
        actions=unwrapped.actions,
        reference_mode=unwrapped.ref_mgr.mode,
        kinematic_offset_w=unwrapped._kinematic_offset_w,
        target_kinematic_offset_b=unwrapped._target_kinematic_offset_b,
        centroid_err_px=geom_corr,
        lateral_1d=True,
        env_id=0,
    )
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
    ref_col = float(unwrapped.feature_tracker.ref_centroid()[0, 0].item())
    ref_row = float(unwrapped.feature_tracker.ref_centroid()[0, 1].item())
    cent_daz = float(geom_corr[0, 0].item())
    cent_dr = float(geom_corr[0, 1].item())
    cent_daz_raw = float(unwrapped._geometric_centroid_errors_px(corrected=False)[0, 0].item())
    fov_rad = math.radians(float(ocfg.hori_fov))
    az_min = -0.5 * fov_rad
    w = max(int(ocfg.width) - 1, 1)
    beacon_col = ref_col + cent_daz
    beacon_row = ref_row + cent_dr
    bearing_deg = math.degrees(float(bearing[0].item()))
    range_m = float(horiz_range[0].item())
    # Small-angle theory: d(col)/d(sway) ≈ (w/fov_rad) / range
    local_slope_theory = (w / fov_rad) / max(range_m, 1.0e-3)

    return {
        "robot_dy": robot_dy,
        "target_dy": target_dy,
        "sway_err": float(m["sway_error_m"]),
        "cent_daz_px": cent_daz,
        "cent_daz_raw_px": cent_daz_raw,
        "cent_dr_px": cent_dr,
        "cent_lat_px": abs(cent_daz),
        "sparse_dx_px": float(m["centroid_dx"]),
        "bearing_deg": bearing_deg,
        "range_m": range_m,
        "beacon_col": beacon_col,
        "beacon_row": beacon_row,
        "ref_col": ref_col,
        "ref_row": ref_row,
        "local_slope_theory": local_slope_theory,
        "valid_pts": int(m["valid_pts"]),
    }


def _print_fit_block(title: str, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    print(f"\n=== {title} ===", flush=True)
    print(f"  N={x.size}  x=[{x.min():+.3f},{x.max():+.3f}]  y=[{y.min():+.2f},{y.max():+.2f}]", flush=True)
    for deg in (1, 2, 3):
        coeffs, r2, rmse = _poly_fit(x, y, deg)
        print(f"  deg={deg}: {_fmt_poly(coeffs)}  R²={r2:.6f}  RMSE={rmse:.4f}", flush=True)
    c1, r2_1, rmse_1 = _poly_fit(x, y, 1)
    pred_lin = np.polyval(c1, x)
    resid = y - pred_lin
    print(f"  linear residual: max={np.max(np.abs(resid)):.3f} px  RMS={np.sqrt(np.mean(resid**2)):.3f} px", flush=True)
    print(f"  corr(signed)={_corr(x, y):+.6f}", flush=True)
    return float(c1[0]), rmse_1


def _print_table(rows: list[dict], *, signed: bool) -> None:
    if signed:
        hdr = (
            f"{'sway_err':>8} {'cent_daz':>9} {'bearing':>8} {'range':>6} "
            f"{'beacon_col':>10} {'theory_k':>9} {'local_k':>8} {'resid':>7}"
        )
        print(hdr, flush=True)
        c1, _, _ = _poly_fit(
            np.array([r["sway_err"] for r in rows]),
            np.array([r["cent_daz_px"] for r in rows]),
            1,
        )
        for r in rows:
            pred = float(np.polyval(c1, r["sway_err"]))
            resid = r["cent_daz_px"] - pred
            local_k = r["cent_daz_px"] / r["sway_err"] if abs(r["sway_err"]) > 1.0e-4 else float("nan")
            print(
                f"{r['sway_err']:+8.3f} {r['cent_daz_px']:+9.2f} {r['bearing_deg']:+8.2f} "
                f"{r['range_m']:6.3f} {r['beacon_col']:10.2f} {r['local_slope_theory']:9.3f} "
                f"{local_k:+8.3f} {resid:+7.2f}",
                flush=True,
            )
    else:
        hdr = f"{'|sway|':>8} {'|cent|':>9} {'ratio':>8} {'bearing':>8} {'range':>6}"
        print(hdr, flush=True)
        for r in rows:
            ae = abs(r["sway_err"])
            ac = r["cent_lat_px"]
            ratio = ac / ae if ae > 1.0e-4 else float("nan")
            print(
                f"{ae:8.3f} {ac:9.2f} {ratio:8.3f} {r['bearing_deg']:+8.2f} {r['range_m']:6.3f}",
                flush=True,
            )


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.lateral_target_moving = False
    env_cfg.freeze_robot_pose = True
    env_cfg.print_ibvs_debug = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset()
    unwrapped.set_target_velocity(0.0, 0.0)
    ocfg = env_cfg.oceansim_sonar_cfg

    ref_col = float(unwrapped.feature_tracker.ref_centroid()[0, 0].item())
    ref_row = float(unwrapped.feature_tracker.ref_centroid()[0, 1].item())
    bias = unwrapped._geometric_centroid_bias_px[0].detach().cpu()
    print(f"\nTeach ref centroid = ({ref_col:.2f}, {ref_row:.2f}) px", flush=True)
    print(
        f"Geometric bias at teach pose (subtracted from obs/reward): "
        f"daz={float(bias[0].item()):+.2f} dr={float(bias[1].item()):+.2f} px",
        flush=True,
    )
    print(f"Sweep: robot_dy in [{args_cli.sweep_min}, {args_cli.sweep_max}], N={args_cli.sweep_n}", flush=True)

    grid = np.linspace(args_cli.sweep_min, args_cli.sweep_max, args_cli.sweep_n)

    # --- A: robot sweep, target=0 ---
    rows_a: list[dict] = []
    for dy in grid:
        rows_a.append(_sample_row(env, unwrapped, robot_dy=float(dy), target_dy=0.0, settle_steps=args_cli.settle_steps, ocfg=ocfg))

    print("\n=== A: bias-corrected cent_daz_px vs sway_error (target_dy=0) ===", flush=True)
    _print_table(rows_a, signed=True)

    x_a = np.array([r["sway_err"] for r in rows_a])
    y_a = np.array([r["cent_daz_px"] for r in rows_a])
    slope_a, rmse_a = _print_fit_block("A corrected signed fit: cent_daz_px = f(sway_error)", x_a, y_a)

    # Local slope via finite difference on geometric curve
    local_slopes = []
    for i in range(1, len(rows_a)):
        ds = rows_a[i]["sway_err"] - rows_a[i - 1]["sway_err"]
        dc = rows_a[i]["cent_daz_px"] - rows_a[i - 1]["cent_daz_px"]
        if abs(ds) > 1.0e-6:
            local_slopes.append(dc / ds)
    if local_slopes:
        ls = np.array(local_slopes)
        print(
            f"  local d(cent)/d(sway) over sweep: mean={ls.mean():+.3f} "
            f"std={ls.std():.4f} min={ls.min():+.3f} max={ls.max():+.3f} px/m",
            flush=True,
        )

    # --- B: |.| mapping (what reward uses) ---
    print("\n=== B: |corrected cent_daz_px| vs |sway_error| ===", flush=True)
    _print_table(rows_a, signed=False)
    abs_x = np.abs(x_a)
    abs_y = np.array([r["cent_lat_px"] for r in rows_a])
    _print_fit_block("B abs fit (after bias correction): |cent| = g(|sway|)", abs_x, abs_y)

    # Bias at zero
    zero_row = min(rows_a, key=lambda r: abs(r["sway_err"]))
    print(
        f"\n  bias-corrected at nearest-zero: sway_err={zero_row['sway_err']:+.4f} m  "
        f"cent_daz_px={zero_row['cent_daz_px']:+.2f} px",
        flush=True,
    )

    # --- C: target sweep, robot=0 (same sway_error, inverted roles) ---
    rows_b: list[dict] = []
    for dy in grid:
        rows_b.append(_sample_row(env, unwrapped, robot_dy=0.0, target_dy=float(dy), settle_steps=args_cli.settle_steps, ocfg=ocfg))

    print("\n=== C: target_dy sweep, robot=0 (sway_err = -target_dy) ===", flush=True)
    x_b = np.array([r["sway_err"] for r in rows_b])
    y_b = np.array([r["cent_daz_px"] for r in rows_b])
    slope_b, rmse_b = _print_fit_block("C signed fit (target moves)", x_b, y_b)

    # Symmetry check: robot+ vs target- should give same cent if truly f(sway_err)
    print("\n=== D: symmetry — same |sway_err|, robot vs target ===", flush=True)
    print(f"{'|sway|':>7} {'cent(robot+)':>12} {'cent(target+)':>14} {'diff':>7}", flush=True)
    for ae in (0.2, 0.5, 0.8, 1.0, 1.2):
        ra = min(rows_a, key=lambda r: abs(abs(r["sway_err"]) - ae))
        rb = min(rows_b, key=lambda r: abs(abs(r["sway_err"]) - ae))
        diff = ra["cent_daz_px"] - rb["cent_daz_px"]
        print(
            f"{ae:7.2f} {ra['cent_daz_px']:+12.2f} {rb['cent_daz_px']:+14.2f} {diff:+7.2f}",
            flush=True,
        )

    # --- E: ideal track diagonal ---
    rows_e: list[dict] = []
    for dy in grid:
        rows_e.append(_sample_row(env, unwrapped, robot_dy=float(dy), target_dy=float(dy), settle_steps=args_cli.settle_steps, ocfg=ocfg))

    cent_e = np.array([r["cent_daz_px"] for r in rows_e])
    print("\n=== E: ideal track (robot_dy = target_dy) ===", flush=True)
    print(
        f"  cent_daz_px mean={cent_e.mean():+.2f} std={cent_e.std():.3f} "
        f"max|.|={np.max(np.abs(cent_e)):.2f} px",
        flush=True,
    )

    # --- F: invertibility sway_est = cent / slope ---
    print("\n=== F: invert sway from pixel (linear inverse) ===", flush=True)
    print(f"  using slope_a={slope_a:+.3f} px/m from global linear fit", flush=True)
    print(f"{'cent_daz':>9} {'sway_true':>10} {'sway_est':>10} {'err_est':>10}", flush=True)
    for r in rows_a[:: max(len(rows_a) // 8, 1)]:
        if abs(slope_a) < 1.0e-6:
            break
        sway_est = r["cent_daz_px"] / slope_a
        err_est = sway_est - r["sway_err"]
        print(
            f"{r['cent_daz_px']:+9.2f} {r['sway_err']:+10.3f} {sway_est:+10.3f} {err_est:+10.3f}",
            flush=True,
        )

    # --- Summary ---
    print("\n======== SUMMARY (bias-corrected) ========", flush=True)
    print("  Obs/reward use cent_daz_px after subtracting teach-pose geometry bias", flush=True)
    print(f"  A robot sweep: linear slope ≈ {slope_a:+.3f} px/m, RMSE={rmse_a:.3f} px", flush=True)
    print(f"  C target sweep: linear slope ≈ {slope_b:+.3f} px/m, RMSE={rmse_b:.3f} px", flush=True)
    if abs(slope_a - slope_b) > 0.05:
        print("  WARNING: robot vs target slopes differ — correspondence not symmetric in sim setup", flush=True)
    c2, r2_2, _ = _poly_fit(abs_x, abs_y, 2)
    c1_abs, r2_abs_1, rmse_abs = _poly_fit(abs_x, abs_y, 1)
    print(f"  |cent| vs |sway| linear R²={r2_abs_1:.6f} RMSE={rmse_abs:.3f} px", flush=True)
    c2, r2_2, _ = _poly_fit(x_a, y_a, 2)
    c1, r2_1, _ = _poly_fit(x_a, y_a, 1)
    if r2_2 - r2_1 > 0.001:
        print(
            f"  Signed nonlinearity: quadratic R² gain over linear = {r2_2 - r2_1:.6f} "
            f"(quadratic coeff x²={c2[0]:+.4f})",
            flush=True,
        )
    else:
        print("  Signed map is effectively linear over this sweep range.", flush=True)
    if r2_abs_1 > 0.995:
        print("  |cent| vs |sway| is now monotonic/linear — reward |.| is safe.", flush=True)
    else:
        print("  |cent| vs |sway| still weakly linear — consider lateral_reward_pixel_sq.", flush=True)
    print("==========================================\n", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
