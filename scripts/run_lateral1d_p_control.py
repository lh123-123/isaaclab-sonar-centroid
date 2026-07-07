#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""1D lateral sonar IBVS — P-control **debug baseline only** (not for training).

Production control path: SAC on `Isaac-SonarIbvsDrl-Lateral1D-Direct-v0`
(see `scripts/train_sonar_lateral1d_sac.sh` and `sonar_ibvs_drl.md` §9).

This script reuses the same perception pipeline as DRL but replaces the policy
with analytic P / hybrid / beacon servos for A/B comparison and tuning.

Usage:
  # GUI + sonar viewport
  python scripts/run_lateral1d_p_control.py --sonar_ui

  # Headless run + metrics + save PNGs
  python scripts/run_lateral1d_p_control.py --headless --enable_cameras --max_steps 400 --save_images

  # Auto-tune Kp / Kff (headless)
  python scripts/run_lateral1d_p_control.py --headless --enable_cameras --sweep_gains
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="1D lateral IBVS P-control (no DRL).")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SonarIbvsDrl-Lateral1D-Debug-Direct-v0",
    help="Debug task (manual target) or Isaac-SonarIbvsDrl-Lateral1D-Direct-v0 (env-driven target).",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--kp", type=float, default=3.0, help="P gain on mean horizontal pixel error (IBVS).")
parser.add_argument("--kp_geo", type=float, default=3.0, help="P gain on (target_dy - robot_lat) / amp.")
parser.add_argument("--kff", type=float, default=0.50, help="Feedforward gain on target lateral velocity.")
parser.add_argument(
    "--control_mode",
    type=str,
    choices=("hybrid", "beacon_cart", "beacon_ibvs", "ibvs_only"),
    default="hybrid",
    help="hybrid=IBVS+geo; beacon_cart=peak/track servo; beacon_ibvs=beacon+IBVS (no GT); ibvs_only=mean_ex.",
)
parser.add_argument("--kp_beacon", type=float, default=6.0, help="P gain on cart beacon lateral error (beacon_* modes).")
parser.add_argument("--ki_beacon", type=float, default=0.8, help="Integral gain on beacon/IBVS error (beacon_* modes).")
parser.add_argument("--min_valid_gate", type=int, default=2, help="Min valid features for full P gain; below scales down.")
parser.add_argument("--peak_threshold", type=float, default=0.10, help="Cart image peak threshold for beacon detect.")
parser.add_argument(
    "--max_lin_vel",
    type=float,
    default=None,
    help="Robot max lateral speed (m/s). Default: env cfg (2.0 for lateral1d).",
)
parser.add_argument("--action_smooth", type=float, default=0.25, help="Exponential smoothing on sway [0,1].")
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--warmup_steps", type=int, default=30, help="Steps excluded from metrics average.")
parser.add_argument(
    "--lateral_amp_m",
    type=float,
    default=None,
    help="Target sinusoid amplitude (m). Default: env cfg.",
)
parser.add_argument(
    "--lateral_period_s",
    type=float,
    default=None,
    help="Target sinusoid period (s). Default: env cfg.",
)
parser.add_argument(
    "--debug_interval",
    type=int,
    default=10,
    help="Print detailed P-control debug every N steps (0=off).",
)
parser.add_argument(
    "--save_images",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Save sonar/cart/feature PNGs at capture steps (headless).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/p_control_debug",
    help="Directory for saved PNGs when --save_images.",
)
parser.add_argument(
    "--plot_trajectory",
    action="store_true",
    help="Save target vs robot lateral trajectory PNG after the run.",
)
parser.add_argument(
    "--trajectory_plot",
    type=str,
    default=None,
    help="Output path for trajectory plot (default: <output_dir>/trajectory.png).",
)
parser.add_argument(
    "--compare_ibvs_hybrid",
    action="store_true",
    help="Headless A/B: pure IBVS (no GT) vs hybrid (IBVS+geo GT), save comparison plot.",
)
parser.add_argument(
    "--plot_feature_trails",
    action="store_true",
    help="Run control and save cart image with teach points + feature motion trails.",
)
parser.add_argument(
    "--feature_trails_plot",
    type=str,
    default=None,
    help="Output path for feature trails plot (default: <output_dir>/feature_trails.png).",
)
parser.add_argument(
    "--capture_steps",
    type=str,
    default="0,50,100,150,199",
    help="Comma-separated step indices to save images.",
)
parser.add_argument(
    "--sweep_beacon_ibvs",
    action="store_true",
    help="Grid search Kp_ibvs x Kp_beacon x Kff for beacon_ibvs mode (no pose GT).",
)
parser.add_argument(
    "--sweep_gains",
    action="store_true",
    help="Headless grid search over Kp/Kff; prints best combo.",
)
parser.add_argument(
    "--sonar_ui",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import cv2

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import (
    euler_xyz_from_quat,
    hover_lateral_error_signed,
    robot_lateral_sway_signed,
)
from isaaclab_tasks.direct.sonar_flow_hover.modules.beacon_cart import (
    detect_beacon_peak,
    teach_beacon_centroid,
    tracked_beacon_centroid,
)
from isaaclab_tasks.direct.sonar_flow_hover.modules.lateral_p_controller import (
    Lateral1DPController,
    format_p_control_debug,
)
from isaaclab_tasks.utils import parse_env_cfg


def _parse_capture_steps(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def _array_to_u8(arr: np.ndarray) -> np.ndarray:
    peak = float(max(arr.max(), 1.0e-6))
    return np.clip(arr / peak * 255.0, 0, 255).astype(np.uint8)


def _load_teach_snapshot(unwrapped) -> "ReferenceSnapshot | None":
    from isaaclab_tasks.direct.sonar_flow_hover.reference_snapshot import ReferenceSnapshot

    path = str(getattr(unwrapped.cfg, "reference_snapshot_path", "logs/station_ref/station_ref.npz"))
    try:
        return ReferenceSnapshot.load(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[save] teach snapshot unavailable ({exc})", flush=True)
        return None


def _perception_frame(unwrapped) -> str:
    return str(getattr(unwrapped.cfg, "perception_frame", "cart")).lower()


def _track_image_np(unwrapped) -> np.ndarray:
    """Perception track image (polar or cart) for feature overlays."""
    if _perception_frame(unwrapped) == "polar":
        return unwrapped.track_image[0].detach().cpu().numpy()
    return unwrapped.cart_image[0].detach().cpu().numpy()


def _teach_bg_image(teach_snap, unwrapped, fallback: np.ndarray | None) -> np.ndarray:
    if teach_snap is not None:
        if _perception_frame(unwrapped) == "polar" and teach_snap.track_image is not None:
            return teach_snap.track_image
        if teach_snap.cart_image is not None:
            return teach_snap.cart_image
    if isinstance(fallback, np.ndarray) and fallback.size > 1:
        return fallback
    return np.zeros((128, 128), dtype=np.float32)


def _draw_teach_overlay(cart: np.ndarray, ref_pts: np.ndarray, *, title: str) -> np.ndarray:
    """Annotate teach cart with reference feature points only."""
    u8 = np.clip(cart, 0.0, 1.0)
    if u8.max() <= 1.0 + 1.0e-6:
        u8 = (u8 * 255.0).astype(np.uint8)
    else:
        u8 = _array_to_u8(cart)
    img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    h, w = cart.shape
    for i, pt in enumerate(ref_pts):
        if float(np.linalg.norm(pt)) < 1.0e-3:
            continue
        ix, iy = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        cv2.drawMarker(img, (ix, iy), (0, 200, 0), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
        cv2.putText(img, str(i), (ix + 4, iy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img, title, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _draw_feature_overlay(
    cart: np.ndarray,
    ref_pts: np.ndarray,
    curr_pts: np.ndarray,
    valid: np.ndarray,
    *,
    title: str,
    beacon_x: float | None = None,
    ref_beacon_x: float | None = None,
) -> np.ndarray:
    u8 = np.clip(cart, 0.0, 1.0)
    if u8.max() <= 1.0 + 1.0e-6:
        u8 = (u8 * 255.0).astype(np.uint8)
    else:
        u8 = _array_to_u8(cart)
    img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    h, w = cart.shape
    for i in range(len(ref_pts)):
        if float(np.linalg.norm(ref_pts[i])) < 1.0e-3:
            continue
        rx, ry = int(round(ref_pts[i, 0])), int(round(ref_pts[i, 1]))
        if not (0 <= rx < w and 0 <= ry < h):
            continue
        ok = bool(valid[i])
        cv2.drawMarker(
            img, (rx, ry), (0, 200, 0) if ok else (0, 120, 0),
            markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1,
        )
        if ok:
            cx, cy = int(round(curr_pts[i, 0])), int(round(curr_pts[i, 1]))
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
                cv2.line(img, (rx, ry), (cx, cy), (0, 220, 255), 1, cv2.LINE_AA)
    if ref_beacon_x is not None:
        ix = int(round(ref_beacon_x))
        cv2.line(img, (ix, 0), (ix, h - 1), (255, 200, 0), 1, cv2.LINE_AA)
    if beacon_x is not None:
        bx = int(round(beacon_x))
        cv2.circle(img, (bx, h // 2), 4, (255, 0, 255), -1)
        if ref_beacon_x is not None:
            cv2.line(img, (int(round(ref_beacon_x)), h // 2), (bx, h // 2), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, title, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return img


_teach_saved_dirs: set[str] = set()


def _save_teach_reference_images(output_dir: str, teach) -> list[str]:
    """Write teach sonar/cart/annotated once per output directory."""
    if output_dir in _teach_saved_dirs or teach is None:
        return []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if teach.sonar_image is not None:
        p = out / "teach_sonar.png"
        cv2.imwrite(str(p), _array_to_u8(teach.sonar_image))
        paths.append(str(p))
    track = getattr(teach, "track_image", None) or teach.cart_image
    frame = str(getattr(teach, "perception_frame", "cart") or "cart")
    if track is not None:
        p = out / ("teach_polar.png" if frame == "polar" else "teach_cart.png")
        cv2.imwrite(str(p), _array_to_u8(track))
        paths.append(str(p))
        ann = out / "teach_annotated.png"
        cv2.imwrite(
            str(ann),
            _draw_teach_overlay(
                track,
                teach.ref_points,
                title=f"teach frame={frame} n_pts={len(teach.ref_points)} ref_yaw={teach.ref_yaw or 0.0:+.3f}",
            ),
        )
        paths.append(str(ann))
    elif teach.cart_image is not None:
        p = out / "teach_cart.png"
        cv2.imwrite(str(p), _array_to_u8(teach.cart_image))
        paths.append(str(p))
    _teach_saved_dirs.add(output_dir)
    print(f"[save] teach -> {out}/ ({len(paths)} PNGs)", flush=True)
    return paths


def _save_step_images(
    unwrapped,
    *,
    step: int,
    target_dy: float,
    robot_lat: float,
    metrics: dict,
    ctrl_dbg: dict,
    output_dir: str,
    teach=None,
) -> list[str]:
    unwrapped._ensure_pipeline()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"step{step:04d}"

    sonar = unwrapped.sonar_image[0].detach().cpu().numpy()
    track = _track_image_np(unwrapped)
    cart = unwrapped.cart_image[0].detach().cpu().numpy()
    ft = unwrapped.feature_tracker
    n_track = int(ft.max_track)
    ref = ft.ref_points[0, :n_track].detach().cpu().numpy()
    curr = ft.curr_points[0, :n_track].detach().cpu().numpy()
    valid = ft.valid[0, :n_track].detach().cpu().numpy()

    lat_err = robot_lat - target_dy
    title = (
        f"step={step} tgt_y={target_dy:+.2f} rob_y={robot_lat:+.2f} "
        f"lat_err={lat_err:+.2f}m pixel_err={metrics['pixel_err']:.4f} "
        f"valid={metrics['valid_pts']}"
    )

    paths = _save_teach_reference_images(output_dir, teach)
    sonar_path = out / f"{tag}_sonar.png"
    cart_path = out / f"{tag}_cart.png"
    track_path = out / f"{tag}_track.png"
    ann_path = out / f"{tag}_annotated.png"
    cv2.imwrite(str(sonar_path), _array_to_u8(sonar))
    cv2.imwrite(str(cart_path), _array_to_u8(cart))
    cv2.imwrite(str(track_path), _array_to_u8(track))
    cv2.imwrite(str(ann_path), _draw_feature_overlay(
        track, ref, curr, valid, title=title,
        beacon_x=ctrl_dbg.get("beacon_x"),
        ref_beacon_x=ctrl_dbg.get("ref_beacon_x"),
    ))
    paths.extend([str(sonar_path), str(cart_path), str(track_path), str(ann_path)])
    print(f"[save] {tag} -> {out}/ sonar+cart+annotated ({len(paths)} PNGs total)", flush=True)
    return paths


def _save_trajectory_plot(
    *,
    times_s: list[float],
    target_dy: list[float],
    robot_lat: list[float],
    valid_pts: list[int],
    output_path: str,
    amp: float,
    period: float,
    control_mode: str,
) -> str:
    import matplotlib.pyplot as plt

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(times_s, target_dy, "C0-", linewidth=2.0, label="target body-Y (m)")
    ax.plot(times_s, robot_lat, "C1-", linewidth=1.6, label="robot lateral (m)")
    ax.fill_between(
        times_s,
        target_dy,
        robot_lat,
        color="C3",
        alpha=0.15,
        label="tracking error band",
    )
    if valid_pts:
        ax2 = ax.twinx()
        ax2.plot(times_s, valid_pts, "C2:", linewidth=1.2, alpha=0.85, label="valid features")
        ax2.set_ylabel("valid feature count")
        ax2.set_ylim(-0.5, max(max(valid_pts) + 1, 5))
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    else:
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("lateral offset (m)")
    ax.set_title(
        f"Lateral 1D trajectory  mode={control_mode}  "
        f"target sin ±{amp:.2f}m  T={period:.1f}s"
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] trajectory -> {out.resolve()}", flush=True)
    return str(out.resolve())


def _save_compare_plot(
    *,
    times_s: list[float],
    target_dy: list[float],
    runs: list[tuple[str, list[float], dict[str, float]]],
    output_path: str,
    amp: float,
    period: float,
) -> str:
    import matplotlib.pyplot as plt

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax0 = axes[0]
    ax0.plot(times_s, target_dy, "k--", linewidth=1.8, label="target body-Y")
    colors = ("C1", "C0")
    for i, (label, robot_lat, _m) in enumerate(runs):
        ax0.plot(times_s, robot_lat, colors[i % len(colors)], linewidth=1.6, label=label)
    ax0.set_ylabel("lateral offset (m)")
    ax0.set_title(f"IBVS vs IBVS+GT  target sin ±{amp:.2f}m  T={period:.1f}s")
    ax0.grid(True, alpha=0.3)
    ax0.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
    ax0.legend(loc="upper right", fontsize=9)

    ax1 = axes[1]
    for i, (label, robot_lat, _m) in enumerate(runs):
        err = [r - t for r, t in zip(robot_lat, target_dy)]
        ax1.plot(times_s, err, colors[i % len(colors)], linewidth=1.4, label=f"{label} err")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("robot - target (m)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
    ax1.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] compare -> {out.resolve()}", flush=True)
    return str(out.resolve())


def _save_feature_trails_plot(
    *,
    cart_bg: np.ndarray,
    ref_points: np.ndarray,
    histories: list[list[tuple[float, float, bool]]],
    output_path: str,
    title: str,
) -> str:
    """Cart image with teach refs (green) and per-feature tracking trails."""
    u8 = np.clip(cart_bg, 0.0, 1.0)
    if u8.max() <= 1.0 + 1.0e-6:
        u8 = (u8 * 255.0).astype(np.uint8)
    else:
        u8 = _array_to_u8(cart_bg)
    img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    h, w = cart_bg.shape

    trail_colors = [
        (0, 180, 255),
        (255, 120, 0),
        (0, 220, 120),
        (220, 80, 255),
        (255, 255, 0),
        (80, 255, 255),
    ]
    lost_segments = 0
    for i, hist in enumerate(histories):
        if not hist:
            continue
        color = trail_colors[i % len(trail_colors)]
        prev_valid = False
        for j, (x, y, ok) in enumerate(hist):
            ix, iy = int(round(x)), int(round(y))
            if j > 0:
                px, py, p_ok = hist[j - 1]
                pti = (int(round(px)), int(round(py)))
                ti = (ix, iy)
                if ok and p_ok:
                    cv2.line(img, pti, ti, color, 1, cv2.LINE_AA)
                elif p_ok and not ok:
                    cv2.line(img, pti, ti, (0, 0, 220), 2, cv2.LINE_AA)
                    lost_segments += 1
            if ok:
                cv2.circle(img, (ix, iy), 1, color, -1, cv2.LINE_AA)
            prev_valid = ok
        if hist:
            lx, ly, lok = hist[-1]
            lxi, lyi = int(round(lx)), int(round(ly))
            if 0 <= lxi < w and 0 <= lyi < h:
                dot_color = color if lok else (0, 0, 255)
                cv2.circle(img, (lxi, lyi), 3, dot_color, -1, cv2.LINE_AA)

    for i, pt in enumerate(ref_points):
        if float(np.linalg.norm(pt)) < 1.0e-3:
            continue
        ix, iy = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        cv2.drawMarker(img, (ix, iy), (0, 220, 0), cv2.MARKER_CROSS, 10, 2)
        cv2.putText(img, f"T{i}", (ix + 5, iy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(img, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        "green+=teach  colored trail=tracked  red line=loss  red dot=lost end",
        (4, h - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"[plot] feature trails -> {out.resolve()} (loss_segments={lost_segments})", flush=True)
    return str(out.resolve())


def _save_feature_trails_only(
    cart_bg: np.ndarray,
    histories: list[list[tuple[float, float, bool]]],
    output_path: str,
    *,
    line_bgr: tuple[int, int, int] = (0, 255, 255),
    line_width: int = 1,
) -> str:
    """Track image background with thin feature trails only (no markers/labels)."""
    u8 = np.clip(cart_bg, 0.0, 1.0)
    if u8.max() <= 1.0 + 1.0e-6:
        u8 = (u8 * 255.0).astype(np.uint8)
    else:
        u8 = _array_to_u8(cart_bg)
    img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    h, w = cart_bg.shape
    n_seg = 0
    for hist in histories:
        if not hist:
            continue
        for j in range(1, len(hist)):
            x0, y0, ok0 = hist[j - 1]
            x1, y1, ok1 = hist[j]
            if not (ok0 and ok1):
                continue
            p0 = (int(round(x0)), int(round(y0)))
            p1 = (int(round(x1)), int(round(y1)))
            if not (0 <= p0[0] < w and 0 <= p0[1] < h and 0 <= p1[0] < w and 0 <= p1[1] < h):
                continue
            cv2.line(img, p0, p1, line_bgr, line_width, cv2.LINE_AA)
            n_seg += 1
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"[plot] feature trails only -> {out.resolve()} (segments={n_seg})", flush=True)
    return str(out.resolve())


def _save_centroid_trail_lines_only(
    cart_bg: np.ndarray,
    trail_points: list[tuple[float, float]],
    output_path: str,
    *,
    line_bgr: tuple[int, int, int] = (0, 255, 255),
    line_width: int = 1,
) -> str:
    """Track image background with centroid trail lines only (no markers)."""
    u8 = np.clip(cart_bg, 0.0, 1.0)
    if u8.max() <= 1.0 + 1.0e-6:
        u8 = (u8 * 255.0).astype(np.uint8)
    else:
        u8 = _array_to_u8(cart_bg)
    img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    h, w = cart_bg.shape
    n_seg = 0
    for j in range(1, len(trail_points)):
        x0, y0 = trail_points[j - 1]
        x1, y1 = trail_points[j]
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))
        if not (0 <= p0[0] < w and 0 <= p0[1] < h and 0 <= p1[0] < w and 0 <= p1[1] < h):
            continue
        cv2.line(img, p0, p1, line_bgr, line_width, cv2.LINE_AA)
        n_seg += 1
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"[plot] centroid lines only -> {out.resolve()} (points={len(trail_points)} segments={n_seg})", flush=True)
    return str(out.resolve())


def _save_centroid_trail_plot(
    cart_bg: np.ndarray,
    ref_centroid: tuple[float, float],
    trail_points: list[tuple[float, float]],
    output_path: str,
    *,
    line_bgr: tuple[int, int, int] = (0, 255, 255),
    line_width: int = 1,
) -> str:
    """Track image with teach centroid cross + thin tracked-centroid trail."""
    u8 = np.clip(cart_bg, 0.0, 1.0)
    if u8.max() <= 1.0 + 1.0e-6:
        u8 = (u8 * 255.0).astype(np.uint8)
    else:
        u8 = _array_to_u8(cart_bg)
    img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    h, w = cart_bg.shape
    rx, ry = int(round(ref_centroid[0])), int(round(ref_centroid[1]))
    if 0 <= rx < w and 0 <= ry < h:
        cv2.drawMarker(img, (rx, ry), (0, 220, 0), cv2.MARKER_CROSS, 10, 2)
    n_seg = 0
    for j in range(1, len(trail_points)):
        x0, y0 = trail_points[j - 1]
        x1, y1 = trail_points[j]
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))
        if not (0 <= p0[0] < w and 0 <= p0[1] < h and 0 <= p1[0] < w and 0 <= p1[1] < h):
            continue
        cv2.line(img, p0, p1, line_bgr, line_width, cv2.LINE_AA)
        n_seg += 1
    if trail_points:
        ex, ey = trail_points[-1]
        ix, iy = int(round(ex)), int(round(ey))
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(img, (ix, iy), 3, (255, 0, 255), -1, cv2.LINE_AA)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"[plot] centroid trail -> {out.resolve()} (points={len(trail_points)} segments={n_seg})", flush=True)
    return str(out.resolve())


def _save_cart_motion_trails_only(
    cart_bg: np.ndarray,
    trail_points: list[tuple[float, float]],
    output_path: str,
    *,
    line_bgr: tuple[int, int, int] = (0, 255, 255),
    line_width: int = 2,
) -> str:
    """Cart sonar background with a single motion trail (no labels/markers)."""
    u8 = np.clip(cart_bg, 0.0, 1.0)
    if u8.max() <= 1.0 + 1.0e-6:
        u8 = (u8 * 255.0).astype(np.uint8)
    else:
        u8 = _array_to_u8(cart_bg)
    img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    h, w = cart_bg.shape
    n_seg = 0
    for j in range(1, len(trail_points)):
        x0, y0 = trail_points[j - 1]
        x1, y1 = trail_points[j]
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))
        if not (0 <= p0[0] < w and 0 <= p0[1] < h and 0 <= p1[0] < w and 0 <= p1[1] < h):
            continue
        cv2.line(img, p0, p1, line_bgr, line_width, cv2.LINE_AA)
        n_seg += 1
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"[plot] cart motion trails only -> {out.resolve()} (points={len(trail_points)} segments={n_seg})", flush=True)
    return str(out.resolve())


def _plot_valid_segments(
    ax,
    times: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    color: str,
    label: str,
) -> None:
    """Plot time series with gaps when tracking is invalid."""
    import matplotlib.pyplot as plt

    started = False
    seg_t: list[float] = []
    seg_v: list[float] = []
    for t, v, ok in zip(times, values, valid):
        if ok:
            seg_t.append(float(t))
            seg_v.append(float(v))
        elif seg_t:
            ax.plot(seg_t, seg_v, color=color, linewidth=1.8, label=label if not started else None)
            started = True
            seg_t, seg_v = [], []
    if seg_t:
        ax.plot(seg_t, seg_v, color=color, linewidth=1.8, label=label if not started else None)


def _save_feature_trajectories_clean(
    *,
    ref_points: np.ndarray,
    taught_idx: list[int],
    histories: list[list[tuple[float, float, bool]]],
    output_path: str,
    title: str,
    dt: float,
) -> str:
    """Per-feature pixel trajectories vs time (gaps = lost track)."""
    import matplotlib.pyplot as plt

    n = len(histories)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * max(n, 1)), sharex=True)
    if n == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(n, 1)))
    for j, (feat_idx, hist) in enumerate(zip(taught_idx, histories)):
        ax = axes[j]
        if not hist:
            ax.set_visible(False)
            continue
        times = np.arange(len(hist), dtype=np.float64) * dt
        xs = np.array([h[0] for h in hist], dtype=np.float64)
        ys = np.array([h[1] for h in hist], dtype=np.float64)
        valid = np.array([h[2] for h in hist], dtype=bool)
        ref_x = float(ref_points[feat_idx, 0])
        ref_y = float(ref_points[feat_idx, 1])
        color = colors[j % len(colors)]

        _plot_valid_segments(ax, times, xs, valid, color=tuple(color[:3]), label="tracked x")
        ax.axhline(ref_x, color="green", linestyle="--", linewidth=1.2, label=f"teach x={ref_x:.0f}")
        ax.set_ylabel("pixel x")
        n_ok = int(valid.sum())
        status = "OK" if n_ok == len(hist) else ("LOST" if n_ok == 0 else "partial")
        ax.set_title(
            f"T{feat_idx}  {status}  valid {n_ok}/{len(hist)}  "
            f"teach=({ref_x:.0f}, {ref_y:.0f})",
            fontsize=9,
            loc="left",
        )
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right", fontsize=7, ncol=2)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] feature trajectories -> {out.resolve()}", flush=True)
    return str(out.resolve())


def _save_feature_trajectories_xy(
    *,
    ref_points: np.ndarray,
    taught_idx: list[int],
    histories: list[list[tuple[float, float, bool]]],
    output_path: str,
    title: str,
) -> str:
    """2D cart-plane trajectories; teach = green cross, solid=tracked, no sonar background."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(histories), 1)))

    for j, (feat_idx, hist) in enumerate(zip(taught_idx, histories)):
        if not hist:
            continue
        xs = np.array([h[0] for h in hist], dtype=np.float64)
        ys = np.array([h[1] for h in hist], dtype=np.float64)
        valid = np.array([h[2] for h in hist], dtype=bool)
        ref_x = float(ref_points[feat_idx, 0])
        ref_y = float(ref_points[feat_idx, 1])
        color = colors[j % len(colors)]

        seg_x: list[float] = []
        seg_y: list[float] = []
        for x, y, ok in zip(xs, ys, valid):
            if ok:
                seg_x.append(x)
                seg_y.append(y)
            elif seg_x:
                ax.plot(seg_x, seg_y, color=color, linewidth=2.0, label=f"T{feat_idx}")
                seg_x, seg_y = [], []
        if seg_x:
            ax.plot(seg_x, seg_y, color=color, linewidth=2.0, label=f"T{feat_idx}")
        ax.scatter([ref_x], [ref_y], marker="+", s=120, c="green", linewidths=2, zorder=5)
        ax.annotate(f"T{feat_idx}", (ref_x, ref_y), xytext=(4, 4), textcoords="offset points", fontsize=8, color="green")

    ax.set_xlabel("cart pixel x")
    ax.set_ylabel("cart pixel y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] feature trajectories (xy) -> {out.resolve()}", flush=True)
    return str(out.resolve())


def _robot_lateral_m(unwrapped, env_id: int = 0) -> float:
    _, _, yaw_w = euler_xyz_from_quat(unwrapped.robot.data.root_quat_w)
    return float(
        robot_lateral_sway_signed(
            unwrapped._station_origin_w, unwrapped.robot.data.root_pos_w, yaw_w
        )[env_id].item()
    )


def _body_vel_y(unwrapped, env_id: int = 0) -> float:
    return float(unwrapped.robot.data.root_lin_vel_b[env_id, 1].item())


def _print_p_debug(step: int, unwrapped, ctrl_dbg: dict, m: dict, interval: int) -> None:
    if interval <= 0 or step % interval != 0:
        return
    robot_lat = _robot_lateral_m(unwrapped)
    print(
        format_p_control_debug(
            step=step,
            ctrl_dbg=ctrl_dbg,
            robot_lat_m=robot_lat,
            target_dy_m=ctrl_dbg["target_dy_m"],
            pixel_err=float(m["pixel_err"]),
            pos_err_m=float(m["pos_err_m"]),
            vel_body_y_mps=_body_vel_y(unwrapped),
            valid_pts=int(m["valid_pts"]),
        ),
        flush=True,
    )


def _read_target_state(
    unwrapped,
    *,
    amp: float,
    omega: float,
    dt: float,
) -> tuple[float, float]:
    """Current body-Y target offset and analytical lateral speed (m/s)."""
    dy = float(unwrapped._target_kinematic_offset_b[0, 1].item())
    buf = int(unwrapped.episode_length_buf[0].item())
    phase = buf * dt * omega
    dy_vel = amp * omega * math.cos(phase)
    return dy, dy_vel


def _make_controller(
    unwrapped,
    *,
    dt: float,
    kp: float | None = None,
    kp_beacon: float | None = None,
    kff: float | None = None,
    ki_beacon: float | None = None,
    control_mode: str | None = None,
    kp_geo: float | None = None,
) -> Lateral1DPController:
    mode = control_mode or args_cli.control_mode
    geo = float(kp_geo if kp_geo is not None else args_cli.kp_geo)
    if mode in ("beacon_cart", "beacon_ibvs"):
        geo = 0.0
    return Lateral1DPController(
        kp=float(kp if kp is not None else args_cli.kp),
        kp_geo=geo,
        kp_beacon=float(kp_beacon if kp_beacon is not None else args_cli.kp_beacon),
        ki_beacon=float(ki_beacon if ki_beacon is not None else args_cli.ki_beacon),
        kff=float(kff if kff is not None else args_cli.kff),
        action_smooth=args_cli.action_smooth,
        control_mode=mode,
        peak_threshold=args_cli.peak_threshold,
        min_valid_gate=args_cli.min_valid_gate,
        step_dt=dt,
        device=unwrapped.device,
    )


def _rollout(
    env,
    unwrapped,
    controller: Lateral1DPController,
    *,
    amp: float,
    omega: float,
    steps: int,
    warmup: int,
    dt: float,
    realtime: bool,
    collect_trajectory: bool = False,
    collect_centroid_history: bool = False,
) -> dict[str, float | list[float] | np.ndarray | list[tuple[float, float]]]:
    controller.reset()
    env.reset()
    unwrapped.set_displacement_offset(0.0, 0.0)
    unwrapped._ensure_pipeline()
    controller.latch_teach_setpoint(unwrapped.feature_tracker)

    pixel_errs: list[float] = []
    abs_ex: list[float] = []
    lat_errs: list[float] = []
    traj_t: list[float] = []
    traj_target: list[float] = []
    traj_robot: list[float] = []
    ft = unwrapped.feature_tracker
    blob_trail: list[tuple[float, float]] = []
    valid_history: list[int] = []
    ref_centroid = teach_beacon_centroid(ft, 0)
    last_track = np.zeros((1, 1), dtype=np.float32)

    for step in range(steps):
        t0 = time.time()
        dy, dy_vel = _read_target_state(unwrapped, amp=amp, omega=omega, dt=dt)
        robot_lat = _robot_lateral_m(unwrapped)

        actions, ctrl_dbg = controller.compute_with_debug(
            unwrapped.feature_tracker,
            cart_image=unwrapped.cart_image,
            robot_lat_m=robot_lat,
            target_dy_b=dy,
            target_dy_vel_b=dy_vel,
            lateral_amp_m=amp,
            action_shape=env.action_space.shape,
        )
        env.step(actions)

        if collect_centroid_history:
            unwrapped._ensure_pipeline()
            track_np = _track_image_np(unwrapped)
            last_track = track_np
            valid_history.append(int(ft.valid_count()[0].item()))
            peak = tracked_beacon_centroid(ft, 0)
            if peak is None and _perception_frame(unwrapped) == "cart":
                peak = detect_beacon_peak(track_np, center_xy=ref_centroid)
            if peak is not None:
                blob_trail.append(peak)

        if collect_trajectory:
            traj_t.append(step * dt)
            traj_target.append(dy)
            traj_robot.append(robot_lat)

        if step >= warmup:
            m = unwrapped.get_ibvs_metrics(env_id=0)
            pixel_errs.append(float(m["pixel_err"]))
            lat_errs.append(abs(robot_lat - dy))
            ft = unwrapped.feature_tracker
            err = ft.feature_errors()[0]
            valid = ft.valid[0]
            if bool(valid.any().item()):
                scale = float(getattr(ft, "_lateral_norm", ft.diagonal))
                abs_ex.append(float((err[valid, 0].abs() / scale).mean().item()))
            _print_p_debug(step, unwrapped, ctrl_dbg, m, interval=10)

        if realtime and not args_cli.headless:
            sleep_t = dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    out: dict[str, float | list[float] | np.ndarray | list[tuple[float, float]]] = {
        "pixel_err_mean": sum(pixel_errs) / max(len(pixel_errs), 1),
        "pixel_err_max": max(pixel_errs) if pixel_errs else 0.0,
        "abs_ex_mean": sum(abs_ex) / max(len(abs_ex), 1),
        "lat_err_mean": sum(lat_errs) / max(len(lat_errs), 1),
        "valid_end": float(unwrapped.feature_tracker.valid_count()[0].item()),
    }
    if collect_trajectory:
        out["traj_t"] = traj_t
        out["traj_target"] = traj_target
        out["traj_robot"] = traj_robot
    if collect_centroid_history:
        out["centroid_trail"] = blob_trail
        out["ref_centroid"] = ref_centroid
        out["last_track"] = last_track
        out["valid_history"] = valid_history
        n_taught = int((torch.linalg.norm(ft.ref_points[0], dim=-1) > 1.0e-3).sum().item())
        out["n_taught"] = float(n_taught)
    return out


def _run_feature_trails_plot(
    env,
    unwrapped,
    *,
    amp: float,
    omega: float,
    steps: int,
    warmup: int,
    dt: float,
    output_dir: str,
) -> None:
    ctrl = _make_controller(unwrapped, dt=dt)
    print("\n======== Centroid trail plot (moving target) ========", flush=True)
    print(f"  mode={args_cli.control_mode}  steps={steps}", flush=True)
    m = _rollout(
        env, unwrapped, ctrl,
        amp=amp, omega=omega, steps=steps, warmup=warmup,
        dt=dt,
        realtime=bool(args_cli.real_time and not args_cli.headless),
        collect_centroid_history=True,
    )

    teach_snap = _load_teach_snapshot(unwrapped)
    track_bg = _teach_bg_image(
        teach_snap,
        unwrapped,
        m.get("last_track") if isinstance(m.get("last_track"), np.ndarray) else None,
    )
    frame = _perception_frame(unwrapped)
    ref_centroid = m.get("ref_centroid", teach_beacon_centroid(unwrapped.feature_tracker, 0))
    assert isinstance(ref_centroid, tuple)

    trail_pts: list[tuple[float, float]] = []
    centroid_trail = m.get("centroid_trail", [])
    if isinstance(centroid_trail, list):
        trail_pts = centroid_trail

    trails_path = str(
        Path(output_dir) / ("polar_feature_trails.png" if frame == "polar" else "feature_trails.png")
    )
    _save_centroid_trail_lines_only(track_bg, trail_pts, trails_path, line_width=1)

    rx, ry = ref_centroid
    n_taught = int(m.get("n_taught", 4))
    valid_hist = m.get("valid_history", [])
    centroid_lost_steps = max(steps - len(trail_pts), 0)
    feature_lost_steps = 0
    partial_steps = 0
    if isinstance(valid_hist, list) and valid_hist:
        feature_lost_steps = sum(1 for v in valid_hist if v == 0)
        partial_steps = sum(1 for v in valid_hist if 0 < v < n_taught)
    tracking_ok = feature_lost_steps == 0 and centroid_lost_steps == 0 and int(m["valid_end"]) == n_taught
    if trail_pts:
        ex, ey = trail_pts[-1]
        print(
            f"  ref_centroid=({rx:.1f},{ry:.1f})  end=({ex:.1f},{ey:.1f})  "
            f"centroid_pts={len(trail_pts)}/{steps}  valid_end={int(m['valid_end'])}/{n_taught}  "
            f"|lat|={m['lat_err_mean']:.2f}m",
            flush=True,
        )
    print(
        f"  tracking: {'OK (no loss)' if tracking_ok else 'LOST/PARTIAL'}  "
        f"feature_lost_steps={feature_lost_steps}  partial_steps={partial_steps}  "
        f"centroid_gaps={centroid_lost_steps}",
        flush=True,
    )
    print(f"  saved -> {trails_path}", flush=True)
    print("====================================\n", flush=True)


def _compare_ibvs_hybrid(
    env,
    unwrapped,
    *,
    amp: float,
    omega: float,
    period: float,
    steps: int,
    warmup: int,
    dt: float,
    output_dir: str,
) -> None:
    kp = float(args_cli.kp)
    kff_hybrid = float(args_cli.kff)
    kp_geo = float(args_cli.kp_geo)
    cases = [
        (
            "pure IBVS (Kp only, no GT)",
            _make_controller(unwrapped, dt=dt, control_mode="ibvs_only", kp=kp, kp_geo=0.0, kff=0.0),
        ),
        (
            f"hybrid IBVS+GT (Kp={kp}, Kp_geo={kp_geo}, Kff={kff_hybrid})",
            _make_controller(
                unwrapped, dt=dt, control_mode="hybrid", kp=kp, kp_geo=kp_geo, kff=kff_hybrid,
            ),
        ),
    ]

    print("\n======== IBVS vs IBVS+GT Compare ========", flush=True)
    runs: list[tuple[str, list[float], dict[str, float | list[float]]]] = []
    target_ref: list[float] = []
    times_ref: list[float] = []
    for label, ctrl in cases:
        print(f"\n--- {label} ---", flush=True)
        m = _rollout(
            env, unwrapped, ctrl,
            amp=amp, omega=omega, steps=steps, warmup=warmup,
            dt=dt, realtime=False, collect_trajectory=True,
        )
        print(
            f"  |lat_err|_mean={m['lat_err_mean']:.3f}m  "
            f"pixel_err_mean={m['pixel_err_mean']:.4f}  "
            f"|ex|_mean={m['abs_ex_mean']:.4f}  valid_end={int(m['valid_end'])}",
            flush=True,
        )
        robot = m["traj_robot"]
        assert isinstance(robot, list)
        runs.append((label, robot, m))
        if not times_ref:
            t = m["traj_t"]
            tg = m["traj_target"]
            assert isinstance(t, list) and isinstance(tg, list)
            times_ref = t
            target_ref = tg

    plot_path = args_cli.trajectory_plot or str(Path(output_dir) / "compare_ibvs_hybrid.png")
    _save_compare_plot(
        times_s=times_ref,
        target_dy=target_ref,
        runs=[(lbl, rob, m) for lbl, rob, m in runs],  # type: ignore[arg-type]
        output_path=plot_path,
        amp=amp,
        period=period,
    )
    print("\n  Summary:", flush=True)
    for label, _rob, m in runs:
        print(
            f"    {label}: |lat|={m['lat_err_mean']:.3f}m  pixel={m['pixel_err_mean']:.4f}",
            flush=True,
        )
    print("=========================================\n", flush=True)


def _sweep_beacon_ibvs(env, unwrapped, amp: float, omega: float, steps: int, warmup: int, dt: float):
    ki_grid = [0.0, 0.5, 0.8, 1.2]
    kp_grid = [2.0, 3.0]
    kp_bcn_grid = [4.0, 6.0]
    kff_grid = [0.45, 0.5, 0.6]
    best_score: float | None = None
    best: tuple[float, float, float, float, dict] | None = None

    print("\n======== Beacon+IBVS Gain Sweep (no pose GT) ========", flush=True)
    for kp in kp_grid:
        for kp_bcn in kp_bcn_grid:
            for kff in kff_grid:
                for ki in ki_grid:
                    ctrl = _make_controller(
                        unwrapped, dt=dt, kp=kp, kp_beacon=kp_bcn, kff=kff, ki_beacon=ki,
                        control_mode="beacon_ibvs", kp_geo=0.0,
                    )
                    m = _rollout(
                        env, unwrapped, ctrl,
                        amp=amp, omega=omega, steps=steps, warmup=warmup,
                        dt=dt, realtime=False,
                    )
                    score = m["lat_err_mean"] + 0.25 * m["pixel_err_mean"]
                    print(
                        f"  Kp={kp:.1f} Kp_bcn={kp_bcn:.1f} Ki={ki:.1f} Kff={kff:.2f} -> "
                        f"|lat|={m['lat_err_mean']:.3f} pixel={m['pixel_err_mean']:.4f} score={score:.4f}",
                        flush=True,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (kp, kp_bcn, kff, ki, m)

    assert best is not None
    print(
        f"\n  BEST: Kp={best[0]:.1f} Kp_bcn={best[1]:.1f} Kff={best[2]:.2f} Ki={best[3]:.1f} "
        f"|lat|={best[4]['lat_err_mean']:.3f} pixel={best[4]['pixel_err_mean']:.4f}",
        flush=True,
    )
    print("====================================================\n", flush=True)
    return best


def _sweep(env, unwrapped, amp: float, omega: float, steps: int, warmup: int, dt: float):
    kp_grid = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    kff_grid = [0.0, 0.2, 0.35, 0.5, 0.7]
    best: tuple[float, float, dict] | None = None

    print("\n======== P-Control Gain Sweep ========", flush=True)
    for kp in kp_grid:
        for kff in kff_grid:
            ctrl = _make_controller(unwrapped, dt=dt, kp=kp, kff=kff)
            m = _rollout(
                env, unwrapped, ctrl,
                amp=amp, omega=omega, steps=steps, warmup=warmup,
                dt=dt, realtime=False,
            )
            print(
                f"  Kp={kp:4.1f} Kff={kff:.2f} -> pixel_err={m['pixel_err_mean']:.4f} "
                f"max={m['pixel_err_max']:.4f} |ex|={m['abs_ex_mean']:.4f}",
                flush=True,
            )
            if best is None or m["pixel_err_mean"] < best[2]["pixel_err_mean"]:
                best = (kp, kff, m)

    assert best is not None
    print(
        f"\n  BEST: Kp={best[0]:.1f} Kff={best[1]:.2f} "
        f"pixel_err={best[2]['pixel_err_mean']:.4f}",
        flush=True,
    )
    print("======================================\n", flush=True)
    return best


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.sonar_ui and hasattr(env_cfg, "oceansim_sonar_cfg"):
        env_cfg.oceansim_sonar_cfg.sonar_initialize_viewport = True
    env_cfg.print_ibvs_debug = False  # use script's richer p-dbg lines instead
    env_cfg.ibvs_debug_interval = args_cli.debug_interval
    # Sin target driven by env clock (smooth); avoids script teleporting offset each step.
    env_cfg.lateral_target_moving = True
    if args_cli.max_lin_vel is not None:
        env_cfg.max_lin_vel = float(args_cli.max_lin_vel)

    amp = float(args_cli.lateral_amp_m if args_cli.lateral_amp_m is not None else env_cfg.lateral_target_amp_m)
    period = float(
        args_cli.lateral_period_s if args_cli.lateral_period_s is not None else env_cfg.lateral_target_period_s
    )
    omega = 2.0 * math.pi / max(period, 1.0e-3)
    peak_target_speed = amp * omega

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    dt = float(getattr(env, "step_dt", unwrapped.step_dt))

    print("\n======== 1D Lateral P-Control ========", flush=True)
    print(f"  task            : {args_cli.task}", flush=True)
    print(f"  target          : body-Y sin ±{amp:.2f}m  T={period:.1f}s", flush=True)
    print(f"  target peak vel : {peak_target_speed:.2f} m/s (env sin motion)", flush=True)
    print(f"  max_lin_vel     : {env_cfg.max_lin_vel:.2f} m/s", flush=True)
    print(f"  control_mode    : {args_cli.control_mode}", flush=True)
    print(f"  Kp_ibvs / Kp_geo / Kp_bcn / Ki / Kff / smooth: "
          f"{args_cli.kp} / {args_cli.kp_geo} / {args_cli.kp_beacon} / {args_cli.ki_beacon} / "
          f"{args_cli.kff} / {args_cli.action_smooth}", flush=True)
    print("======================================\n", flush=True)

    if args_cli.sweep_beacon_ibvs:
        _sweep_beacon_ibvs(env, unwrapped, amp, omega, args_cli.max_steps, args_cli.warmup_steps, dt)
        env.close()
        simulation_app.close()
        return

    output_dir = os.path.abspath(args_cli.output_dir)
    if args_cli.plot_feature_trails:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _run_feature_trails_plot(
            env, unwrapped,
            amp=amp, omega=omega,
            steps=args_cli.max_steps, warmup=args_cli.warmup_steps, dt=dt,
            output_dir=output_dir,
        )
        env.close()
        simulation_app.close()
        return

    if args_cli.compare_ibvs_hybrid:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _compare_ibvs_hybrid(
            env, unwrapped,
            amp=amp, omega=omega, period=period,
            steps=args_cli.max_steps, warmup=args_cli.warmup_steps, dt=dt,
            output_dir=output_dir,
        )
        env.close()
        simulation_app.close()
        return

    if args_cli.sweep_gains:
        _sweep(env, unwrapped, amp, omega, args_cli.max_steps, args_cli.warmup_steps, dt)
        env.close()
        simulation_app.close()
        return

    controller = _make_controller(unwrapped, dt=dt)
    controller.reset()
    prev_target_dy: float | None = None
    capture_steps = _parse_capture_steps(args_cli.capture_steps)
    output_dir = os.path.abspath(args_cli.output_dir)
    if args_cli.save_images:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"  save_images     : {output_dir} at steps {sorted(capture_steps)}", flush=True)

    teach_snapshot = _load_teach_snapshot(unwrapped) if args_cli.save_images else None
    if args_cli.save_images and teach_snapshot is not None:
        _save_teach_reference_images(output_dir, teach_snapshot)

    step = 0
    pixel_errs: list[float] = []
    lat_errs: list[float] = []
    traj_t: list[float] = []
    traj_target: list[float] = []
    traj_robot: list[float] = []
    traj_valid: list[int] = []
    env.reset()
    unwrapped.set_displacement_offset(0.0, 0.0)
    unwrapped._ensure_pipeline()
    controller.reset()
    controller.latch_teach_setpoint(unwrapped.feature_tracker)

    while simulation_app.is_running():
        t0 = time.time()
        dy, dy_vel = _read_target_state(unwrapped, amp=amp, omega=omega, dt=dt)
        dy_delta = 0.0 if prev_target_dy is None else dy - prev_target_dy
        prev_target_dy = dy
        robot_lat = _robot_lateral_m(unwrapped)

        actions, ctrl_dbg = controller.compute_with_debug(
            unwrapped.feature_tracker,
            cart_image=unwrapped.cart_image,
            robot_lat_m=robot_lat,
            target_dy_b=dy,
            target_dy_vel_b=dy_vel,
            lateral_amp_m=amp,
            action_shape=env.action_space.shape,
        )
        env.step(actions)

        m = unwrapped.get_ibvs_metrics(env_id=0)
        traj_t.append(step * dt)
        traj_target.append(dy)
        traj_robot.append(robot_lat)
        traj_valid.append(int(m["valid_pts"]))
        if step >= args_cli.warmup_steps:
            pixel_errs.append(float(m["pixel_err"]))
            lat_errs.append(robot_lat - dy)

        if args_cli.save_images and step in capture_steps:
            _save_step_images(
                unwrapped,
                step=step,
                target_dy=dy,
                robot_lat=robot_lat,
                metrics=m,
                ctrl_dbg=ctrl_dbg,
                output_dir=output_dir,
                teach=teach_snapshot,
            )

        _print_p_debug(step, unwrapped, ctrl_dbg, m, args_cli.debug_interval)

        if step % 50 == 0 and args_cli.debug_interval > 0:
            robot_lat = _robot_lateral_m(unwrapped)
            lat_err = robot_lat - dy
            print(
                f"[p-sum] step={step:4d} target_dy={dy:+.3f} d_dy={dy_delta:+.4f} "
                f"lat_err={lat_err:+.3f}m pixel_err_mean_so_far="
                f"{sum(pixel_errs)/max(len(pixel_errs),1):.4f} valid={m['valid_pts']}",
                flush=True,
            )

        step += 1
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            mean_pe = sum(pixel_errs) / max(len(pixel_errs), 1)
            mean_lat = sum(abs(v) for v in lat_errs) / max(len(lat_errs), 1)
            robot_lat = _robot_lateral_m(unwrapped)
            print(
                f"\n[p-ctrl] done steps={step} mode={args_cli.control_mode} "
                f"pixel_err_mean={mean_pe:.4f} |lat_err|_mean={mean_lat:.3f} "
                f"pixel_err_last={m['pixel_err']:.4f} valid={m['valid_pts']} "
                f"robot_lat={robot_lat:+.3f} target_dy={dy:+.3f} lat_err={robot_lat-dy:+.3f}",
                flush=True,
            )
            if args_cli.save_images:
                print(
                    f"[save] output: {output_dir} "
                    f"(teach_* + {len(capture_steps)} steps x sonar/cart/annotated)",
                    flush=True,
                )
            if args_cli.plot_trajectory and traj_t:
                plot_path = args_cli.trajectory_plot or str(Path(output_dir) / "trajectory.png")
                _save_trajectory_plot(
                    times_s=traj_t,
                    target_dy=traj_target,
                    robot_lat=traj_robot,
                    valid_pts=traj_valid,
                    output_path=plot_path,
                    amp=amp,
                    period=period,
                    control_mode=args_cli.control_mode,
                )
            break

        if args_cli.real_time and not args_cli.headless:
            sleep_t = dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
