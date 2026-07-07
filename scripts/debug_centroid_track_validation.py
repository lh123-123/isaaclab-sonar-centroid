#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 1: compare p_perc_abs vs p_sonar_abs over robot sway sweep (sim)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sonar centroid track — perception vs GT sweep.")
parser.add_argument("--task", type=str, default="Isaac-SonarCentroidTrack-Debug-Direct-v0")
parser.add_argument("--sweep_min", type=float, default=-1.5)
parser.add_argument("--sweep_max", type=float, default=1.5)
parser.add_argument("--sweep_n", type=int, default=21)
parser.add_argument("--settle_steps", type=int, default=6)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/centroid_track/phase1_validation",
    help="Directory for annotated polar frames and summary plots.",
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
from isaaclab_tasks.direct.sonar_centroid_track.metrics.validation import ValidationReport
from isaaclab_tasks.utils import parse_env_cfg


def _polar_to_u8(polar: np.ndarray) -> np.ndarray:
    x = polar.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi > lo:
        x = (x - lo) / (hi - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def _draw_centroid_overlay(
    polar: np.ndarray,
    *,
    sway: float,
    p_sonar,
    p_perc,
    p_fixed,
    valid_ratio: float,
) -> np.ndarray:
    """BGR overlay: sonar=green, perc=blue, fixed=yellow."""
    del sway, valid_ratio
    gray = _polar_to_u8(polar)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _dot(pt, color: tuple[int, int, int]) -> None:
        if pt is None:
            return
        c, r = int(round(pt.col)), int(round(pt.row))
        cv2.circle(img, (c, r), 1, color, -1, lineType=cv2.LINE_AA)

    _dot(p_fixed, (0, 255, 255))
    _dot(p_sonar, (0, 255, 0))
    _dot(p_perc, (255, 0, 0))
    return img


def _save_summary_plot(
    out_dir: Path,
    *,
    sways: list[float],
    d_cols: list[float],
    d_rows: list[float],
    track_us: list[float],
    valids: list[float],
    verdict_ok: bool,
    verdict_msg: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Phase 1: p_perc_abs vs p_sonar_abs (polar pixels)", fontsize=12)

    ax = axes[0, 0]
    ax.plot(sways, d_cols, "o-", color="C0", ms=4)
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("robot sway (m)")
    ax.set_ylabel("d_col = perc - sonar (px)")
    ax.set_title("Column error")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(sways, d_rows, "o-", color="C1", ms=4)
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("robot sway (m)")
    ax.set_ylabel("d_row = perc - sonar (px)")
    ax.set_title("Row error")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(sways, track_us, "o-", color="C2", ms=4)
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("robot sway (m)")
    ax.set_ylabel("p_sonar.u (px, fixed frame)")
    ax.set_title("Sonar lateral error (fixed frame)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(sways, valids, "o-", color="C3", ms=4)
    ax.set_xlabel("robot sway (m)")
    ax.set_ylabel("valid ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Detection ok")
    ax.grid(True, alpha=0.3)

    status = "PASS" if verdict_ok else "FAIL"
    fig.text(0.5, 0.01, f"VERDICT: {status} — {verdict_msg}", ha="center", fontsize=10)

    out = out_dir / "summary_metrics.png"
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[save] summary -> {out.resolve()}", flush=True)


def _save_montage(out_dir: Path, frames: list[tuple[float, np.ndarray]]) -> None:
    if not frames:
        return
    import matplotlib.pyplot as plt

    n = len(frames)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (sway, bgr) in zip(axes_flat, frames):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb, aspect="auto")
        ax.set_title(f"sway={sway:+.2f}m", fontsize=9)
        ax.axis("off")
    for ax in axes_flat[len(frames) :]:
        ax.axis("off")
    fig.suptitle("Phase 1 polar montage", fontsize=11)
    out = out_dir / "polar_montage.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[save] montage -> {out.resolve()} ({len(frames)} panels)", flush=True)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.lateral_target_moving = False
    env_cfg.freeze_robot_pose = True
    env_cfg.print_centroid_debug = False

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    polar_dir = out_dir / "polar_frames"
    polar_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset()
    unwrapped.set_target_velocity(0.0, 0.0)

    zero = torch.zeros(env.action_space.shape, device=unwrapped.device, dtype=torch.long)
    report = ValidationReport()
    grid = np.linspace(args_cli.sweep_min, args_cli.sweep_max, args_cli.sweep_n)

    sways: list[float] = []
    d_cols: list[float] = []
    d_rows: list[float] = []
    track_us: list[float] = []
    valids: list[float] = []
    montage_frames: list[tuple[float, np.ndarray]] = []
    montage_indices = set(
        int(round(i)) for i in np.linspace(0, len(grid) - 1, min(5, len(grid)), dtype=int)
    )

    print("\n=== Phase 1: p_perc_abs vs p_sonar_abs (absolute coords) ===\n", flush=True)
    print(f"  output_dir: {out_dir.resolve()}\n", flush=True)
    print(f"{'sway':>7} {'perc_c':>7} {'sonar_c':>7} {'d_col':>7} {'sonar_u':>8} {'valid':>5}", flush=True)

    for i, dy in enumerate(grid):
        unwrapped.set_target_offset(0.0, 0.0)
        unwrapped.set_displacement_offset(0.0, float(dy))
        for _ in range(args_cli.settle_steps):
            env.step(zero)

        frame = unwrapped.get_centroid_frame(0)
        sway = float(unwrapped._robot_sway_m()[0].item())
        p_fixed = unwrapped.fixed.p_fixed_abs_env(0)
        polar_np = unwrapped.polar_image[0].detach().cpu().numpy()

        if frame.p_perc_abs is not None and frame.p_sonar_abs is not None:
            report.add_perc_sample(frame.p_perc_abs, frame.p_sonar_abs, label=f"sway={sway:+.2f}")
            gap = frame.p_perc_abs.col - frame.p_sonar_abs.col
            gap_row = frame.p_perc_abs.row - frame.p_sonar_abs.row
            su = frame.p_sonar.col if frame.p_sonar is not None else float("nan")
            sways.append(sway)
            d_cols.append(gap)
            d_rows.append(gap_row)
            track_us.append(su)
            valids.append(frame.valid_ratio)
            print(
                f"{sway:+7.3f} {frame.p_perc_abs.col:7.2f} {frame.p_sonar_abs.col:7.2f} "
                f"{gap:+7.2f} {su:8.2f} {frame.valid_ratio:5.2f}",
                flush=True,
            )

            overlay = _draw_centroid_overlay(
                polar_np,
                sway=sway,
                p_sonar=frame.p_sonar_abs,
                p_perc=frame.p_perc_abs,
                p_fixed=p_fixed,
                valid_ratio=frame.valid_ratio,
            )
            frame_path = polar_dir / f"polar_sway_{sway:+.3f}m.png"
            cv2.imwrite(str(frame_path), overlay)
            if i in montage_indices:
                montage_frames.append((sway, overlay))
        else:
            print(f"{sway:+7.3f}   missing perc or sonar GT", flush=True)

    ok, msg = report.perc_verdict()
    print("\n======== VERDICT ========", flush=True)
    print(f"  Phase 1 PASS={ok}  {msg}", flush=True)
    print("=========================\n", flush=True)

    if sways:
        _save_summary_plot(out_dir, sways=sways, d_cols=d_cols, d_rows=d_rows, track_us=track_us, valids=valids,
                           verdict_ok=ok, verdict_msg=msg)
        _save_montage(out_dir, montage_frames)
        print(f"[save] {len(sways)} polar frames -> {polar_dir.resolve()}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
