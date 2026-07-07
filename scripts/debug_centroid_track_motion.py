#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Motion debug: image centroid track under real lateral velocity."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Centroid track — image detection under real robot motion.")
parser.add_argument("--task", type=str, default="Isaac-SonarCentroidTrack-Debug-Direct-v0")
parser.add_argument("--steps", type=int, default=0, help="Total steps (0=auto for triangle/ramp).")
parser.add_argument("--vy", type=float, default=0.12, help="Body-Y velocity magnitude (m/s).")
parser.add_argument("--sway_amp", type=float, default=1.3, help="Peak |sway| for triangle motion (m).")
parser.add_argument(
    "--motion",
    type=str,
    choices=("triangle", "ramp", "sine"),
    default="triangle",
    help="triangle: 0→+A→−A→0; ramp: 0→+A→0; sine: vy*sin(2pi t/T).",
)
parser.add_argument("--period_s", type=float, default=16.0, help="Sine motion period (s).")
parser.add_argument("--save_every", type=int, default=20, help="Save polar overlay every N steps.")
parser.add_argument(
    "--calib_settle_steps",
    type=int,
    default=8,
    help="Physics settle steps after reset before re-calibrating fixed point.",
)
parser.add_argument("--output_dir", type=str, default="logs/centroid_track/phase1_motion_track")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.sonar_centroid_track.modules.constants import ACTION_STAY
from isaaclab_tasks.utils import parse_env_cfg


def _polar_to_u8(polar: np.ndarray) -> np.ndarray:
    x = polar.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi > lo:
        x = (x - lo) / (hi - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def _draw_overlay(polar: np.ndarray, frame, p_fixed) -> np.ndarray:
    gray = _polar_to_u8(polar)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _dot(pt, color: tuple[int, int, int]) -> None:
        if pt is None:
            return
        c, r = int(round(pt.col)), int(round(pt.row))
        cv2.circle(img, (c, r), 1, color, -1, lineType=cv2.LINE_AA)

    _dot(p_fixed, (0, 255, 255))
    _dot(frame.p_sonar_abs, (0, 255, 0))
    _dot(frame.p_perc_abs, (255, 0, 0))
    return img


def _triangle_leg_steps(sway_amp: float, vy: float, step_dt: float) -> int:
    return max(int(round(abs(sway_amp) / max(abs(vy) * step_dt, 1.0e-6))), 1)


def _motion_total_steps(motion: str, *, sway_amp: float, vy: float, step_dt: float, fallback: int) -> int:
    if motion == "triangle":
        leg = _triangle_leg_steps(sway_amp, vy, step_dt)
        return 4 * leg
    if motion == "ramp":
        leg = _triangle_leg_steps(sway_amp, vy, step_dt)
        return 2 * leg
    return fallback if fallback > 0 else 240


def _vy_cmd(
    step: int,
    *,
    motion: str,
    vy: float,
    sway_amp: float,
    period_s: float,
    step_dt: float,
    total_steps: int,
) -> float:
    if motion == "sine":
        omega = 2.0 * np.pi / max(period_s, 1.0e-3)
        t = step * step_dt
        return float(vy * np.sin(omega * t))
    if motion == "triangle":
        leg = _triangle_leg_steps(sway_amp, vy, step_dt)
        if step < leg:
            return float(vy)
        if step < 3 * leg:
            return float(-vy)
        if step < 4 * leg:
            return float(vy)
        return 0.0
    half = max(total_steps // 2, 1)
    return float(vy if step < half else -vy)


def _save_summary(
    out_dir: Path,
    *,
    steps: list[int],
    sways: list[float],
    track_perc: list[float],
    sonar_u: list[float],
    valids: list[float],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Motion debug (sonar bright-spot, real vy)", fontsize=11)

    ax = axes[0, 0]
    ax.plot(sways, track_perc, "o-", ms=3, color="C1", label="|sonar−perc| col")
    ax.set_xlabel("robot sway (m)")
    ax.set_ylabel("px")
    ax.set_title("Sonar vs geometry (debug)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, sonar_u, "o-", ms=3, color="C0", label="|p_sonar.u|")
    ax.set_xlabel("step")
    ax.set_ylabel("px (fixed frame)")
    ax.set_title("Sonar lateral error vs fixed point")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, valids, "o-", ms=3, color="C2", label="detected")
    ax.axhline(0.25, color="r", ls="--", lw=1, label="MIN_VALID_RATIO=0.25")
    ax.set_xlabel("step")
    ax.set_ylabel("detection ok (0/1)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.05, 0.6, "green = sonar bright-spot (obs/reward)", fontsize=10)
    ax.text(0.05, 0.4, "blue = geometry (debug only)", fontsize=10)

    out = out_dir / "summary_motion_track.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[save] summary -> {out.resolve()}", flush=True)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.freeze_robot_pose = False
    env_cfg.lateral_target_moving = False
    env_cfg.spawn_radius = 0.0
    env_cfg.eval_spawn_radius = 0.0
    env_cfg.print_centroid_debug = False
    env_cfg.debug_disable_lost_terminate = True
    env_cfg.max_lateral_velocity = max(abs(float(args_cli.vy)), 1.0e-3)

    out_dir = Path(args_cli.output_dir)
    frames_dir = out_dir / "polar_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset()

    stay = torch.tensor([ACTION_STAY], device=unwrapped.device, dtype=torch.long)
    for _ in range(int(args_cli.calib_settle_steps)):
        env.step(stay)
    unwrapped.recalibrate_fixed_point()
    p_fixed0 = unwrapped.fixed.p_fixed_abs_env(0)
    frame0 = unwrapped.get_centroid_frame(0)
    if p_fixed0 is not None and frame0.p_sonar_abs is not None:
        gap0 = math.hypot(p_fixed0.col - frame0.p_sonar_abs.col, p_fixed0.row - frame0.p_sonar_abs.row)
        print(
            f"  fixed-point teach: ({p_fixed0.col:.1f},{p_fixed0.row:.1f}) "
            f"sonar=({frame0.p_sonar_abs.col:.1f},{frame0.p_sonar_abs.row:.1f}) gap={gap0:.2f}px",
            flush=True,
        )

    step_dt = float(unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)
    total_steps = _motion_total_steps(
        args_cli.motion,
        sway_amp=float(args_cli.sway_amp),
        vy=float(args_cli.vy),
        step_dt=step_dt,
        fallback=int(args_cli.steps),
    )
    leg_steps = _triangle_leg_steps(float(args_cli.sway_amp), float(args_cli.vy), step_dt)

    print("\n=== Motion track debug (image centroid, real vy) ===\n", flush=True)
    if args_cli.motion == "triangle":
        a = float(args_cli.sway_amp)
        print(
            f"  trajectory: 0 → +{a:.2f} → −{a:.2f} → 0 m  "
            f"(leg={leg_steps} steps, vy=±{args_cli.vy}, dt={step_dt:.3f}s)",
            flush=True,
        )
    else:
        print(f"  motion={args_cli.motion} vy={args_cli.vy} steps={total_steps} dt={step_dt:.3f}s", flush=True)
    print(f"  output_dir: {out_dir.resolve()}", flush=True)

    steps_log: list[int] = []
    sways_log: list[float] = []
    track_perc_log: list[float] = []
    sonar_u_log: list[float] = []
    valids_log: list[float] = []
    montage: list[tuple[int, float, np.ndarray]] = []

    print(f"  perception: SonarGtDetector bright-spot only", flush=True)
    print(f"{'step':>5} {'sway':>7} {'det':>5} {'|sn-pr|':>7} {'|sn.u|':>7} {'vy':>7}", flush=True)

    for step in range(total_steps):
        vy = _vy_cmd(
            step,
            motion=args_cli.motion,
            vy=float(args_cli.vy),
            sway_amp=float(args_cli.sway_amp),
            period_s=float(args_cli.period_s),
            step_dt=step_dt,
            total_steps=total_steps,
        )
        unwrapped.set_lateral_velocity_override(torch.tensor([vy], device=unwrapped.device, dtype=torch.float32))
        env.step(stay)
        unwrapped.set_lateral_velocity_override(None)

        frame = unwrapped.get_centroid_frame(0)
        sway = float(unwrapped._robot_sway_m()[0].item())
        p_fixed = unwrapped.fixed.p_fixed_abs_env(0)

        d_sp = float("nan")
        sonar_u = float("nan")
        if frame.p_sonar_abs is not None and frame.p_perc_abs is not None:
            d_sp = abs(frame.p_sonar_abs.col - frame.p_perc_abs.col)
        if frame.p_sonar is not None:
            sonar_u = abs(frame.p_sonar.col)

        steps_log.append(step)
        sways_log.append(sway)
        track_perc_log.append(d_sp)
        sonar_u_log.append(sonar_u)
        valids_log.append(frame.valid_ratio)

        det_s = "1" if frame.valid_ratio >= 0.5 else "0"
        if step % args_cli.save_every == 0 or step == total_steps - 1:
            polar_np = unwrapped.polar_image[0].detach().cpu().numpy()
            overlay = _draw_overlay(polar_np, frame, p_fixed)
            path = frames_dir / f"step_{step:04d}_sway_{sway:+.3f}m_det{det_s}.png"
            cv2.imwrite(str(path), overlay)
            if len(montage) < 5:
                montage.append((step, sway, overlay))
            print(
                f"{step:5d} {sway:+7.3f} {det_s:>5} {d_sp:7.2f} {sonar_u:7.2f} {vy:+7.3f}",
                flush=True,
            )
            if frame.valid_ratio < 0.25:
                print(f"       ⚠ ball not detected on polar frame", flush=True)

    _save_summary(
        out_dir,
        steps=steps_log,
        sways=sways_log,
        track_perc=track_perc_log,
        sonar_u=sonar_u_log,
        valids=valids_log,
    )

    if montage:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(montage), figsize=(3.2 * len(montage), 3))
        for ax, (st, sw, bgr) in zip(np.atleast_1d(axes), montage):
            ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), aspect="auto")
            ax.set_title(f"step={st} sway={sw:+.2f}m", fontsize=9)
            ax.axis("off")
        fig.suptitle("Motion montage (yellow=fix green=sonar blue=perc)", fontsize=10)
        mp = out_dir / "polar_montage_motion.png"
        fig.tight_layout()
        fig.savefig(mp, dpi=150)
        plt.close(fig)
        print(f"[save] montage -> {mp.resolve()}", flush=True)

    tp = np.array(track_perc_log, dtype=np.float64)
    tp = tp[np.isfinite(tp)]
    print("\n=== Summary ===", flush=True)
    if len(tp):
        print(f"  |sonar−perc| col: mean={tp.mean():.2f} px  p90={np.percentile(tp, 90):.2f} px", flush=True)
    det = np.array(valids_log)
    print(
        f"  detection rate: {det.mean():.2f}  missed={int((det < 0.5).sum())}/{len(det)} steps",
        flush=True,
    )
    print(f"  sway range: [{min(sways_log):+.3f}, {max(sways_log):+.3f}] m\n", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
