#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2: image P-control while target moves 0 → +A → 0 (one round trip)."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Centroid track P-control eval (moving target).")
parser.add_argument("--task", type=str, default="Isaac-SonarCentroidTrack-Debug-Direct-v0")
parser.add_argument("--steps", type=int, default=0, help="Total steps (0=auto for motion).")
parser.add_argument("--deadband", type=float, default=0.003, help="Col-delta deadband (normalized).")
parser.add_argument("--kp", type=float, default=1.0, help="P gain on (green_col − yellow_col)/W.")
parser.add_argument("--kff", type=float, default=1.0, help="Feedforward on target lateral velocity.")
parser.add_argument("--kp_sway", type=float, default=1.0, help="Position P (sway_hybrid only, deprecated).")
parser.add_argument("--max_vy", type=float, default=0.0, help="Robot vy clamp (0=auto: 1.15×target_vy).")
parser.add_argument(
    "--control",
    type=str,
    choices=("image_delta", "sway_hybrid"),
    default="image_delta",
    help="image_delta: pure image P on green−yellow col; sway_hybrid: + world position loop.",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=("discrete", "continuous", "oracle"),
    default="continuous",
    help="discrete: bang-bang; continuous: vy=Kp*u; oracle: geometry p_perc (upper bound).",
)
parser.add_argument(
    "--target_motion",
    type=str,
    choices=("static", "round_trip", "triangle"),
    default="round_trip",
    help="round_trip: 0→+A→0; triangle: 0→+A→−A→0.",
)
parser.add_argument("--sway_amp", type=float, default=1.3, help="Peak |target dy| (m).")
parser.add_argument("--target_vy", type=float, default=0.28, help="Target lateral speed magnitude (m/s).")
parser.add_argument("--calib_settle_steps", type=int, default=8)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/centroid_track/phase2_p_roundtrip",
)
parser.add_argument(
    "--sonar_ui",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="OceanSim sonar viewport (GUI).",
)
parser.add_argument(
    "--step_sleep",
    type=float,
    default=0.0,
    help="Pause between steps in seconds (slow down GUI demo).",
)
parser.add_argument(
    "--tune",
    action="store_true",
    help="Single Isaac session: sweep gain grid and print |d_col| comparison.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.headless:
    args_cli.enable_cameras = True
    args_cli.sonar_ui = False
elif not args_cli.headless:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.sonar_centroid_track.modules.constants import ACTION_STAY
from isaaclab_tasks.direct.sonar_centroid_track.modules.ibvs_p_controller import (
    image_delta_lateral_velocity,
    moving_target_lateral_velocity,
    tracking_error_to_discrete_action,
)
from isaaclab_tasks.utils import parse_env_cfg


def _motion_leg_steps(sway_amp: float, vy: float, step_dt: float) -> int:
    return max(int(round(abs(sway_amp) / max(abs(vy) * step_dt, 1.0e-6))), 1)


def _motion_total_steps(
    target_motion: str, sway_amp: float, vy: float, step_dt: float, fallback: int
) -> int:
    if fallback > 0:
        return fallback
    leg = _motion_leg_steps(sway_amp, vy, step_dt)
    if target_motion == "round_trip":
        return 2 * leg
    if target_motion == "triangle":
        return 4 * leg
    return fallback if fallback > 0 else 240


def _target_vy_motion(
    step: int, *, target_motion: str, sway_amp: float, vy: float, step_dt: float
) -> float:
    leg = _motion_leg_steps(sway_amp, vy, step_dt)
    if target_motion == "round_trip":
        if step < leg:
            return float(vy)
        if step < 2 * leg:
            return float(-vy)
        return 0.0
    if target_motion == "triangle":
        if step < leg:
            return float(vy)
        if step < 3 * leg:
            return float(-vy)
        if step < 4 * leg:
            return float(vy)
        return 0.0
    return 0.0


def _target_dy_motion(
    step: int, *, target_motion: str, sway_amp: float, vy: float, step_dt: float
) -> float:
    dy = 0.0
    for s in range(step):
        dy += _target_vy_motion(
            s, target_motion=target_motion, sway_amp=sway_amp, vy=vy, step_dt=step_dt
        ) * step_dt
    total = _motion_total_steps(target_motion, sway_amp, vy, step_dt, 0)
    if target_motion == "round_trip":
        return float(np.clip(dy, 0.0, sway_amp) if step <= total else 0.0)
    if target_motion == "triangle":
        return float(np.clip(dy, -sway_amp, sway_amp) if step <= total else 0.0)
    return 0.0


def _delta_col_from_frame(frame, p_fixed) -> float | None:
    if frame.p_sonar_abs is None or p_fixed is None:
        return None
    return float(frame.p_sonar_abs.col - p_fixed.col)


def _save_summary(
    out_dir: Path,
    *,
    steps: list[int],
    target_dy: list[float],
    robot_sway: list[float],
    sonar_u: list[float],
    track_d: list[float],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Phase 2 image P-control (moving target)", fontsize=11)

    ax = axes[0, 0]
    ax.plot(steps, target_dy, "-", color="C3", lw=1.5, label="target dy")
    ax.plot(steps, robot_sway, "-", color="C0", lw=1.2, alpha=0.9, label="robot sway")
    ax.set_xlabel("step")
    ax.set_ylabel("lateral (m)")
    ax.set_title("Target vs robot sway")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    err = np.abs(np.array(target_dy) - np.array(robot_sway))
    ax.plot(steps, err, "-", color="C1", lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("|sway − target| (m)")
    ax.set_title("Tracking error")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, sonar_u, "-", color="C2", lw=1.2)
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("step")
    ax.set_ylabel("|green−yellow| col (px)")
    ax.set_title("Image col error (sonar x-axis)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, track_d, "-", color="C4", lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("||p_sonar|| (px)")
    ax.set_title("Fixed-frame sonar distance")
    ax.grid(True, alpha=0.3)

    out = out_dir / "summary_p_control.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[save] summary -> {out.resolve()}", flush=True)


TUNE_GRID: list[dict] = [
    {"tag": "id_kp0.8", "control": "image_delta", "kp": 0.8, "kff": 1.00, "kp_sway": 0.0, "max_vy": 0.0},
    {"tag": "id_kp1.0", "control": "image_delta", "kp": 1.0, "kff": 1.00, "kp_sway": 0.0, "max_vy": 0.0},
    {"tag": "id_kp1.5", "control": "image_delta", "kp": 1.5, "kff": 1.00, "kp_sway": 0.0, "max_vy": 0.0},
    {"tag": "id_kp2.0", "control": "image_delta", "kp": 2.0, "kff": 1.05, "kp_sway": 0.0, "max_vy": 0.36},
    {"tag": "id_ff1.05", "control": "image_delta", "kp": 1.0, "kff": 1.05, "kp_sway": 0.0, "max_vy": 0.0},
]


def _resolve_max_vy(cfg_max: float, target_vy: float) -> float:
    if cfg_max > 0.0:
        return cfg_max
    return max(target_vy * 1.15, 0.12)


def _print_metrics(tag: str, metrics: dict) -> None:
    print(
        f"  [{tag}] |d_col| mean={metrics['d_col_mean']:.2f} p90={metrics['d_col_p90']:.2f} "
        f"max={metrics['d_col_max']:.2f}  d_col_bias={metrics['d_col_bias']:+.2f}  "
        f"|sway| mean={metrics['sway_err_mean']:.3f} max={metrics['sway_err_max']:.3f}",
        flush=True,
    )


def _run_trial(
    env,
    unwrapped,
    *,
    control: str,
    kp: float,
    kff: float,
    kp_sway: float,
    max_vy: float,
    deadband: float,
    target_motion: str,
    sway_amp: float,
    target_vy: float,
    step_dt: float,
    total_steps: int,
    calib_settle_steps: int,
    mode: str,
    step_sleep: float,
    verbose: bool,
) -> dict:
    stay = torch.tensor([ACTION_STAY], device=unwrapped.device, dtype=torch.long)
    env.reset()
    for _ in range(calib_settle_steps):
        unwrapped.set_target_offset(0.0, 0.0)
        env.step(stay)
    unwrapped.recalibrate_fixed_point()
    unwrapped.set_target_offset(0.0, 0.0)

    polar_w = float(unwrapped.cfg.oceansim_sonar_cfg.width)
    delta_col_log: list[float] = []
    signed_col_log: list[float] = []
    sway_errs: list[float] = []
    track_d_log: list[float] = []

    for step in range(total_steps):
        tgt_vy = 0.0
        tgt_dy = 0.0
        if target_motion in ("triangle", "round_trip"):
            tgt_dy = _target_dy_motion(
                step + 1,
                target_motion=target_motion,
                sway_amp=sway_amp,
                vy=target_vy,
                step_dt=step_dt,
            )
            tgt_vy = _target_vy_motion(
                step,
                target_motion=target_motion,
                sway_amp=sway_amp,
                vy=target_vy,
                step_dt=step_dt,
            )
            unwrapped.set_target_offset(0.0, tgt_dy)

        unwrapped.ensure_tracking()
        frame = unwrapped.get_centroid_frame(0)
        p_fixed = unwrapped.fixed.p_fixed_abs_env(0)
        delta_col = _delta_col_from_frame(frame, p_fixed)
        delta_norm = float(delta_col) / polar_w if delta_col is not None else 0.0
        sway_pre = float(unwrapped._robot_sway_m()[0].item())
        sway_err = tgt_dy - sway_pre if target_motion in ("triangle", "round_trip") else 0.0

        if mode == "discrete":
            action = tracking_error_to_discrete_action(delta_norm, deadband=deadband)
            env.step(torch.tensor([action], device=unwrapped.device, dtype=torch.long))
        elif control == "sway_hybrid":
            rvy = moving_target_lateral_velocity(
                delta_norm, tgt_vy, sway_err,
                kff=kff, kp_img=kp, kp_sway=kp_sway, max_vy=max_vy, deadband_img=deadband,
            )
            unwrapped.set_lateral_velocity_override(
                torch.tensor([rvy], device=unwrapped.device, dtype=torch.float32)
            )
            env.step(stay)
            unwrapped.set_lateral_velocity_override(None)
        else:
            rvy = image_delta_lateral_velocity(
                delta_norm, tgt_vy, kp=kp, kff=kff, max_vy=max_vy, deadband=deadband,
            )
            unwrapped.set_lateral_velocity_override(
                torch.tensor([rvy], device=unwrapped.device, dtype=torch.float32)
            )
            env.step(stay)
            unwrapped.set_lateral_velocity_override(None)

        if step_sleep > 0.0:
            time.sleep(step_sleep)

        sway = float(unwrapped._robot_sway_m()[0].item())
        tgt = float(unwrapped._target_dy_m()[0].item())
        if delta_col is not None:
            delta_col_log.append(abs(delta_col))
            signed_col_log.append(delta_col)
        track_d = frame.sonar_origin_dist
        if track_d is not None:
            track_d_log.append(track_d)
        sway_errs.append(abs(sway - tgt))

        if verbose and (step % max(total_steps // 8, 1) == 0 or step == total_steps - 1):
            dc = abs(delta_col) if delta_col is not None else float("nan")
            print(
                f"    step={step:4d} tgt={tgt:+6.3f} sway={sway:+6.3f} d_col={dc:6.2f}",
                flush=True,
            )

    dc = np.array(delta_col_log, dtype=np.float64)
    sc = np.array(signed_col_log, dtype=np.float64)
    se = np.array(sway_errs, dtype=np.float64)
    td = np.array(track_d_log, dtype=np.float64)
    return {
        "d_col_mean": float(dc.mean()) if len(dc) else float("nan"),
        "d_col_p90": float(np.percentile(dc, 90)) if len(dc) else float("nan"),
        "d_col_max": float(dc.max()) if len(dc) else float("nan"),
        "d_col_bias": float(sc.mean()) if len(sc) else float("nan"),
        "sway_err_mean": float(se.mean()) if len(se) else float("nan"),
        "sway_err_max": float(se.max()) if len(se) else float("nan"),
        "track_mean": float(td.mean()) if len(td) else float("nan"),
    }


def _run_tune(env, unwrapped, step_dt: float, total_steps: int, out_dir: Path) -> None:
    print("\n=== P-control tune (single session) ===\n", flush=True)
    print(
        f"  target_vy={args_cli.target_vy}  sway_amp={args_cli.sway_amp}  steps={total_steps}",
        flush=True,
    )
    print(f"{'tag':<12} {'ctrl':<12} {'kp':>5} {'kff':>5} {'kps':>5} |d_col|_mean p90  sway_err", flush=True)

    results: list[tuple[dict, dict]] = []
    for cfg in TUNE_GRID:
        tag = cfg["tag"]
        max_vy = _resolve_max_vy(float(cfg.get("max_vy", 0.0)), float(args_cli.target_vy))
        print(f"\n--- {tag} control={cfg['control']} kp={cfg['kp']} kff={cfg['kff']} kp_sway={cfg['kp_sway']} ---", flush=True)
        metrics = _run_trial(
            env, unwrapped,
            control=str(cfg["control"]),
            kp=float(cfg["kp"]),
            kff=float(cfg["kff"]),
            kp_sway=float(cfg["kp_sway"]),
            max_vy=max_vy,
            deadband=float(args_cli.deadband),
            target_motion=str(args_cli.target_motion),
            sway_amp=float(args_cli.sway_amp),
            target_vy=float(args_cli.target_vy),
            step_dt=step_dt,
            total_steps=total_steps,
            calib_settle_steps=int(args_cli.calib_settle_steps),
            mode=str(args_cli.mode),
            step_sleep=float(args_cli.step_sleep),
            verbose=False,
        )
        results.append((cfg, metrics))
        print(
            f"{tag:<12} {cfg['control']:<12} {cfg['kp']:5.2f} {cfg['kff']:5.2f} {cfg['kp_sway']:5.2f} "
            f"{metrics['d_col_mean']:6.2f} {metrics['d_col_p90']:5.2f}  "
            f"{metrics['sway_err_mean']:.3f}/{metrics['sway_err_max']:.3f}",
            flush=True,
        )

    best_cfg, best_m = min(results, key=lambda x: (x[1]["d_col_mean"], x[1]["sway_err_mean"]))
    print("\n=== BEST (min |d_col| mean) ===", flush=True)
    print(
        f"  tag={best_cfg['tag']}  control={best_cfg['control']}  "
        f"kp={best_cfg['kp']}  kff={best_cfg['kff']}  kp_sway={best_cfg['kp_sway']}",
        flush=True,
    )
    _print_metrics(best_cfg["tag"], best_m)

    # Re-run best with verbose step log + plot
    print("\n=== BEST re-run (verbose) ===", flush=True)
    best_max_vy = _resolve_max_vy(float(best_cfg.get("max_vy", 0.0)), float(args_cli.target_vy))
    best_metrics = _run_trial(
        env, unwrapped,
        control=str(best_cfg["control"]),
        kp=float(best_cfg["kp"]),
        kff=float(best_cfg["kff"]),
        kp_sway=float(best_cfg["kp_sway"]),
        max_vy=best_max_vy,
        deadband=float(args_cli.deadband),
        target_motion=str(args_cli.target_motion),
        sway_amp=float(args_cli.sway_amp),
        target_vy=float(args_cli.target_vy),
        step_dt=step_dt,
        total_steps=total_steps,
        calib_settle_steps=int(args_cli.calib_settle_steps),
        mode=str(args_cli.mode),
        step_sleep=float(args_cli.step_sleep),
        verbose=True,
    )
    _print_metrics(str(best_cfg["tag"]), best_metrics)
    tune_out = out_dir / "tune_results.txt"
    with tune_out.open("w") as f:
        f.write("tag,control,kp,kff,kp_sway,d_col_mean,d_col_p90,d_col_max,sway_err_mean,sway_err_max\n")
        for cfg, m in results:
            f.write(
                f"{cfg['tag']},{cfg['control']},{cfg['kp']},{cfg['kff']},{cfg['kp_sway']},"
                f"{m['d_col_mean']:.4f},{m['d_col_p90']:.4f},{m['d_col_max']:.4f},"
                f"{m['sway_err_mean']:.4f},{m['sway_err_max']:.4f}\n"
            )
        f.write(f"\nBEST,{best_cfg['tag']},{best_cfg['control']},{best_cfg['kp']},...\n")
    print(f"[save] tune table -> {tune_out.resolve()}", flush=True)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.freeze_robot_pose = False
    env_cfg.lateral_target_moving = False
    env_cfg.spawn_radius = 0.0
    env_cfg.eval_spawn_radius = 0.0
    env_cfg.print_centroid_debug = False
    env_cfg.debug_disable_lost_terminate = True
    if args_cli.sonar_ui and hasattr(env_cfg, "oceansim_sonar_cfg"):
        env_cfg.oceansim_sonar_cfg.sonar_initialize_viewport = True
    max_vy = float(args_cli.max_vy)
    if max_vy <= 0.0:
        max_vy = max(float(args_cli.target_vy) * 1.15, 0.12)
    if args_cli.mode == "continuous":
        env_cfg.max_lateral_velocity = max_vy

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    step_dt = float(unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)
    total_steps = _motion_total_steps(
        args_cli.target_motion,
        float(args_cli.sway_amp),
        float(args_cli.target_vy),
        step_dt,
        int(args_cli.steps),
    )

    if args_cli.tune:
        _run_tune(env, unwrapped, step_dt, total_steps, out_dir)
        env.close()
        simulation_app.close()
        return

    env.reset()

    stay = torch.tensor([ACTION_STAY], device=unwrapped.device, dtype=torch.long)
    for _ in range(int(args_cli.calib_settle_steps)):
        unwrapped.set_target_offset(0.0, 0.0)
        env.step(stay)
    unwrapped.recalibrate_fixed_point()
    unwrapped.set_target_offset(0.0, 0.0)

    leg = _motion_leg_steps(float(args_cli.sway_amp), float(args_cli.target_vy), step_dt)

    print("\n=== Phase 2 image P-control (moving target) ===\n", flush=True)
    a = float(args_cli.sway_amp)
    if args_cli.target_motion == "round_trip":
        print(
            f"  target: 0 → +{a:.2f} → 0 m  "
            f"@ vy={args_cli.target_vy} m/s  leg={leg} steps  total={total_steps}",
            flush=True,
        )
    elif args_cli.target_motion == "triangle":
        print(
            f"  target: 0 → +{a:.2f} → −{a:.2f} → 0 m  "
            f"@ vy={args_cli.target_vy} m/s  leg={leg} steps  total={total_steps}",
            flush=True,
        )
    print(
        f"  control={args_cli.control}  kff={args_cli.kff}  kp={args_cli.kp}  max_vy={max_vy:.2f}",
        flush=True,
    )
    if args_cli.control == "image_delta":
        print("  law: vy = kff·target_vy + kp·(green_col − yellow_col)/W", flush=True)
    print(f"  output_dir: {out_dir.resolve()}", flush=True)
    if not args_cli.headless:
        print("  GUI: Isaac viewport + OceanSim sonar window (close sim to exit).", flush=True)
    print("", flush=True)
    if args_cli.headless:
        print(f"{'step':>5} {'d_col':>8} {'|d_col|':>8}", flush=True)
    else:
        print(f"{'step':>5} {'tgt_dy':>7} {'sway':>7} {'|err|':>7} {'d_col':>7} {'||sn||':>7} {'rvy':>7}", flush=True)

    steps_log: list[int] = []
    target_dy_log: list[float] = []
    sway_log: list[float] = []
    delta_col_log: list[float] = []
    signed_col_log: list[float] = []
    track_d_log: list[float] = []
    sway_errs: list[float] = []

    polar_w = float(unwrapped.cfg.oceansim_sonar_cfg.width)

    for step in range(total_steps):
        tgt_vy = 0.0
        tgt_dy = 0.0
        if args_cli.target_motion in ("triangle", "round_trip"):
            tgt_dy = _target_dy_motion(
                step + 1,
                target_motion=args_cli.target_motion,
                sway_amp=float(args_cli.sway_amp),
                vy=float(args_cli.target_vy),
                step_dt=step_dt,
            )
            tgt_vy = _target_vy_motion(
                step,
                target_motion=args_cli.target_motion,
                sway_amp=float(args_cli.sway_amp),
                vy=float(args_cli.target_vy),
                step_dt=step_dt,
            )
            unwrapped.set_target_offset(0.0, tgt_dy)

        unwrapped.ensure_tracking()
        frame = unwrapped.get_centroid_frame(0)
        p_fixed = unwrapped.fixed.p_fixed_abs_env(0)
        delta_col = _delta_col_from_frame(frame, p_fixed)
        delta_norm = float(delta_col) / polar_w if delta_col is not None else 0.0
        sway_pre = float(unwrapped._robot_sway_m()[0].item())
        sway_err = tgt_dy - sway_pre if args_cli.target_motion in ("triangle", "round_trip") else 0.0

        if args_cli.mode == "discrete":
            action = tracking_error_to_discrete_action(delta_norm, deadband=args_cli.deadband)
            act_t = torch.tensor([action], device=unwrapped.device, dtype=torch.long)
            unwrapped.set_lateral_velocity_override(None)
            env.step(act_t)
            rvy = float("nan")
        elif args_cli.control == "sway_hybrid":
            rvy = moving_target_lateral_velocity(
                delta_norm,
                tgt_vy,
                sway_err,
                kff=float(args_cli.kff),
                kp_img=float(args_cli.kp),
                kp_sway=float(args_cli.kp_sway),
                max_vy=max_vy,
                deadband_img=float(args_cli.deadband),
            )
            unwrapped.set_lateral_velocity_override(
                torch.tensor([rvy], device=unwrapped.device, dtype=torch.float32)
            )
            env.step(stay)
            unwrapped.set_lateral_velocity_override(None)
        else:
            rvy = image_delta_lateral_velocity(
                delta_norm,
                tgt_vy,
                kp=float(args_cli.kp),
                kff=float(args_cli.kff),
                max_vy=max_vy,
                deadband=float(args_cli.deadband),
            )
            unwrapped.set_lateral_velocity_override(
                torch.tensor([rvy], device=unwrapped.device, dtype=torch.float32)
            )
            env.step(stay)
            unwrapped.set_lateral_velocity_override(None)

        if float(args_cli.step_sleep) > 0.0:
            time.sleep(float(args_cli.step_sleep))

        sway = float(unwrapped._robot_sway_m()[0].item())
        tgt = float(unwrapped._target_dy_m()[0].item())
        sonar_u = abs(delta_col) if delta_col is not None else float("nan")
        signed_col = float(delta_col) if delta_col is not None else float("nan")
        track_d = frame.sonar_origin_dist if frame.sonar_origin_dist is not None else float("nan")
        err = abs(sway - tgt)

        steps_log.append(step)
        target_dy_log.append(tgt)
        sway_log.append(sway)
        delta_col_log.append(sonar_u)
        signed_col_log.append(signed_col)
        track_d_log.append(track_d if track_d is not None else float("nan"))
        sway_errs.append(err)

        if args_cli.headless:
            print(f"{step:5d} {signed_col:+8.2f} {sonar_u:8.2f}", flush=True)
        elif step % max(total_steps // 12, 1) == 0 or step == total_steps - 1:
            rvy_s = f"{rvy:+7.3f}" if args_cli.mode != "discrete" else "    nan"
            print(
                f"{step:5d} {tgt:+7.3f} {sway:+7.3f} {err:7.3f} {sonar_u:7.2f} "
                f"{track_d if track_d is not None else float('nan'):7.2f} {rvy_s}",
                flush=True,
            )

    _save_summary(
        out_dir,
        steps=steps_log,
        target_dy=target_dy_log,
        robot_sway=sway_log,
        sonar_u=delta_col_log,
        track_d=track_d_log,
    )

    print("\n=== Phase 2 P-control summary ===", flush=True)
    print(f"  control={args_cli.control}  mode={args_cli.mode}", flush=True)
    dc = np.array(delta_col_log, dtype=np.float64)
    dc = dc[np.isfinite(dc)]
    sc = np.array(signed_col_log, dtype=np.float64)
    sc = sc[np.isfinite(sc)]
    if len(dc):
        print(
            f"  |d_col| (green−yellow) px: mean={dc.mean():.2f}  p90={np.percentile(dc, 90):.2f}  max={dc.max():.2f}",
            flush=True,
        )
        print(f"  d_col signed mean={sc.mean():+.2f} px  (bias: + = green right of yellow)", flush=True)
    td = np.array(track_d_log, dtype=np.float64)
    td = td[np.isfinite(td)]
    if len(td):
        print(f"  ||p_sonar|| mean={td.mean():.2f} px  p90={np.percentile(td, 90):.2f} px", flush=True)
    se = np.array(sway_errs, dtype=np.float64)
    print(f"  |sway−target| mean={se.mean():.3f} m  max={se.max():.3f} m", flush=True)
    print(f"  sway range [{min(sway_log):+.3f}, {max(sway_log):+.3f}] m\n", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
