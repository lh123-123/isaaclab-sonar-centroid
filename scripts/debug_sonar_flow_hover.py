#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive GUI debug for Sonar IBVS-DRL (OceanSim viewport + feature tracking logs)."""

from __future__ import annotations

import argparse
import math
import os
import random
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GUI debug: sonar flow hover with OceanSim viewport.")
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Debug-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--sonar_ui",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable OceanSim sonar viewport (default: on).",
)
parser.add_argument(
    "--freeze_robot",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Hold AUV at hover pose for pure sonar debug.",
)
parser.add_argument(
    "--save_sonar",
    action="store_true",
    default=False,
    help="Save sonar PNG every debug interval to logs/sonar_flow_hover_debug/.",
)
parser.add_argument(
    "--offset_dx",
    type=float,
    default=0.0,
    help="Initial world-frame X offset from hover (m).",
)
parser.add_argument(
    "--offset_dy",
    type=float,
    default=0.0,
    help="Initial world-frame Y offset from hover (m).",
)
parser.add_argument(
    "--tracking_mode",
    type=str,
    choices=("patch", "lk"),
    default="lk",
    help="Feature tracker: lk=classic optical flow, patch=teach patch NCC.",
)
parser.add_argument(
    "--sweep_left",
    action="store_true",
    default=False,
    help="Smooth target motion along -X (AUV fixed).",
)
parser.add_argument(
    "--left_travel_m",
    type=float,
    default=3.0,
    help="Total body-frame left travel (m, +Y) for --sweep_left.",
)
parser.add_argument(
    "--left_duration_s",
    type=float,
    default=10.0,
    help="Duration (s) for --sweep_left ramp.",
)
parser.add_argument(
    "--sweep_random",
    action="store_true",
    default=False,
    help="Random smooth motion in AUV body Y-Z plane (left/right + up/down).",
)
parser.add_argument(
    "--random_y_m",
    type=float,
    default=2.0,
    help="Body-frame lateral amplitude (m) for --sweep_random.",
)
parser.add_argument(
    "--random_z_m",
    type=float,
    default=1.0,
    help="Body-frame vertical amplitude (m) for --sweep_random.",
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
    help="RNG seed for --sweep_random trajectory.",
)
parser.add_argument(
    "--smooth_target",
    action="store_true",
    default=False,
    help="Slow circular target motion (AUV fixed). Overrides --auto_sweep jumps.",
)
parser.add_argument(
    "--target_radius",
    type=float,
    default=0.15,
    help="Radius (m) for --smooth_target circular path.",
)
parser.add_argument(
    "--target_period_s",
    type=float,
    default=12.0,
    help="Full circle period (s) for --smooth_target.",
)
parser.add_argument(
    "--sweep_target",
    action="store_true",
    default=False,
    help="Sweep target position (AUV fixed). Default sweep moves AUV offset.",
)
parser.add_argument(
    "--auto_sweep",
    action="store_true",
    default=False,
    help="Cycle small displacements every --sweep_interval_s (for UI inspection).",
)
parser.add_argument(
    "--sweep_interval_steps",
    type=int,
    default=0,
    help="Steps per sweep offset (overrides --sweep_interval_s when > 0; default 12 if --no-real-time).",
)
parser.add_argument(
    "--sweep_interval_s",
    type=float,
    default=4.0,
    help="Seconds to hold each offset when --auto_sweep is set (wall clock; use --sweep_interval_steps for headless).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Exit after N steps (0 = run until window closed).",
)
parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Sleep to match simulation step_dt (default: on).",
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

ROBOT_SWEEP: list[tuple[float, float, str]] = [
    (0.00, 0.00, "origin"),
    (0.25, 0.00, "+x 0.25"),
    (0.00, 0.25, "+y 0.25"),
    (-0.25, 0.00, "-x 0.25"),
    (0.00, -0.25, "-y 0.25"),
    (0.15, 0.15, "diag 0.15"),
]

TARGET_SWEEP: list[tuple[float, float, str]] = [
    (0.00, 0.00, "origin"),
    (0.15, 0.00, "tgt +x 0.15"),
    (0.00, 0.15, "tgt +y 0.15"),
    (-0.15, 0.00, "tgt -x 0.15"),
    (0.00, -0.15, "tgt -y 0.15"),
    (0.10, 0.10, "tgt diag 0.10"),
]


class _RandomYzWalk:
    """Velocity-limited random walk in body Y-Z (smooth, bounded)."""

    def __init__(self, rng: random.Random, amp_y: float, amp_z: float, dt: float) -> None:
        self.rng = rng
        self.amp_y = max(amp_y, 1.0e-6)
        self.amp_z = max(amp_z, 1.0e-6)
        self.dt = max(dt, 1.0e-4)
        self.dy = 0.0
        self.dz = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.max_vy = self.amp_y / 2.5
        self.max_vz = self.amp_z / 2.5

    def step(self) -> tuple[float, float, float]:
        self.vy += self.rng.uniform(-0.22, 0.22)
        self.vz += self.rng.uniform(-0.14, 0.14)
        self.vy *= 0.88
        self.vz *= 0.88
        self.vy = max(-self.max_vy, min(self.max_vy, self.vy))
        self.vz = max(-self.max_vz, min(self.max_vz, self.vz))
        self.dy = max(-self.amp_y, min(self.amp_y, self.dy + self.vy * self.dt))
        self.dz = max(-self.amp_z, min(self.amp_z, self.dz + self.vz * self.dt))
        if abs(self.dy) >= self.amp_y * 0.97:
            self.vy *= -0.45
        if abs(self.dz) >= self.amp_z * 0.97:
            self.vz *= -0.45
        return 0.0, self.dy, self.dz


def _maybe_save_sonar(env, step: int, out_dir: str) -> None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return
    img = env.cart_image[0].detach().cpu().numpy()
    if img.size == 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    arr = img
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(out_dir, f"sonar_{step:05d}.png"))


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.sweep_target or args_cli.smooth_target or args_cli.sweep_left or args_cli.sweep_random:
        env_cfg.freeze_robot_pose = True
    elif args_cli.freeze_robot is not None:
        env_cfg.freeze_robot_pose = bool(args_cli.freeze_robot)
    if args_cli.sonar_ui and hasattr(env_cfg, "oceansim_sonar_cfg"):
        env_cfg.oceansim_sonar_cfg.sonar_initialize_viewport = True
    # Use taught snapshot during sweep debug (capture mode re-extracts only at reset).
    if args_cli.auto_sweep or args_cli.smooth_target or args_cli.sweep_left or args_cli.sweep_random or args_cli.offset_dx != 0.0 or args_cli.offset_dy != 0.0:
        env_cfg.reference_mode = "file"
    env_cfg.print_ibvs_debug = True
    env_cfg.ibvs_debug_interval = 10
    env_cfg.feature_tracking_mode = args_cli.tracking_mode

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    dt = getattr(env, "step_dt", unwrapped.step_dt)
    save_dir = os.path.abspath("logs/sonar_flow_hover_debug")

    sweep_offsets = TARGET_SWEEP if args_cli.sweep_target else ROBOT_SWEEP
    sweep_mode = "target" if args_cli.sweep_target else "robot"
    if args_cli.smooth_target:
        sweep_mode = "smooth_target"
    elif args_cli.sweep_left:
        sweep_mode = "left_smooth"
    elif args_cli.sweep_random:
        sweep_mode = "random_yz"

    sweep_step_interval = args_cli.sweep_interval_steps
    if args_cli.auto_sweep and sweep_step_interval <= 0 and not args_cli.real_time:
        sweep_step_interval = 12

    omega = 2.0 * math.pi / max(args_cli.target_period_s, 1.0e-3)
    left_travel = max(args_cli.left_travel_m, 0.0)
    left_duration = max(args_cli.left_duration_s, 1.0e-3)
    left_total_steps = max(1, int(round(left_duration / dt)))

    print("\n======== Sonar IBVS-DRL GUI Debug ========")
    print(f"  task         : {args_cli.task}")
    print(f"  sonar_ui     : {args_cli.sonar_ui}")
    print(f"  freeze_robot : {env_cfg.freeze_robot_pose}")
    print(f"  sweep_mode   : {sweep_mode}")
    print(f"  tracking_mode: {args_cli.tracking_mode}")
    print(f"  save_sonar   : {args_cli.save_sonar} -> {save_dir}")
    print(f"  offset       : dx={args_cli.offset_dx:+.2f} dy={args_cli.offset_dy:+.2f}")
    if args_cli.sweep_left:
        print(
            f"  sweep_left   : body_left=+{left_travel:.2f}m duration={left_duration:.1f}s "
            f"(~{left_travel / left_duration:.3f} m/s)"
        )
    elif args_cli.sweep_random:
        print(
            f"  sweep_random : body Y-Z walk amp_y=±{args_cli.random_y_m:.2f}m "
            f"amp_z=±{args_cli.random_z_m:.2f}m seed={args_cli.random_seed}"
        )
    elif args_cli.smooth_target:
        lin_speed = args_cli.target_radius * omega
        print(
            f"  smooth_target: R={args_cli.target_radius:.2f}m "
            f"period={args_cli.target_period_s:.1f}s (~{lin_speed:.3f} m/s)"
        )
    elif args_cli.auto_sweep:
        if sweep_step_interval > 0:
            print(f"  auto_sweep   : True (interval={sweep_step_interval} steps)")
        else:
            print(f"  auto_sweep   : True (interval={args_cli.sweep_interval_s}s)")
    else:
        print("  auto_sweep   : False")
    print(f"  max_steps    : {args_cli.max_steps or 'until close'}")
    print("  Controls: zero-action stepping; close Isaac Sim window to exit.")
    print("==========================================\n")

    def _apply_sweep_offset(dx: float, dy: float, dz: float = 0.0) -> None:
        if args_cli.smooth_target or args_cli.sweep_left or args_cli.sweep_random:
            unwrapped.set_smooth_target_offset(dx, dy, dz)
        elif args_cli.sweep_target:
            unwrapped.set_displacement_offset(0.0, 0.0)
            unwrapped.set_target_offset(dx, dy)
        else:
            unwrapped.set_target_offset(0.0, 0.0)
            unwrapped.set_displacement_offset(dx, dy)

    env.reset()
    if args_cli.sweep_target or args_cli.smooth_target or args_cli.sweep_left or args_cli.sweep_random:
        unwrapped.set_displacement_offset(0.0, 0.0)
        unwrapped.set_target_velocity(0.0, 0.0)
    _apply_sweep_offset(args_cli.offset_dx, args_cli.offset_dy)

    random_traj: _RandomYzWalk | None = None
    if args_cli.sweep_random:
        random_traj = _RandomYzWalk(
            random.Random(args_cli.random_seed),
            args_cli.random_y_m,
            args_cli.random_z_m,
            dt,
        )

    sweep_idx = 0
    sweep_t0 = time.time()
    steps_since_sweep = 0
    current_label = "manual"
    if args_cli.auto_sweep and not args_cli.smooth_target and not args_cli.sweep_left and not args_cli.sweep_random:
        dx, dy, current_label = sweep_offsets[0]
        _apply_sweep_offset(dx, dy)
        print(f"[debug] auto_sweep ({sweep_mode}) -> {current_label} dx={dx:+.2f} dy={dy:+.2f}", flush=True)
    elif args_cli.smooth_target:
        current_label = "smooth_circle"
    elif args_cli.sweep_left:
        current_label = "left_smooth"
    elif args_cli.sweep_random:
        current_label = "random_yz"

    step = 0
    while simulation_app.is_running():
        t0 = time.time()

        if args_cli.sweep_left:
            t = min(1.0, step / left_total_steps)
            dx = 0.0
            dy = left_travel * t
            _apply_sweep_offset(dx, dy)
            current_label = f"body_left dy={dy:+.2f}m ({int(t * 100)}%)"
        elif args_cli.sweep_random and random_traj is not None:
            dx, dy, dz = random_traj.step()
            _apply_sweep_offset(dx, dy, dz)
            current_label = f"random_yz dy={dy:+.2f} dz={dz:+.2f}"
        elif args_cli.smooth_target:
            phase = step * dt * omega
            dx = args_cli.target_radius * math.cos(phase)
            dy = args_cli.target_radius * math.sin(phase)
            _apply_sweep_offset(dx, dy)
            current_label = f"smooth R={args_cli.target_radius:.2f} ({math.degrees(phase) % 360:.0f}deg)"
        elif args_cli.auto_sweep:
            do_sweep = False
            if sweep_step_interval > 0:
                if step > 0 and steps_since_sweep >= sweep_step_interval:
                    do_sweep = True
            elif (time.time() - sweep_t0) >= args_cli.sweep_interval_s:
                do_sweep = True
            if do_sweep:
                sweep_idx = (sweep_idx + 1) % len(sweep_offsets)
                dx, dy, current_label = sweep_offsets[sweep_idx]
                _apply_sweep_offset(dx, dy)
                sweep_t0 = time.time()
                steps_since_sweep = 0
                print(f"[debug] auto_sweep ({sweep_mode}) -> {current_label} dx={dx:+.2f} dy={dy:+.2f}", flush=True)

        actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
        env.step(actions)

        if args_cli.save_sonar and step % env_cfg.ibvs_debug_interval == 0:
            _maybe_save_sonar(unwrapped, step, save_dir)

        step += 1
        steps_since_sweep += 1
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            m = unwrapped.get_ibvs_metrics(env_id=0)
            print(
                f"[debug] max_steps={args_cli.max_steps} reached; "
                f"valid_pts={m['valid_pts']} pixel_err={m['pixel_err']:.4f} "
                f"offset={m['offset_m']:.3f}m sweep={current_label}",
                flush=True,
            )
            break
        if args_cli.real_time:
            sleep_t = dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
