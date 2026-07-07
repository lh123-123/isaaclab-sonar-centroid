#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless: save Cartesian sonar images with IBVS feature overlays.

Default sweep moves the **target** in a small range while the AUV stays frozen at teach pose.
Use ``--sweep_robot`` to restore the old AUV-offset sweep.
"""

from __future__ import annotations

import argparse
import math
import os
import random

import cv2
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Save sonar cart images with feature point overlays.")
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Debug-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/sonar_flow_hover_debug/feature_viz_lk_smooth",
    help="Directory for annotated PNGs.",
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=5,
    help="Zero-action steps after each pose before capture.",
)
parser.add_argument(
    "--tracking_mode",
    type=str,
    choices=("patch", "lk"),
    default="lk",
    help="Feature tracker: lk=classic optical flow, patch=teach patch NCC.",
)
parser.add_argument(
    "--smooth_target",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Smooth circular target motion (default). Use --no-smooth_target for discrete jumps.",
)
parser.add_argument(
    "--target_radius",
    type=float,
    default=0.15,
    help="Circle radius (m) for smooth target motion.",
)
parser.add_argument(
    "--target_period_s",
    type=float,
    default=12.0,
    help="Full circle period (s) for smooth target motion.",
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
    default=1.5,
    help="Body-frame lateral amplitude (m) for --sweep_random.",
)
parser.add_argument(
    "--random_z_m",
    type=float,
    default=0.8,
    help="Body-frame vertical amplitude (m) for --sweep_random.",
)
parser.add_argument(
    "--random_duration_s",
    type=float,
    default=15.0,
    help="Duration (s) for --sweep_random.",
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
    help="RNG seed for --sweep_random trajectory.",
)
parser.add_argument(
    "--sweep_left",
    action="store_true",
    default=False,
    help="Smooth target motion along -X (AUV/camera fixed). Overrides circular smooth sweep.",
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
    default=4.0,
    help="Duration (s) for --sweep_left ramp 0 -> +left_travel_m body-Y.",
)
parser.add_argument(
    "--capture_interval_s",
    type=float,
    default=3.0,
    help="Seconds between saved frames during smooth motion.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=10,
    help="Zero-action steps at origin before smooth sweep.",
)
parser.add_argument(
    "--sweep_robot",
    action="store_true",
    default=False,
    help="Sweep AUV kinematic offset (legacy). Default: sweep target position.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O."
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
from isaaclab_tasks.utils import parse_env_cfg

# Small-range target motion (meters, world XY relative to teach target pose).
TARGET_SWEEP: list[tuple[float, float, str]] = [
    (0.00, 0.00, "origin"),
    (0.15, 0.00, "tgt_plus_x_0p15"),
    (0.00, 0.15, "tgt_plus_y_0p15"),
    (-0.15, 0.00, "tgt_minus_x_0p15"),
    (0.00, -0.15, "tgt_minus_y_0p15"),
    (0.10, 0.10, "tgt_diag_0p10"),
]

ROBOT_SWEEP: list[tuple[float, float, str]] = [
    (0.00, 0.00, "origin"),
    (0.25, 0.00, "plus_x_0p25"),
    (0.00, 0.25, "plus_y_0p25"),
    (-0.25, 0.00, "minus_x_0p25"),
    (0.15, 0.15, "diag_0p15"),
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


def _cart_to_bgr(cart: np.ndarray) -> np.ndarray:
    u8 = (np.clip(cart, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def _taught_indices(ref_pts: np.ndarray) -> list[int]:
    norms = np.linalg.norm(ref_pts, axis=1)
    return [i for i in range(len(ref_pts)) if norms[i] > 1.0e-3]


def _draw_features(
    cart: np.ndarray,
    ref_pts: np.ndarray,
    curr_pts: np.ndarray,
    valid: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    img = _cart_to_bgr(cart)
    h, w = cart.shape
    taught = _taught_indices(ref_pts)
    n_lost = 0

    for i in taught:
        ref = ref_pts[i]
        rx, ry = int(round(ref[0])), int(round(ref[1]))
        if not (0 <= rx < w and 0 <= ry < h):
            continue
        is_valid = bool(valid[i])
        if not is_valid:
            n_lost += 1
        green = (0, 200, 0) if is_valid else (0, 120, 0)
        cv2.drawMarker(img, (rx, ry), green, markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
        if is_valid:
            cx, cy = int(round(curr_pts[i, 0])), int(round(curr_pts[i, 1]))
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
                cv2.line(img, (rx, ry), (cx, cy), (0, 220, 255), 1, cv2.LINE_AA)

    n_valid = sum(1 for i in taught if valid[i])
    cv2.putText(
        img,
        f"{label}  track={n_valid}/{len(taught)} lost={n_lost}",
        (6, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "green+=ref(red=tracked) darkgreen=lost  red=curr",
        (6, h - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return img


def _capture_frame(
    unwrapped,
    env,
    zero: torch.Tensor,
    *,
    dx: float,
    dy: float,
    tag: str,
    out_dir: str,
    settle_steps: int,
    smooth: bool,
    dz: float = 0.0,
    reposition: bool = True,
) -> str:
    if reposition:
        if smooth:
            unwrapped.set_smooth_target_offset(dx, dy, dz)
        else:
            unwrapped.set_displacement_offset(0.0, 0.0)
            unwrapped.set_target_offset(dx, dy, dz)

        for _ in range(max(settle_steps, 1)):
            env.step(zero)
    unwrapped._ensure_pipeline()

    ft = unwrapped.feature_tracker
    cart = unwrapped.cart_image[0].detach().cpu().numpy()
    ref = ft.ref_points[0].detach().cpu().numpy()
    curr = ft.curr_points[0].detach().cpu().numpy()
    valid = ft.valid[0].detach().cpu().numpy()
    n_track = int(ft.max_track)
    ref = ref[:n_track]
    curr = curr[:n_track]
    valid = valid[:n_track]

    m = unwrapped.get_ibvs_metrics(env_id=0)
    taught_n = len(_taught_indices(ref))
    fname = f"features_{tag}_body_dx{dx:+.2f}_dy{dy:+.2f}_dz{dz:+.2f}.png"
    fpath = os.path.join(out_dir, fname)
    img = _draw_features(
        cart,
        ref,
        curr,
        valid,
        label=f"{tag} body=({dx:+.2f},{dy:+.2f},{dz:+.2f}) err={m['pixel_err']:.4f}",
    )
    cv2.imwrite(fpath, img)
    print(
        f"[viz] {fname} track={m['valid_pts']}/{taught_n} "
        f"lost={taught_n - m['valid_pts']} pixel_err={m['pixel_err']:.4f}",
        flush=True,
    )
    return fpath


def main() -> None:
    out_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    use_random = bool(args_cli.sweep_random) and not args_cli.sweep_robot
    use_left = bool(args_cli.sweep_left) and not args_cli.sweep_robot and not use_random
    use_smooth = bool(args_cli.smooth_target) and not args_cli.sweep_robot and not use_left and not use_random
    if use_random:
        sweep_mode = "random_yz"
    elif use_left:
        sweep_mode = "left_smooth"
    elif use_smooth:
        sweep_mode = "smooth_target"
    else:
        sweep_mode = "robot" if args_cli.sweep_robot else "target_jump"

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.reference_mode = "file"
    env_cfg.print_ibvs_debug = False
    env_cfg.freeze_robot_pose = True
    env_cfg.feature_tracking_mode = args_cli.tracking_mode

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    zero = torch.zeros(env.action_space.shape, device=unwrapped.device)

    env.reset()
    unwrapped.set_displacement_offset(0.0, 0.0)
    unwrapped.set_target_velocity(0.0, 0.0)
    unwrapped.set_target_offset(0.0, 0.0)
    saved: list[str] = []

    print(f"[viz] sweep_mode={sweep_mode}  tracking={args_cli.tracking_mode}  settle_steps={args_cli.settle_steps}", flush=True)

    if use_random:
        dt = float(getattr(env, "step_dt", unwrapped.step_dt))
        duration = max(args_cli.random_duration_s, 1.0e-3)
        total_steps = max(1, int(round(duration / dt)))
        steps_per_capture = max(1, int(round(args_cli.capture_interval_s / dt)))
        traj = _RandomYzWalk(
            random.Random(args_cli.random_seed),
            args_cli.random_y_m,
            args_cli.random_z_m,
            dt,
        )
        print(
            f"[viz] random_yz_walk amp_y=±{args_cli.random_y_m:.2f}m amp_z=±{args_cli.random_z_m:.2f}m "
            f"duration={duration:.1f}s seed={args_cli.random_seed}",
            flush=True,
        )
        for _ in range(max(args_cli.warmup_steps, 1)):
            env.step(zero)
        saved.append(
            _capture_frame(
                unwrapped, env, zero, dx=0.0, dy=0.0, dz=0.0,
                tag="origin", out_dir=out_dir, settle_steps=args_cli.settle_steps, smooth=True,
            )
        )
        capture_idx = 0
        step = 0
        next_capture = steps_per_capture
        while step < total_steps:
            step += 1
            t = step * dt
            dx, dy, dz = traj.step()
            unwrapped.set_smooth_target_offset(dx, dy, dz)
            env.step(zero)
            if step >= next_capture or step == total_steps:
                capture_idx += 1
                tag = f"rand_{capture_idx:02d}_t{t:.1f}s"
                saved.append(
                    _capture_frame(
                        unwrapped, env, zero, dx=dx, dy=dy, dz=dz,
                        tag=tag, out_dir=out_dir, settle_steps=0, smooth=True, reposition=False,
                    )
                )
                next_capture += steps_per_capture
    elif use_left:
        dt = float(getattr(env, "step_dt", unwrapped.step_dt))
        travel = max(args_cli.left_travel_m, 0.0)
        duration = max(args_cli.left_duration_s, 1.0e-3)
        total_steps = max(1, int(round(duration / dt)))
        steps_per_capture = max(1, int(round(args_cli.capture_interval_s / dt)))

        print(
            f"[viz] left_smooth body_left=+{travel:.2f}m duration={duration:.1f}s "
            f"capture_every={args_cli.capture_interval_s:.1f}s  auv_fixed=True",
            flush=True,
        )

        for _ in range(max(args_cli.warmup_steps, 1)):
            env.step(zero)
        saved.append(
            _capture_frame(
                unwrapped,
                env,
                zero,
                dx=0.0,
                dy=0.0,
                tag="origin",
                out_dir=out_dir,
                settle_steps=1,
                smooth=True,
            )
        )

        step = 1
        next_capture = steps_per_capture
        while step <= total_steps:
            t = min(1.0, step / total_steps)
            dx = 0.0
            dy = travel * t
            unwrapped.set_smooth_target_offset(dx, dy)
            env.step(zero)
            if step >= next_capture or step == total_steps:
                tag = f"left_{int(t * 100):03d}pct"
                saved.append(
                    _capture_frame(
                        unwrapped,
                        env,
                        zero,
                        dx=dx,
                        dy=dy,
                        tag=tag,
                        out_dir=out_dir,
                        settle_steps=1,
                        smooth=True,
                    )
                )
                next_capture += steps_per_capture
            step += 1
    elif use_smooth:
        dt = float(getattr(env, "step_dt", unwrapped.step_dt))
        omega = 2.0 * math.pi / max(args_cli.target_period_s, 1.0e-3)
        steps_per_capture = max(1, int(round(args_cli.capture_interval_s / dt)))
        total_steps = max(steps_per_capture, int(round(args_cli.target_period_s / dt)))

        print(
            f"[viz] smooth R={args_cli.target_radius:.2f}m "
            f"period={args_cli.target_period_s:.1f}s capture_every={args_cli.capture_interval_s:.1f}s",
            flush=True,
        )

        for _ in range(max(args_cli.warmup_steps, 1)):
            env.step(zero)
        saved.append(
            _capture_frame(
                unwrapped,
                env,
                zero,
                dx=0.0,
                dy=0.0,
                tag="origin",
                out_dir=out_dir,
                settle_steps=1,
                smooth=True,
            )
        )

        step = 0
        next_capture = 0
        while step < total_steps:
            phase = step * dt * omega
            dx = args_cli.target_radius * math.cos(phase)
            dy = args_cli.target_radius * math.sin(phase)
            unwrapped.set_smooth_target_offset(dx, dy)
            env.step(zero)
            if step >= next_capture:
                deg = math.degrees(phase) % 360.0
                tag = f"smooth_{deg:03.0f}deg"
                saved.append(
                    _capture_frame(
                        unwrapped,
                        env,
                        zero,
                        dx=dx,
                        dy=dy,
                        tag=tag,
                        out_dir=out_dir,
                        settle_steps=1,
                        smooth=True,
                    )
                )
                next_capture += steps_per_capture
            step += 1
    else:
        sweep = ROBOT_SWEEP if args_cli.sweep_robot else TARGET_SWEEP
        for dx, dy, tag in sweep:
            if args_cli.sweep_robot:
                unwrapped.set_displacement_offset(dx, dy)
                unwrapped.set_target_offset(0.0, 0.0)
            else:
                unwrapped.set_displacement_offset(0.0, 0.0)
                unwrapped.set_target_offset(dx, dy)

            saved.append(
                _capture_frame(
                    unwrapped,
                    env,
                    zero,
                    dx=dx,
                    dy=dy,
                    tag=tag,
                    out_dir=out_dir,
                    settle_steps=args_cli.settle_steps,
                    smooth=False,
                )
            )

    env.close()
    simulation_app.close()

    print(f"\n[viz] Saved {len(saved)} images -> {out_dir}")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
