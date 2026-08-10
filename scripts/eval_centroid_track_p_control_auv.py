#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AUV Phase A1: image P-control, triangle target @ constant vy (same as Cuboid Phase 2)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="WarpAUV centroid track — image P-control eval.")
parser.add_argument("--task", type=str, default="Isaac-SonarCentroidTrack-AUV-PControl-Direct-v0")
parser.add_argument("--steps", type=int, default=0)
parser.add_argument("--deadband", type=float, default=0.003)
parser.add_argument("--kp", type=float, default=0.0, help="Image P gain on delta_col (image_range mode)")
parser.add_argument("--kff", type=float, default=0.0, help="Feedforward on target_vy (0=pure image P, no GT oracle)")
parser.add_argument("--max_vy", type=float, default=0.35, help="Clamp lateral cmd vy (image_range mode)")
parser.add_argument(
    "--control",
    type=str,
    choices=("position_lead", "image_range"),
    default="image_range",
    help="position_lead: oracle+lead; image_range: pure image P + range hold",
)
parser.add_argument("--kp_sway", type=float, default=0.0)
parser.add_argument("--auv_pos_kp", type=float, default=10.0, help="0=use cfg default")
parser.add_argument("--auv_pos_kd", type=float, default=0.0, help="0=use cfg default")
parser.add_argument("--auv_max_wrench", type=float, default=200.0, help="0=use cfg default")
parser.add_argument("--auv_max_force", type=float, default=0.0, help="0=use cfg default")
parser.add_argument("--lead_s", type=float, default=0.48, help="Feedforward lead time on target vy (s)")
parser.add_argument("--kp_vy", type=float, default=0.55, help="High-level vy error gain on sway setpoint (s)")
parser.add_argument(
    "--auv_vy_ff",
    type=float,
    default=0.0,
    help="Low-level decoupled vy feedforward gain (0=use cfg default, tune best=70)",
)
parser.add_argument(
    "--desired_range_m",
    type=float,
    default=0.0,
    help="Target sonar range to ball (m); 0 = teach range at calib",
)
parser.add_argument(
    "--kp_range",
    type=float,
    default=1.5,
    help="Range-hold gain: desired_surge = kp_range * (range_meas - desired_range); 0=off",
)
parser.add_argument("--max_surge", type=float, default=0.8, help="Clamp |desired_surge| (m)")
parser.add_argument(
    "--kp_surge_hold",
    type=float,
    default=0.8,
    help="Pull robot surge back toward station: desired_surge -= kp_surge_hold * surge",
)
parser.add_argument("--target_motion", type=str, choices=("sine", "triangle", "round_trip"), default="sine")
parser.add_argument("--sway_amp", type=float, default=1.3)
parser.add_argument("--target_vy", type=float, default=0.28, help="Peak |vy| for sine; constant |vy| for triangle legs")
parser.add_argument(
    "--target_period_s",
    type=float,
    default=0.0,
    help="Sine period (s); 0 = auto from sway_amp and target_vy peak",
)
parser.add_argument("--calib_settle_steps", type=int, default=8)
parser.add_argument("--output_dir", type=str, default="logs/centroid_track/auv_p_triangle")
parser.add_argument(
    "--save_every",
    type=int,
    default=0,
    help="Save polar overlay (green=sonar, yellow=fixed) every N steps; 0=off",
)
parser.add_argument("--step_sleep", type=float, default=0.0)
parser.add_argument(
    "--sonar_ui",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="OceanSim sonar viewport (GUI).",
)
parser.add_argument("--tune", action="store_true", help="Sweep image + low-level gains in one session")
parser.add_argument(
    "--auv_control",
    type=str,
    choices=("decoupled", "coupled"),
    default="decoupled",
    help="decoupled=mission-frame scalar PD + attitude hold; coupled=legacy 3D pos PD",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.headless:
    args_cli.sonar_ui = False
elif not args_cli.headless:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.sonar_centroid_track.modules.coordinates import (
    gt_sonar_beacon_horiz_range_m,
)
from isaaclab_tasks.direct.sonar_centroid_track.modules.target_motion import (
    motion_leg_steps,
    sine_dy_and_vy,
    sine_period_for_peak_vy,
    triangle_cycle_steps,
)
from isaaclab_tasks.utils import parse_env_cfg

# Position + lead only (kp_img=0). Tune kp_sway, lead_s, low-level PD.
POS_LEAD_TUNE_GRID: list[dict] = [
    {"tag": "pl_35_045", "kp_sway": 3.5, "lead_s": 0.45, "auv_pos_kp": 8.0, "auv_pos_kd": 2.0, "auv_max_force": 220.0},
    {"tag": "pl_45_045", "kp_sway": 4.5, "lead_s": 0.45, "auv_pos_kp": 8.0, "auv_pos_kd": 2.0, "auv_max_force": 220.0},
    {"tag": "pl_50_048", "kp_sway": 5.0, "lead_s": 0.48, "auv_pos_kp": 10.0, "auv_pos_kd": 2.5, "auv_max_force": 260.0},
    {"tag": "pl_55_050", "kp_sway": 5.5, "lead_s": 0.50, "auv_pos_kp": 10.0, "auv_pos_kd": 2.5, "auv_max_force": 260.0},
    {"tag": "pl_50_055", "kp_sway": 5.0, "lead_s": 0.55, "auv_pos_kp": 12.0, "auv_pos_kd": 3.0, "auv_max_force": 280.0},
    {"tag": "pl_60_052", "kp_sway": 6.0, "lead_s": 0.52, "auv_pos_kp": 12.0, "auv_pos_kd": 3.0, "auv_max_force": 280.0},
    {"tag": "pl_45_060", "kp_sway": 4.5, "lead_s": 0.60, "auv_pos_kp": 10.0, "auv_pos_kd": 2.5, "auv_max_force": 280.0},
    {"tag": "pl_55_058", "kp_sway": 5.5, "lead_s": 0.58, "auv_pos_kp": 12.0, "auv_pos_kd": 3.0, "auv_max_force": 300.0},
]

REFINE_TUNE_GRID: list[dict] = [
    {"tag": "r58_vy03", "kp_sway": 5.5, "lead_s": 0.58, "kp_vy": 0.30, "auv_pos_kp": 12.0, "auv_pos_kd": 3.0, "auv_max_force": 300.0},
    {"tag": "r58_vy05", "kp_sway": 5.5, "lead_s": 0.58, "kp_vy": 0.50, "auv_pos_kp": 12.0, "auv_pos_kd": 3.0, "auv_max_force": 300.0},
    {"tag": "r65_vy05", "kp_sway": 6.5, "lead_s": 0.65, "kp_vy": 0.50, "auv_pos_kp": 12.0, "auv_pos_kd": 3.0, "auv_max_force": 300.0},
    {"tag": "r65_vy06", "kp_sway": 6.5, "lead_s": 0.65, "kp_vy": 0.60, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5, "auv_max_force": 300.0},
    {"tag": "r72_vy05", "kp_sway": 6.0, "lead_s": 0.72, "kp_vy": 0.50, "auv_pos_kp": 12.0, "auv_pos_kd": 3.0, "auv_max_force": 300.0},
    {"tag": "r58_ff70", "kp_sway": 5.5, "lead_s": 0.58, "kp_vy": 0.45, "auv_pos_kp": 12.0, "auv_pos_kd": 3.0, "auv_max_force": 300.0, "auv_vy_ff": 70.0},
]

# Round-2 sweep around frozen baseline (kp_range=0; lateral PD only).
PD_TUNE_GRID: list[dict] = [
    {"tag": "base_r65_v06", "kp_sway": 6.5, "lead_s": 0.65, "kp_vy": 0.60, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "lead070_v06", "kp_sway": 6.5, "lead_s": 0.70, "kp_vy": 0.60, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "lead075_v06", "kp_sway": 6.5, "lead_s": 0.75, "kp_vy": 0.60, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "lead065_v07", "kp_sway": 6.5, "lead_s": 0.65, "kp_vy": 0.70, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "lead065_v08", "kp_sway": 6.5, "lead_s": 0.65, "kp_vy": 0.80, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "kps70_l065", "kp_sway": 7.0, "lead_s": 0.65, "kp_vy": 0.60, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "kps75_l065", "kp_sway": 7.5, "lead_s": 0.65, "kp_vy": 0.60, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "kps60_l070", "kp_sway": 6.0, "lead_s": 0.70, "kp_vy": 0.65, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "pkp16_kd40", "kp_sway": 6.5, "lead_s": 0.65, "kp_vy": 0.60, "auv_pos_kp": 16.0, "auv_pos_kd": 4.0},
    {"tag": "pkp12_kd30", "kp_sway": 6.5, "lead_s": 0.65, "kp_vy": 0.60, "auv_pos_kp": 12.0, "auv_pos_kd": 3.0},
    {"tag": "img08_l065", "kp": 0.8, "kp_sway": 6.5, "lead_s": 0.65, "kp_vy": 0.60, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "img12_l065", "kp": 1.2, "kp_sway": 6.5, "lead_s": 0.65, "kp_vy": 0.60, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "img15_l070", "kp": 1.5, "kp_sway": 6.5, "lead_s": 0.70, "kp_vy": 0.65, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "ff55_l065", "kp_sway": 6.5, "lead_s": 0.65, "kp_vy": 0.60, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5, "auv_vy_ff": 55.0},
    {"tag": "ff70_l065", "kp_sway": 6.5, "lead_s": 0.65, "kp_vy": 0.60, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5, "auv_vy_ff": 70.0},
    {"tag": "ff85_l065", "kp_sway": 6.5, "lead_s": 0.65, "kp_vy": 0.60, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5, "auv_vy_ff": 85.0},
]

# Distance hold on frozen lateral baseline (ff70_l065).
PD_RANGE_TUNE_GRID: list[dict] = [
    {
        "tag": "ff70_r08_h20",
        "kp_sway": 6.5,
        "lead_s": 0.65,
        "kp_vy": 0.60,
        "auv_pos_kp": 14.0,
        "auv_pos_kd": 3.5,
        "auv_vy_ff": 70.0,
        "kp_range": 0.8,
        "kp_surge_hold": 2.0,
    },
    {
        "tag": "ff70_r10_h20",
        "kp_sway": 6.5,
        "lead_s": 0.65,
        "kp_vy": 0.60,
        "auv_pos_kp": 14.0,
        "auv_pos_kd": 3.5,
        "auv_vy_ff": 70.0,
        "kp_range": 1.0,
        "kp_surge_hold": 2.0,
    },
    {
        "tag": "ff70_r12_h25",
        "kp_sway": 6.5,
        "lead_s": 0.65,
        "kp_vy": 0.60,
        "auv_pos_kp": 14.0,
        "auv_pos_kd": 3.5,
        "auv_vy_ff": 70.0,
        "kp_range": 1.2,
        "kp_surge_hold": 2.5,
    },
    {
        "tag": "ff70_r15_h25",
        "kp_sway": 6.5,
        "lead_s": 0.65,
        "kp_vy": 0.60,
        "auv_pos_kp": 14.0,
        "auv_pos_kd": 3.5,
        "auv_vy_ff": 70.0,
        "kp_range": 1.5,
        "kp_surge_hold": 2.5,
    },
    {
        "tag": "ff70_r10_h30",
        "kp_sway": 6.5,
        "lead_s": 0.65,
        "kp_vy": 0.60,
        "auv_pos_kp": 14.0,
        "auv_pos_kd": 3.5,
        "auv_vy_ff": 70.0,
        "kp_range": 1.0,
        "kp_surge_hold": 3.0,
    },
    {
        "tag": "ff70_r12_h30",
        "kp_sway": 6.5,
        "lead_s": 0.65,
        "kp_vy": 0.60,
        "auv_pos_kp": 14.0,
        "auv_pos_kd": 3.5,
        "auv_vy_ff": 70.0,
        "kp_range": 1.2,
        "kp_surge_hold": 3.0,
    },
    {
        "tag": "ff70_r20_h20",
        "kp_sway": 6.5,
        "lead_s": 0.65,
        "kp_vy": 0.60,
        "auv_pos_kp": 14.0,
        "auv_pos_kd": 3.5,
        "auv_vy_ff": 70.0,
        "kp_range": 2.0,
        "kp_surge_hold": 2.0,
    },
]

TUNE_GRID: list[dict] = PD_TUNE_GRID
IMAGE_P_TUNE_GRID: list[dict] = [
    {"tag": "ip_k08", "kp": 0.8, "kff": 0.0, "kp_range": 0.0, "max_vy": 0.35, "auv_pos_kp": 12.0, "auv_pos_kd": 3.0},
    {"tag": "ip_k10", "kp": 1.0, "kff": 0.0, "kp_range": 0.0, "max_vy": 0.35, "auv_pos_kp": 12.0, "auv_pos_kd": 3.0},
    {"tag": "ip_k12", "kp": 1.2, "kff": 0.0, "kp_range": 0.0, "max_vy": 0.35, "auv_pos_kp": 12.0, "auv_pos_kd": 3.0},
    {"tag": "ip_k15", "kp": 1.5, "kff": 0.0, "kp_range": 0.0, "max_vy": 0.35, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "ip_k18", "kp": 1.8, "kff": 0.0, "kp_range": 0.0, "max_vy": 0.35, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "ip_k20", "kp": 2.0, "kff": 0.0, "kp_range": 0.0, "max_vy": 0.35, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "ip_k12_mv28", "kp": 1.2, "kff": 0.0, "kp_range": 0.0, "max_vy": 0.28, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "ip_k12_mv32", "kp": 1.2, "kff": 0.0, "kp_range": 0.0, "max_vy": 0.32, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
    {"tag": "ip_k10_pk16", "kp": 1.0, "kff": 0.0, "kp_range": 0.0, "max_vy": 0.35, "auv_pos_kp": 16.0, "auv_pos_kd": 4.0},
    {"tag": "ip_k12_pk16", "kp": 1.2, "kff": 0.0, "kp_range": 0.0, "max_vy": 0.35, "auv_pos_kp": 16.0, "auv_pos_kd": 4.0},
    {"tag": "ip_k12_ff70", "kp": 1.2, "kff": 0.0, "kp_range": 0.0, "max_vy": 0.35, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5, "auv_vy_ff": 70.0},
    {"tag": "ip_k15_pk14", "kp": 1.5, "kff": 0.0, "kp_range": 0.0, "max_vy": 0.32, "auv_pos_kp": 14.0, "auv_pos_kd": 3.5},
]

# Legacy: image P + sonar range hold.
IMAGE_RANGE_TUNE_GRID: list[dict] = [
    {"tag": "ir_k08_r15", "kp": 0.8, "kff": 0.0, "kp_range": 1.5, "max_surge": 0.6},
    {"tag": "ir_k10_r15", "kp": 1.0, "kff": 0.0, "kp_range": 1.5, "max_surge": 0.6},
    {"tag": "ir_k12_r20", "kp": 1.2, "kff": 0.0, "kp_range": 2.0, "max_surge": 0.8},
    {"tag": "ir_k15_r20", "kp": 1.5, "kff": 0.0, "kp_range": 2.0, "max_surge": 0.8},
    {"tag": "ir_k10_r25", "kp": 1.0, "kff": 0.0, "kp_range": 2.5, "max_surge": 0.8},
    {"tag": "ir_k12_r25", "kp": 1.2, "kff": 0.0, "kp_range": 2.5, "max_surge": 1.0},
    {"tag": "ir_k15_r30", "kp": 1.5, "kff": 0.0, "kp_range": 3.0, "max_surge": 1.0},
    {"tag": "ir_k20_r25", "kp": 2.0, "kff": 0.0, "kp_range": 2.5, "max_surge": 0.8},
]


def _polar_to_u8(polar: np.ndarray) -> np.ndarray:
    x = polar.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi > lo:
        x = (x - lo) / (hi - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def _draw_green_yellow_overlay(polar: np.ndarray, frame, p_fixed) -> np.ndarray:
    """BGR overlay: yellow=fixed teach point, green=sonar bright-spot."""
    img = cv2.cvtColor(_polar_to_u8(polar), cv2.COLOR_GRAY2BGR)

    def _dot(pt, color: tuple[int, int, int]) -> None:
        if pt is None:
            return
        c, r = int(round(pt.col)), int(round(pt.row))
        cv2.circle(img, (c, r), 1, color, -1, lineType=cv2.LINE_AA)

    _dot(p_fixed, (0, 255, 255))
    _dot(frame.p_sonar_abs, (0, 255, 0))
    return img


def _leg_steps(sway_amp: float, vy: float, step_dt: float) -> int:
    return motion_leg_steps(sway_amp, vy, step_dt)


def _sine_period_s(sway_amp: float, vy_peak: float, period_override: float) -> float:
    if float(period_override) > 0.0:
        return float(period_override)
    return sine_period_for_peak_vy(sway_amp, vy_peak)


def _total_steps(
    target_motion: str,
    sway_amp: float,
    vy: float,
    step_dt: float,
    fallback: int,
    *,
    period_s: float = 0.0,
) -> int:
    if fallback > 0:
        return fallback
    if target_motion == "sine":
        return max(int(round(_sine_period_s(sway_amp, vy, period_s) / step_dt)), 1)
    leg = _leg_steps(sway_amp, vy, step_dt)
    if target_motion == "triangle":
        return triangle_cycle_steps(sway_amp, vy, step_dt)
    return 2 * leg


def _target_vy(
    step: int,
    *,
    target_motion: str,
    sway_amp: float,
    vy: float,
    step_dt: float,
    period_s: float = 0.0,
) -> float:
    if target_motion == "sine":
        _, vy_s = sine_dy_and_vy(
            step, amp=sway_amp, period_s=_sine_period_s(sway_amp, vy, period_s), step_dt=step_dt
        )
        return float(vy_s)
    leg = _leg_steps(sway_amp, vy, step_dt)
    if target_motion == "triangle":
        if step < leg:
            return float(vy)
        if step < 3 * leg:
            return float(-vy)
        if step < 4 * leg:
            return float(vy)
        return 0.0
    if step < leg:
        return float(vy)
    if step < 2 * leg:
        return float(-vy)
    return 0.0


def _target_dy(
    step: int,
    *,
    target_motion: str,
    sway_amp: float,
    vy: float,
    step_dt: float,
    period_s: float = 0.0,
) -> float:
    if target_motion == "sine":
        dy, _ = sine_dy_and_vy(
            step, amp=sway_amp, period_s=_sine_period_s(sway_amp, vy, period_s), step_dt=step_dt
        )
        return float(dy)
    dy = 0.0
    for s in range(step):
        dy += _target_vy(
            s, target_motion=target_motion, sway_amp=sway_amp, vy=vy, step_dt=step_dt, period_s=period_s
        ) * step_dt
    total = _total_steps(target_motion, sway_amp, vy, step_dt, 0, period_s=period_s)
    if target_motion == "triangle":
        return float(np.clip(dy, -sway_amp, sway_amp) if step <= total else 0.0)
    return float(np.clip(dy, 0.0, sway_amp) if step <= total else 0.0)


def _zero_pwm(unwrapped) -> torch.Tensor:
    n = int(getattr(unwrapped.cfg, "num_auv_actions", 6))
    return torch.zeros(n, device=unwrapped.device, dtype=torch.float32)


def _resolve_max_vy(cfg_max: float, target_vy: float, override: float) -> float:
    del cfg_max, override
    return max(target_vy * 1.25, 0.12)


def _robot_sway_vy(unwrapped) -> float:
    yaw = unwrapped._fixed_robot_yaw_w
    lateral_w = torch.stack([-torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)], dim=-1)
    return float(torch.sum(unwrapped.robot.data.root_lin_vel_w * lateral_w, dim=-1)[0].item())


def _robot_surge_m(unwrapped) -> float:
    yaw = unwrapped._fixed_robot_yaw_w
    forward_w = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
    delta = unwrapped.robot.data.root_pos_w - unwrapped._station_origin_w
    return float(torch.sum(delta * forward_w, dim=-1)[0].item())


def _robot_attitude_deg(unwrapped) -> tuple[float, float, float]:
    from isaaclab.utils.math import euler_xyz_from_quat

    roll, pitch, yaw = euler_xyz_from_quat(unwrapped.robot.data.root_quat_w)
    return (
        float(torch.rad2deg(roll[0]).item()),
        float(torch.rad2deg(pitch[0]).item()),
        float(torch.rad2deg(yaw[0]).item()),
    )


def _sync_auv_low_level_gains(
    unwrapped,
    *,
    auv_pos_kp: float,
    auv_pos_kd: float,
    auv_max_wrench: float,
    auv_max_force: float,
) -> None:
    if auv_pos_kp > 0.0:
        unwrapped.cfg.auv_pos_kp = float(auv_pos_kp)
        unwrapped.cfg.auv_kp_sway = float(auv_pos_kp)
        unwrapped.cfg.auv_kp_surge = max(float(auv_pos_kp) * 0.85, 8.0)
        unwrapped.cfg.auv_kp_depth = max(float(auv_pos_kp) * 1.15, 10.0)
    if auv_pos_kd > 0.0:
        unwrapped.cfg.auv_pos_kd = float(auv_pos_kd)
        unwrapped.cfg.auv_kd_sway = float(auv_pos_kd)
        unwrapped.cfg.auv_kd_surge = max(float(auv_pos_kd) * 0.85, 2.0)
        unwrapped.cfg.auv_kd_depth = max(float(auv_pos_kd) * 1.15, 2.5)
    if auv_max_wrench > 0.0:
        unwrapped.cfg.auv_max_wrench = float(auv_max_wrench)
    if auv_max_force > 0.0:
        unwrapped.cfg.auv_max_force = float(auv_max_force)
    unwrapped._auv_lat._base.kp = float(unwrapped.cfg.auv_pos_kp)
    unwrapped._auv_lat._base.kd = float(unwrapped.cfg.auv_pos_kd)
    unwrapped._auv_lat._base.max_wrench = float(unwrapped.cfg.auv_max_wrench)


def _sonar_range_cfg(unwrapped) -> dict:
    ocfg = unwrapped.cfg.oceansim_sonar_cfg
    return {
        "min_range": float(ocfg.min_range),
        "max_range": float(ocfg.max_range),
        "polar_height": int(ocfg.height),
    }


def _desired_surge_from_range(
    *,
    range_meas: float | None,
    target_range_m: float,
    kp_range: float,
    max_surge: float,
    surge_robot_m: float = 0.0,
    kp_surge_hold: float = 0.0,
    range_deadband_m: float = 0.0,
    kp_range_fwd: float | None = None,
    kp_range_back: float | None = None,
    range_close_margin_m: float = 0.25,
) -> float:
    if kp_range <= 0.0 and kp_surge_hold <= 0.0:
        return 0.0
    if range_meas is None:
        if kp_surge_hold > 0.0:
            return float(np.clip(-float(kp_surge_hold) * float(surge_robot_m), -max_surge, max_surge))
        return 0.0
    import torch

    from isaaclab_tasks.direct.sonar_centroid_track.modules.auv_image_range_base import (
        desired_surge_from_range,
    )

    out = desired_surge_from_range(
        range_meas=torch.tensor([float(range_meas)], dtype=torch.float32),
        range_ref_gt_m=torch.tensor([float(target_range_m)], dtype=torch.float32),
        valid=torch.ones(1),
        surge_robot_m=torch.tensor([float(surge_robot_m)], dtype=torch.float32),
        kp_range=float(kp_range),
        kp_surge_hold=float(kp_surge_hold),
        max_surge=float(max_surge),
        range_deadband_m=float(range_deadband_m),
        kp_range_fwd=kp_range_fwd,
        kp_range_back=kp_range_back,
        range_close_margin_m=float(range_close_margin_m),
    )
    return float(out[0].item())


def _desired_sway(
    *,
    tgt_dy: float,
    tgt_vy: float,
    sway: float,
    sway_vy: float,
    delta_norm: float,
    kp_img: float,
    kp_sway: float,
    kp_vy: float,
    sway_amp: float,
    lead_s: float,
) -> float:
    """Position setpoint with lead + sway P + optional vy correction + image term."""
    img_m = float(kp_img) * float(delta_norm) * float(sway_amp) * 0.35
    lead_m = float(lead_s) * float(tgt_vy)
    vel_m = float(kp_vy) * (float(tgt_vy) - float(sway_vy))
    return float(tgt_dy) + lead_m + float(kp_sway) * (float(tgt_dy) - float(sway)) + vel_m + img_m


def _desired_sway_image(
    *,
    sway: float,
    delta_norm: float,
    tgt_vy: float,
    kp: float,
    kff: float,
    max_vy: float,
    deadband: float,
    step_dt: float,
) -> float:
    """Pure image P + target vy FF → integrated sway setpoint."""
    from isaaclab_tasks.direct.sonar_centroid_track.modules.ibvs_p_controller import (
        image_delta_lateral_velocity,
    )

    rvy = image_delta_lateral_velocity(
        delta_norm, tgt_vy, kp=float(kp), kff=float(kff), max_vy=float(max_vy), deadband=float(deadband)
    )
    return float(sway) + float(rvy) * float(step_dt)


def _run_trial(
    env,
    unwrapped,
    *,
    kp: float,
    kff: float,
    max_vy: float,
    control: str,
    kp_sway: float,
    auv_pos_kp: float,
    auv_pos_kd: float,
    auv_max_wrench: float,
    auv_max_force: float,
    kp_vy: float,
    auv_vy_ff: float,
    auv_control: str,
    lead_s: float,
    desired_range_m: float,
    kp_range: float,
    max_surge: float,
    kp_surge_hold: float,
    target_period_s: float,
    target_motion: str,
    sway_amp: float,
    target_vy: float,
    step_dt: float,
    total_steps: int,
    calib_settle_steps: int,
    deadband: float,
    step_sleep: float,
    verbose: bool,
    save_every: int = 0,
    frames_dir: Path | None = None,
    montage: list | None = None,
) -> dict:
    unwrapped.cfg.auv_control_mode = str(auv_control)
    _sync_auv_low_level_gains(
        unwrapped,
        auv_pos_kp=auv_pos_kp,
        auv_pos_kd=auv_pos_kd,
        auv_max_wrench=auv_max_wrench,
        auv_max_force=auv_max_force,
    )
    if auv_vy_ff > 0.0:
        unwrapped.cfg.auv_vy_ff = float(auv_vy_ff)

    idle = _zero_pwm(unwrapped)
    polar_w = float(unwrapped.cfg.oceansim_sonar_cfg.width)
    sonar_rng = _sonar_range_cfg(unwrapped)

    env.reset()
    for _ in range(calib_settle_steps):
        unwrapped.set_target_offset(0.0, 0.0)
        env.step(idle)
    unwrapped.recalibrate_fixed_point()
    unwrapped.set_target_offset(0.0, 0.0)
    if hasattr(unwrapped, "recalibrate_sonar_range"):
        unwrapped.recalibrate_sonar_range()

    p_fixed = unwrapped.fixed.p_fixed_abs_env(0)
    if hasattr(unwrapped, "range_ref_for_control_m"):
        range_ref_m = float(unwrapped.range_ref_for_control_m()[0].item())
    elif p_fixed is not None:
        from isaaclab_tasks.direct.sonar_centroid_track.modules.coordinates import polar_row_to_range_m

        range_ref_m = polar_row_to_range_m(p_fixed.row, **sonar_rng)
    else:
        range_ref_m = float(getattr(unwrapped.cfg, "beacon_offset_local", (3.0, 0.0, 0.0))[0])
    target_range_m = float(desired_range_m) if float(desired_range_m) > 0.0 else float(range_ref_m)

    delta_col_log: list[float] = []
    sway_errs: list[float] = []
    range_errs: list[float] = []
    range_meas_log: list[float] = []
    roll_abs: list[float] = []
    pitch_abs: list[float] = []

    for step in range(total_steps):
        tgt_dy = _target_dy(
            step + 1,
            target_motion=target_motion,
            sway_amp=sway_amp,
            vy=target_vy,
            step_dt=step_dt,
            period_s=target_period_s,
        )
        tgt_vy = _target_vy(
            step,
            target_motion=target_motion,
            sway_amp=sway_amp,
            vy=target_vy,
            step_dt=step_dt,
            period_s=target_period_s,
        )
        unwrapped.set_target_offset(0.0, tgt_dy)

        frame = unwrapped.get_centroid_frame(0)
        p_fixed = unwrapped.fixed.p_fixed_abs_env(0)
        delta_col = None
        delta_row = None
        range_meas_t, _, _, range_valid_t = unwrapped.beam_peak_range_meas_m()
        range_meas = (
            float(range_meas_t[0].item()) if float(range_valid_t[0].item()) >= 0.5 else None
        )
        if frame.p_sonar_abs is not None and p_fixed is not None:
            delta_col = float(frame.p_sonar_abs.col - p_fixed.col)
            delta_row = float(frame.p_sonar_abs.row - p_fixed.row)
        delta_norm = float(delta_col) / polar_w if delta_col is not None else 0.0
        if abs(delta_norm) <= deadband:
            delta_norm = 0.0
        sway_pre = float(unwrapped._robot_sway_m()[0].item())
        sway_vy_pre = _robot_sway_vy(unwrapped)
        if control == "image_range":
            from isaaclab_tasks.direct.sonar_centroid_track.modules.ibvs_p_controller import (
                image_delta_lateral_velocity,
            )

            rvy = float(
                image_delta_lateral_velocity(
                    delta_norm,
                    0.0,
                    kp=float(kp),
                    kff=0.0,
                    max_vy=float(max_vy),
                    deadband=float(deadband),
                )
            )
            rvy_t = torch.tensor([rvy], device=unwrapped.device, dtype=torch.float32)
            unwrapped.set_lateral_velocity_override(rvy_t)
            unwrapped.set_auv_feedforward_vy(rvy_t)
            desired = None
        else:
            desired = _desired_sway(
                tgt_dy=tgt_dy, tgt_vy=tgt_vy, sway=sway_pre, sway_vy=sway_vy_pre, delta_norm=delta_norm,
                kp_img=kp, kp_sway=kp_sway, kp_vy=kp_vy, sway_amp=sway_amp, lead_s=lead_s,
            )
            desired = float(np.clip(desired, -sway_amp * 1.05, sway_amp * 1.05))
        hold_surge = float(kp_range) > 0.0 or float(kp_surge_hold) > 0.0
        desired_surge = (
            _desired_surge_from_range(
                range_meas=range_meas,
                target_range_m=target_range_m,
                kp_range=kp_range,
                max_surge=max_surge,
                surge_robot_m=_robot_surge_m(unwrapped),
                kp_surge_hold=kp_surge_hold,
                range_deadband_m=float(getattr(unwrapped.cfg, "sonar_range_deadband_m", 0.0)),
                kp_range_fwd=float(getattr(unwrapped.cfg, "auv_kp_range_fwd", 0.35)),
                kp_range_back=float(getattr(unwrapped.cfg, "auv_kp_range_back", 3.0)),
                range_close_margin_m=float(getattr(unwrapped.cfg, "sonar_range_close_margin_m", 0.25)),
            )
            if hold_surge
            else None
        )

        if desired is not None:
            unwrapped.set_lateral_sway_override(
                torch.tensor([desired], device=unwrapped.device, dtype=torch.float32)
            )
        if desired_surge is not None:
            unwrapped.set_lateral_surge_override(
                torch.tensor([desired_surge], device=unwrapped.device, dtype=torch.float32)
            )
        if control != "image_range" and float(kff) > 0.0:
            unwrapped.set_auv_feedforward_vy(torch.tensor([tgt_vy], device=unwrapped.device, dtype=torch.float32))
        env.step(idle)
        unwrapped.set_lateral_velocity_override(None)
        unwrapped.set_lateral_sway_override(None)
        unwrapped.set_lateral_surge_override(None)
        unwrapped.set_auv_feedforward_vy(None)

        if step_sleep > 0.0:
            time.sleep(step_sleep)

        sway = float(unwrapped._robot_sway_m()[0].item())
        tgt = float(unwrapped._target_dy_m()[0].item())
        if delta_col is not None:
            delta_col_log.append(abs(delta_col))
        sway_errs.append(abs(sway - tgt))
        if range_meas is not None:
            range_meas_log.append(float(range_meas))
            range_errs.append(abs(float(range_meas) - target_range_m))
        r_deg, p_deg, _ = _robot_attitude_deg(unwrapped)
        roll_abs.append(abs(r_deg))
        pitch_abs.append(abs(p_deg))

        if save_every > 0 and frames_dir is not None and (
            step % save_every == 0 or step == total_steps - 1
        ):
            polar_np = unwrapped.polar_image[0].detach().cpu().numpy()
            frame_save = unwrapped.get_centroid_frame(0)
            p_fixed_save = unwrapped.fixed.p_fixed_abs_env(0)
            dc_save = (
                abs(frame_save.p_sonar_abs.col - p_fixed_save.col)
                if frame_save.p_sonar_abs is not None and p_fixed_save is not None
                else float("nan")
            )
            overlay = _draw_green_yellow_overlay(polar_np, frame_save, p_fixed_save)
            det_s = "1" if frame_save.valid_ratio >= 0.5 else "0"
            path = frames_dir / f"step_{step:04d}_sway_{sway:+.3f}m_dcol_{dc_save:.1f}px_det{det_s}.png"
            cv2.imwrite(str(path), overlay)
            if montage is not None and len(montage) < 6:
                montage.append((step, sway, overlay.copy()))
            print(f"[save] {path.name}  green-yellow |d_col|={dc_save:.2f} px", flush=True)

        if verbose and (step % max(total_steps // 8, 1) == 0 or step == total_steps - 1):
            dc = abs(delta_col) if delta_col is not None else float("nan")
            dr = abs(delta_row) if delta_row is not None else float("nan")
            rng = range_meas if range_meas is not None else float("nan")
            surge = _robot_surge_m(unwrapped)
            r_deg, p_deg, _ = _robot_attitude_deg(unwrapped)
            rng_gt = gt_sonar_beacon_horiz_range_m(
                unwrapped._beacon_pos_w,
                unwrapped.robot.data.root_pos_w,
                unwrapped.robot.data.root_quat_w,
                unwrapped._sonar_mount_offset_b,
                env_id=0,
            )
            print(
                f"    step={step:4d} tgt={tgt:+6.3f} sway={sway:+6.3f} "
                f"|d_col|={dc:5.2f} |d_row|={dr:5.2f} "
                f"rng_sn={rng:5.2f} rng_gt={rng_gt:5.2f} hold={target_range_m:5.2f} "
                f"surge={surge:+5.3f} d_sn={rng - target_range_m:+.2f} d_gt={rng_gt - target_range_m:+.2f} "
                f"dsp={float(desired_surge) if desired_surge is not None else 0.0:+.2f} "
                f"|roll|={abs(r_deg):4.1f}° |pitch|={abs(p_deg):4.1f}°",
                flush=True,
            )

    dc = np.array(delta_col_log, dtype=np.float64)
    se = np.array(sway_errs, dtype=np.float64)
    re = np.array(range_errs, dtype=np.float64)
    rm = np.array(range_meas_log, dtype=np.float64)
    ra = np.array(roll_abs, dtype=np.float64)
    pa = np.array(pitch_abs, dtype=np.float64)
    n_tail = min(80, len(rm))
    n_head = min(80, len(rm))
    range_drift = (
        float(rm[-n_tail:].mean() - rm[:n_head].mean()) if len(rm) >= 2 else float("nan")
    )
    return {
        "d_col_mean": float(dc.mean()) if len(dc) else float("nan"),
        "d_col_p90": float(np.percentile(dc, 90)) if len(dc) else float("nan"),
        "d_col_max": float(dc.max()) if len(dc) else float("nan"),
        "sway_err_mean": float(se.mean()) if len(se) else float("nan"),
        "sway_err_max": float(se.max()) if len(se) else float("nan"),
        "range_err_mean": float(re.mean()) if len(re) else float("nan"),
        "range_err_max": float(re.max()) if len(re) else float("nan"),
        "range_meas_mean": float(rm.mean()) if len(rm) else float("nan"),
        "range_ref_m": float(range_ref_m),
        "target_range_m": float(target_range_m),
        "range_drift_m": range_drift,
        "roll_abs_mean": float(ra.mean()) if len(ra) else float("nan"),
        "roll_abs_max": float(ra.max()) if len(ra) else float("nan"),
        "pitch_abs_mean": float(pa.mean()) if len(pa) else float("nan"),
        "pitch_abs_max": float(pa.max()) if len(pa) else float("nan"),
    }


def _trial_kwargs(
    *,
    cfg: dict | None,
    args,
    target_vy: float,
    step_dt: float,
    total_steps: int,
    verbose: bool,
) -> dict:
    c = cfg or {}
    control = str(args.control)
    return dict(
        kp=float(c.get("kp", 0.0 if control == "position_lead" else args.kp)),
        kff=float(c.get("kff", args.kff)),
        max_vy=float(c.get("max_vy", args.max_vy)),
        control=control,
        kp_sway=float(c.get("kp_sway", args.kp_sway)),
        auv_pos_kp=float(c.get("auv_pos_kp", args.auv_pos_kp)),
        auv_pos_kd=float(c.get("auv_pos_kd", args.auv_pos_kd)),
        auv_max_wrench=float(c.get("auv_max_wrench", args.auv_max_wrench)),
        auv_max_force=float(c.get("auv_max_force", args.auv_max_force)),
        kp_vy=float(c.get("kp_vy", args.kp_vy)),
        auv_vy_ff=float(c.get("auv_vy_ff", args.auv_vy_ff)),
        auv_control=str(getattr(args, "auv_control", "decoupled")),
        lead_s=float(c.get("lead_s", args.lead_s)),
        desired_range_m=float(args.desired_range_m),
        kp_range=float(c.get("kp_range", args.kp_range)),
        max_surge=float(c.get("max_surge", args.max_surge)),
        kp_surge_hold=float(c.get("kp_surge_hold", args.kp_surge_hold)),
        target_period_s=float(args.target_period_s),
        target_motion=str(args.target_motion),
        sway_amp=float(args.sway_amp),
        target_vy=target_vy,
        step_dt=step_dt,
        total_steps=total_steps,
        calib_settle_steps=int(args.calib_settle_steps),
        deadband=float(args.deadband),
        step_sleep=float(args.step_sleep),
        verbose=verbose,
        save_every=int(getattr(args, "save_every", 0)),
        frames_dir=getattr(args, "_frames_dir", None),
        montage=getattr(args, "_montage", None),
    )


def _score_trial(m: dict, control: str, *, kp_range: float = 0.0) -> tuple:
    if control == "image_range":
        if float(kp_range) <= 0.0:
            return (float(m["d_col_mean"]), float(m["sway_err_mean"]))
        drift_pen = max(0.0, -float(m.get("range_drift_m", 0.0)))
        return (float(m["range_err_mean"]), drift_pen, float(m["d_col_mean"]))
    if control == "position_lead" and float(kp_range) > 0.0:
        drift_pen = abs(float(m.get("range_drift_m", 0.0)))
        return (drift_pen, float(m["range_err_mean"]), float(m["d_col_mean"]))
    return (float(m["d_col_mean"]), float(m["sway_err_mean"]))


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.lateral_target_moving = False
    env_cfg.spawn_radius = 0.0
    env_cfg.eval_spawn_radius = 0.0
    env_cfg.print_centroid_debug = False
    env_cfg.debug_disable_lost_terminate = True
    if args_cli.sonar_ui and hasattr(env_cfg, "oceansim_sonar_cfg"):
        env_cfg.oceansim_sonar_cfg.sonar_initialize_viewport = True

    target_vy = float(args_cli.target_vy)
    if float(args_cli.auv_pos_kp) > 0.0:
        env_cfg.auv_pos_kp = float(args_cli.auv_pos_kp)
    if float(args_cli.auv_pos_kd) > 0.0:
        env_cfg.auv_pos_kd = float(args_cli.auv_pos_kd)
    if float(args_cli.auv_max_wrench) > 0.0:
        env_cfg.auv_max_wrench = float(args_cli.auv_max_wrench)
    if float(args_cli.auv_max_force) > 0.0:
        env_cfg.auv_max_force = float(args_cli.auv_max_force)
    env_cfg.auv_control_mode = str(args_cli.auv_control)

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir: Path | None = None
    montage: list | None = None
    if int(args_cli.save_every) > 0:
        frames_dir = out_dir / "polar_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        montage = []
        args_cli._frames_dir = frames_dir  # noqa: SLF001
        args_cli._montage = montage  # noqa: SLF001
        print(f"  polar frames -> {frames_dir.resolve()}  (yellow=fixed, green=sonar)", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    step_dt = float(unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)
    period_s = _sine_period_s(float(args_cli.sway_amp), target_vy, float(args_cli.target_period_s))
    tune_steps = int(args_cli.steps) if int(args_cli.steps) > 0 else (
        max(int(round(0.45 * period_s / step_dt)), 1)
        if str(args_cli.target_motion) == "sine"
        else 3 * _leg_steps(float(args_cli.sway_amp), target_vy, step_dt)
    )
    full_steps = _total_steps(
        args_cli.target_motion,
        float(args_cli.sway_amp),
        target_vy,
        step_dt,
        int(args_cli.steps),
        period_s=period_s,
    )

    if args_cli.tune:
        control = str(args_cli.control)
        if control == "image_range":
            tune_grid = IMAGE_P_TUNE_GRID if float(args_cli.kp_range) <= 0.0 else IMAGE_RANGE_TUNE_GRID
        elif control == "position_lead" and float(args_cli.kp_range) > 0.0:
            tune_grid = PD_RANGE_TUNE_GRID
        else:
            tune_grid = TUNE_GRID
        print(f"\n=== AUV tune ({control}, 3-leg preview) ===\n", flush=True)
        if control == "image_range":
            print(f"{'tag':<14} {'kp':>4} {'kff':>4} {'krng':>4} {'fmax':>4} |d_col| rng_err drift", flush=True)
        elif control == "position_lead" and float(args_cli.kp_range) > 0.0:
            print(f"{'tag':<16} {'krng':>4} {'hold':>4} |d_col| rng_err drift  sway_err", flush=True)
        else:
            print(f"{'tag':<14} {'kps':>4} {'lead':>5} {'kvy':>4} {'img':>4} {'pkp':>4} |d_col| mean  sway_err", flush=True)
        results: list[tuple[dict, dict]] = []
        for cfg in tune_grid:
            tag = str(cfg["tag"])
            m = _run_trial(env, unwrapped, **_trial_kwargs(
                cfg=cfg, args=args_cli, target_vy=target_vy, step_dt=step_dt,
                total_steps=tune_steps, verbose=False,
            ))
            results.append((cfg, m))
            if control == "image_range":
                print(
                    f"{tag:<14} {cfg.get('kp', args_cli.kp):4.1f} {cfg.get('kff', args_cli.kff):4.2f} "
                    f"{cfg.get('kp_range', args_cli.kp_range):4.1f} {cfg.get('max_surge', args_cli.max_surge):4.1f} "
                    f"{m['d_col_mean']:8.2f} {m['range_err_mean']:6.3f} {m['range_drift_m']:+.3f}",
                    flush=True,
                )
            elif control == "position_lead" and float(args_cli.kp_range) > 0.0:
                print(
                    f"{tag:<16} {cfg.get('kp_range', args_cli.kp_range):4.1f} "
                    f"{cfg.get('kp_surge_hold', args_cli.kp_surge_hold):4.1f} "
                    f"{m['d_col_mean']:8.2f} {m['range_err_mean']:6.3f} {m['range_drift_m']:+.3f}  "
                    f"{m['sway_err_mean']:.3f}",
                    flush=True,
                )
            else:
                print(
                    f"{tag:<14} {cfg.get('kp_sway', 0):4.1f} {cfg.get('lead_s', args_cli.lead_s):5.2f} "
                    f"{cfg.get('kp_vy', args_cli.kp_vy):4.2f} {cfg.get('kp', 0.0):4.1f} "
                    f"{cfg.get('auv_pos_kp', 0):4.1f} "
                    f"{m['d_col_mean']:8.2f}  {m['sway_err_mean']:.3f}",
                    flush=True,
                )

        best_cfg, best_m = min(
            results,
            key=lambda x: _score_trial(x[1], control, kp_range=float(x[0].get("kp_range", args_cli.kp_range))),
        )
        print(
            f"\n=== BEST tune: {best_cfg['tag']} |d_col|={best_m['d_col_mean']:.2f} "
            f"range_err={best_m['range_err_mean']:.3f} drift={best_m['range_drift_m']:+.3f} ===",
            flush=True,
        )
        print("=== Full triangle re-run ===\n", flush=True)
        full_total = _total_steps(
            args_cli.target_motion, float(args_cli.sway_amp), target_vy, step_dt, 0, period_s=period_s
        )
        final = _run_trial(env, unwrapped, **_trial_kwargs(
            cfg=best_cfg, args=args_cli, target_vy=target_vy, step_dt=step_dt,
            total_steps=full_total, verbose=True,
        ))
        print("\n=== Full triangle summary ===", flush=True)
        print(
            f"  best={best_cfg['tag']}  |d_col| mean={final['d_col_mean']:.2f}  "
            f"p90={final['d_col_p90']:.2f}  max={final['d_col_max']:.2f}",
            flush=True,
        )
        print(f"  |sway−target| mean={final['sway_err_mean']:.3f} m  max={final['sway_err_max']:.3f} m", flush=True)
        print(
            f"  range target={final['target_range_m']:.2f} m  meas mean={final['range_meas_mean']:.2f} m  "
            f"|err| mean={final['range_err_mean']:.3f} m  drift={final['range_drift_m']:+.3f} m\n",
            flush=True,
        )
        tune_out = out_dir / f"tune_results_{control}.txt"
        with tune_out.open("w") as f:
            f.write("tag,d_col_mean,range_err_mean,range_drift_m,sway_err_mean\n")
            for cfg, m in results:
                f.write(
                    f"{cfg['tag']},{m['d_col_mean']:.4f},{m['range_err_mean']:.4f},"
                    f"{m['range_drift_m']:.4f},{m['sway_err_mean']:.4f}\n"
                )
            f.write(
                f"FULL,{best_cfg['tag']},{final['d_col_mean']:.4f},{final['range_err_mean']:.4f},"
                f"{final['range_drift_m']:.4f},{final['sway_err_mean']:.4f}\n"
            )
        print(f"[save] {tune_out.resolve()}", flush=True)
        env.close()
        simulation_app.close()
        return

    leg = _leg_steps(float(args_cli.sway_amp), target_vy, step_dt)
    motion_desc = (
        f"sine A={args_cli.sway_amp:.2f} m  T={period_s:.1f} s  vy_peak={target_vy} m/s  total={full_steps}"
        if args_cli.target_motion == "sine"
        else f"0 → +{args_cli.sway_amp:.2f} → −{args_cli.sway_amp:.2f} → 0 m  @ vy={target_vy} m/s  leg={leg}  total={full_steps}"
    )
    print(f"\n=== AUV control ({args_cli.control}, {args_cli.target_motion}, {args_cli.auv_control}) ===\n", flush=True)
    print(f"  ball: {motion_desc}", flush=True)
    if args_cli.control == "image_range":
        ff_note = f"kff={args_cli.kff}" if float(args_cli.kff) > 0.0 else "kff=0 (no GT vy FF)"
        print(
            f"  kp={args_cli.kp}  {ff_note}  max_vy={args_cli.max_vy}  "
            f"kp_range={args_cli.kp_range}  desired_range_m={args_cli.desired_range_m or 'teach'}\n",
            flush=True,
        )
    else:
        print(
            f"  kp_sway={args_cli.kp_sway}  lead_s={args_cli.lead_s}  kp_vy={args_cli.kp_vy}  "
            f"kp_range={args_cli.kp_range}\n",
            flush=True,
        )

    final = _run_trial(env, unwrapped, **_trial_kwargs(
        cfg=None, args=args_cli, target_vy=target_vy, step_dt=step_dt,
        total_steps=full_steps, verbose=True,
    ))
    print("\n=== Summary ===", flush=True)
    print(
        f"  |d_col| mean={final['d_col_mean']:.2f}  p90={final['d_col_p90']:.2f}  max={final['d_col_max']:.2f}",
        flush=True,
    )
    print(f"  |sway−target| mean={final['sway_err_mean']:.3f} m  max={final['sway_err_max']:.3f} m", flush=True)
    print(
        f"  |roll| mean={final['roll_abs_mean']:.2f}°  max={final['roll_abs_max']:.2f}°  "
        f"|pitch| mean={final['pitch_abs_mean']:.2f}°  max={final['pitch_abs_max']:.2f}°",
        flush=True,
    )
    print(
        f"  range target={final['target_range_m']:.2f} m (teach={final['range_ref_m']:.2f})  "
        f"|range−target| mean={final['range_err_mean']:.3f} m  meas mean={final['range_meas_mean']:.2f} m  "
        f"drift={final['range_drift_m']:+.3f} m\n",
        flush=True,
    )

    if montage:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(montage), figsize=(3.2 * len(montage), 3))
        for ax, (st, sw, bgr) in zip(np.atleast_1d(axes), montage):
            ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), aspect="auto")
            ax.set_title(f"step={st} sway={sw:+.2f}m", fontsize=9)
            ax.axis("off")
        fig.suptitle("AUV track (yellow=fixed teach, green=sonar detect)", fontsize=10)
        mp = out_dir / "polar_montage.png"
        fig.tight_layout()
        fig.savefig(mp, dpi=150)
        plt.close(fig)
        print(f"[save] montage -> {mp.resolve()}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
