#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose WarpAUV PID surge response (HoloOcean-style [v, ω])."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PID surge diagnostic.")
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--cmd_v", type=float, default=1.0, help="Normalized action a_v in [-1,1] (×1.5 → m/s).")
parser.add_argument("--cmd_w", type=float, default=0.0, help="Normalized action a_ω in [-1,1] (×0.5 → rad/s).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply, quat_conjugate
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    env.reset()

    action = torch.zeros(raw.num_envs, 2, device=raw.device)
    action[:, 0] = args_cli.cmd_v
    action[:, 1] = args_cli.cmd_w
    cmd_v = args_cli.cmd_v * raw.cfg.action_scale_v
    cmd_w = args_cli.cmd_w * raw.cfg.action_scale_w
    print(f"[pid-diag] cmd_vel=({cmd_v:.2f} m/s, {cmd_w:.2f} rad/s)", flush=True)

    for step in range(args_cli.steps):
        env.step(action)
        quat_b = quat_conjugate(raw._robot.data.root_quat_w)
        lin_b = quat_apply(quat_b, raw._robot.data.root_lin_vel_w)
        ang_b = quat_apply(quat_b, raw._robot.data.root_ang_vel_w)
        z = raw._robot.data.root_pos_w[0, 2].item()
        pwm = raw._pwm[0]
        if step % 5 == 0 or step == args_cli.steps - 1:
            print(
                f"[pid-diag] step={step:03d} "
                f"vx_b={lin_b[0,0]:.3f} vy_b={lin_b[0,1]:.3f} vz_b={lin_b[0,2]:.3f} "
                f"wz={ang_b[0,2]:.3f} z={z:.2f} "
                f"pwm=[{pwm[0]:.2f},{pwm[1]:.2f},{pwm[2]:.2f},{pwm[3]:.2f},{pwm[4]:.2f},{pwm[5]:.2f}]",
                flush=True,
            )

    lin_b = quat_apply(quat_conjugate(raw._robot.data.root_quat_w), raw._robot.data.root_lin_vel_w)
    vx = lin_b[0, 0].item()
    z = raw._robot.data.root_pos_w[0, 2].item()
    print(f"[pid-diag] final vx_b={vx:.3f} (target {cmd_v:.2f}) z={z:.2f}", flush=True)
    env.close()
    ok_dir = (cmd_v == 0.0) or (vx * cmd_v > 0.0)
    ok_mag = abs(cmd_v) < 0.1 or abs(vx) >= 0.6 * abs(cmd_v)
    ok_depth = abs(z - raw.cfg.starting_depth) < 1.0
    if not (ok_dir and ok_mag and ok_depth):
        raise SystemExit(
            f"FAIL: dir={ok_dir} mag={ok_mag} depth={ok_depth} "
            f"(vx={vx:.3f} cmd={cmd_v:.2f} z={z:.2f})"
        )
    print("[pid-diag] PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
