#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""1D lateral follow debug: target sweeps body-Y, AUV sways via IBVS P-control."""

from __future__ import annotations

import argparse
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug 1D lateral sonar IBVS following.")
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Lateral1D-Debug-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--lateral_amp_m",
    type=float,
    default=None,
    help="Target body-Y sinusoid amplitude (m). Default: cfg.lateral_target_amp_m.",
)
parser.add_argument(
    "--lateral_period_s",
    type=float,
    default=None,
    help="Target sinusoid period (s). Default: cfg.lateral_target_period_s.",
)
parser.add_argument(
    "--follow_gain",
    type=float,
    default=3.0,
    help="P-gain on mean horizontal pixel error -> sway action.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=300,
    help="Simulation steps (0 = until window closed).",
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
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _lateral_sway_action(unwrapped, gain: float) -> torch.Tensor:
    """Map mean valid horizontal pixel error to 1-d sway action."""
    actions = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)
    ft = unwrapped.feature_tracker
    err = ft.feature_errors()[0]
    valid = ft.valid[0]
    if not bool(valid.any().item()):
        return actions
    mean_ex = err[valid, 0].mean() / ft.diagonal
    sway = torch.clamp(mean_ex * gain, -1.0, 1.0)
    if actions.shape[-1] == 1:
        actions[0, 0] = sway
    else:
        actions[0, 1] = sway
    return actions


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.sonar_ui and hasattr(env_cfg, "oceansim_sonar_cfg"):
        env_cfg.oceansim_sonar_cfg.sonar_initialize_viewport = True
    env_cfg.print_ibvs_debug = True
    env_cfg.ibvs_debug_interval = 20

    amp = float(args_cli.lateral_amp_m if args_cli.lateral_amp_m is not None else env_cfg.lateral_target_amp_m)
    period = float(
        args_cli.lateral_period_s if args_cli.lateral_period_s is not None else env_cfg.lateral_target_period_s
    )
    omega = 2.0 * math.pi / max(period, 1.0e-3)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    dt = float(getattr(env, "step_dt", unwrapped.step_dt))

    print("\n======== 1D Lateral IBVS Follow Debug ========")
    print(f"  task          : {args_cli.task}")
    print(f"  target sway   : body-Y sin ±{amp:.2f}m  period={period:.1f}s")
    print(f"  follow_gain   : {args_cli.follow_gain}")
    print(f"  AUV motion    : sway only (surge/heave/yaw locked)")
    print("==============================================\n")

    env.reset()
    unwrapped.set_displacement_offset(0.0, 0.0)
    unwrapped.set_target_velocity(0.0, 0.0)
    unwrapped.set_smooth_target_offset(0.0, 0.0, 0.0)

    step = 0
    while simulation_app.is_running():
        t0 = time.time()
        phase = step * dt * omega
        dy = amp * math.sin(phase)
        unwrapped.set_smooth_target_offset(0.0, dy, 0.0)

        actions = _lateral_sway_action(unwrapped, args_cli.follow_gain)
        env.step(actions)

        if step % 20 == 0:
            m = unwrapped.get_ibvs_metrics(env_id=0)
            robot_y = float(unwrapped.robot.data.root_pos_w[0, 1].item())
            target_dy = float(unwrapped._target_kinematic_offset_b[0, 1].item())
            sway = actions[0, 0].item() if actions.shape[-1] == 1 else actions[0, 1].item()
            print(
                f"[lat1d] step={step:4d} target_dy={target_dy:+.3f}m "
                f"robot_world_y={robot_y:+.3f} valid={m['valid_pts']} "
                f"pixel_err={m['pixel_err']:.4f} sway_action={sway:+.3f}",
                flush=True,
            )

        step += 1
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            m = unwrapped.get_ibvs_metrics(env_id=0)
            print(
                f"[lat1d] done steps={step} pixel_err={m['pixel_err']:.4f} "
                f"valid_pts={m['valid_pts']}",
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
