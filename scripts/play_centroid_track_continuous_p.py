#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GUI demo: continuous vy control on sine target (image P → Box action)."""

from __future__ import annotations

import argparse
import time

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Continuous P-control on centroid track env (GUI).")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SonarCentroidTrack-Continuous-Eval-Direct-v0",
)
parser.add_argument("--kp", type=float, default=1.0, help="P gain on (green_col − yellow_col)/W.")
parser.add_argument("--kff", type=float, default=1.0, help="Feedforward on target lateral velocity.")
parser.add_argument("--deadband", type=float, default=0.003)
parser.add_argument("--max_vy", type=float, default=0.0, help="0 = use env max_lateral_velocity.")
parser.add_argument("--step_sleep", type=float, default=0.02, help="Pause between steps (GUI).")
parser.add_argument("--sonar_ui", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.headless:
    args_cli.enable_cameras = True
else:
    args_cli.sonar_ui = False
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.sonar_centroid_track.modules.ibvs_p_controller import image_delta_lateral_velocity
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    max_v = float(args_cli.max_vy) if args_cli.max_vy > 0.0 else float(unwrapped.cfg.max_lateral_velocity)
    polar_w = float(unwrapped.cfg.oceansim_sonar_cfg.width)
    max_ep = int(unwrapped.max_episode_length)

    obs, _ = env.reset()
    ep_r = 0.0
    step = 0

    print(
        f"[continuous_p] task={args_cli.task} kp={args_cli.kp} kff={args_cli.kff} "
        f"max_vy={max_v:.3f} max_ep={max_ep}",
        flush=True,
    )

    while simulation_app.is_running() and step < max_ep:
        unwrapped.ensure_tracking()
        p_rel = unwrapped.tracker.detected_rel(env_id=0)
        u_norm = float(p_rel.col) / polar_w if p_rel is not None else 0.0
        tgt_vy = float(unwrapped._target_dy_vel_m()[0].item())

        vy = image_delta_lateral_velocity(
            u_norm,
            tgt_vy,
            kp=float(args_cli.kp),
            kff=float(args_cli.kff),
            max_vy=max_v,
            deadband=float(args_cli.deadband),
        )
        action = torch.tensor([[vy / max_v]], device=unwrapped.device, dtype=torch.float32)

        obs, reward, terminated, truncated, _ = env.step(action)
        ep_r += float(reward[0].item())
        step += 1

        if step % 20 == 0:
            dc = float(p_rel.col) if p_rel is not None else float("nan")
            print(
                f"  step={step:3d} u_norm={u_norm:+.4f} vy={vy:+.3f} a={action[0, 0].item():+.3f} "
                f"R={ep_r:+.2f}",
                flush=True,
            )

        if args_cli.step_sleep > 0.0:
            time.sleep(args_cli.step_sleep)

        if bool(terminated[0].item()) or bool(truncated[0].item()):
            lost = bool(terminated[0].item())
            print(f"[done] steps={step} R={ep_r:+.2f} lost={lost}", flush=True)
            obs, _ = env.reset()
            ep_r = 0.0
            step = 0

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
