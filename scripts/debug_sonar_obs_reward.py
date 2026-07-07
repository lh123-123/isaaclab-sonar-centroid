#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Debug policy observation components and reward terms for Sonar IBVS-DRL.

Prints per-interval:
  - feat_err / valid / body / prev_action (69-d for Lateral1D)
  - reward breakdown: pixel, action, safety

Usage:
  python scripts/debug_sonar_obs_reward.py --headless --enable_cameras
  python scripts/debug_sonar_obs_reward.py --sonar_ui --max_steps 120 --action_mode random
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug obs + reward for sonar IBVS-DRL.")
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Lateral1D-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=120)
parser.add_argument("--interval", type=int, default=20)
parser.add_argument(
    "--action_mode",
    type=str,
    choices=("zero", "random"),
    default="zero",
    help="zero=hold still, random=uniform [-1,1] sway",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--sonar_ui",
    action=argparse.BooleanOptionalAction,
    default=False,
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.sonar_flow_hover.modules.debug import (
    ObsValidityTracker,
    format_obs_debug,
    format_reward_debug,
    print_obs_validity_report,
)
from isaaclab_tasks.direct.sonar_flow_hover.modules.rewards import (
    compute_reward_lateral_1d_terms,
    compute_reward_terms,
)
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.sonar_ui and hasattr(env_cfg, "oceansim_sonar_cfg"):
        env_cfg.oceansim_sonar_cfg.sonar_initialize_viewport = True

    print("\n======== Obs + Reward Debug ========")
    print(f"  task        : {args_cli.task}")
    print(f"  action_mode : {args_cli.action_mode}")
    print(f"  rew weights : w1={env_cfg.rew_w1_pixel} w2={env_cfg.rew_w2_action} w3={env_cfg.rew_w3_safety}")
    print(f"  safety      : depth=[{env_cfg.min_depth}, {env_cfg.max_depth}] min_valid={env_cfg.min_valid_features}")
    print(f"  obs dim     : {env_cfg.observation_space}  action dim: {env_cfg.action_space.shape[0]}")
    print(f"  obs mode    : {getattr(env_cfg, 'lateral_1d_obs_mode', 'full')}")
    print("====================================\n")

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    validity = ObsValidityTracker()
    compact = str(getattr(env_cfg, "lateral_1d_obs_mode", "full")).lower() == "compact"

    obs, _ = env.reset()

    for step in range(args_cli.max_steps + 1):
        if step % max(args_cli.interval, 1) == 0:
            obs_row = obs["policy"][0]
            validity.record(step=step, obs_row=obs_row, unwrapped=unwrapped)
            reward_terms_fn = (
                compute_reward_lateral_1d_terms
                if bool(getattr(env_cfg, "lateral_1d_mode", False))
                else compute_reward_terms
            )
            terms = reward_terms_fn(
                feature_tracker=unwrapped.feature_tracker,
                actions=unwrapped.actions,
                depth=unwrapped.robot.data.root_pos_w[:, 2],
                w1_pixel=float(env_cfg.rew_w1_pixel),
                w2_action=float(env_cfg.rew_w2_action),
                w3_safety=float(env_cfg.rew_w3_safety),
                min_depth=float(env_cfg.min_depth),
                max_depth=float(env_cfg.max_depth),
                min_valid_features=int(env_cfg.min_valid_features),
            )
            print(f"\n--- step {step} ---", flush=True)
            print(format_obs_debug(obs_row, env_id=0), flush=True)
            print(format_reward_debug(terms, env_id=0, cfg=env_cfg), flush=True)

        if step >= args_cli.max_steps:
            break

        if args_cli.action_mode == "zero":
            actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
        else:
            actions = torch.empty(env.action_space.shape, device=unwrapped.device).uniform_(-1.0, 1.0)

        obs, reward, terminated, truncated, _ = env.step(actions)
        if bool((terminated | truncated)[0].item()) and step < args_cli.max_steps:
            print(f"[debug] episode reset at step {step}", flush=True)
            obs, _ = env.reset()

    checks = validity.summarize(cfg=env_cfg, compact=compact)
    print_obs_validity_report(checks)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
