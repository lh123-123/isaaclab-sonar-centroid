#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate Sonar IBVS-DRL: zero-action smoke test or trained skrl checkpoint."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate sonar IBVS-DRL env.")
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Eval-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=300, help="Simulation steps before exit.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to skrl .pt checkpoint (optional).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.print_ibvs_debug = False

    use_policy = bool(args_cli.checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    agent = None
    if use_policy:
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        env = SkrlVecEnvWrapper(env, ml_framework="torch")
        runner = Runner(env, agent_cfg)
        print(f"[eval] loading checkpoint: {args_cli.checkpoint}", flush=True)
        runner.agent.load(args_cli.checkpoint)
        runner.agent.set_running_mode("eval")
        agent = runner.agent
        obs, _ = env.reset()
    else:
        env.reset()
        obs = None

    n_envs = unwrapped.num_envs
    dev = unwrapped.device
    ft = unwrapped.feature_tracker

    print(
        f"[eval] task={args_cli.task} num_envs={n_envs} max_steps={args_cli.max_steps} "
        f"policy={'trained' if use_policy else 'zero-action'}",
        flush=True,
    )
    print(f"[eval] obs_space={env.observation_space} action_space={env.action_space}", flush=True)

    valid_sum = torch.zeros(n_envs, device=dev)
    pixel_err_sum = torch.zeros(n_envs, device=dev)
    pos_err_sum = torch.zeros(n_envs, device=dev)
    reward_sum = torch.zeros(n_envs, device=dev)
    vel_sum = torch.zeros(n_envs, device=dev)

    for step in range(args_cli.max_steps):
        if use_policy:
            with torch.inference_mode():
                outputs = agent.act(obs, timestep=step, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, reward, terminated, truncated, _ = env.step(actions)
        else:
            actions = torch.zeros(env.action_space.shape, device=dev)
            _, reward, terminated, truncated, _ = env.step(actions)

        valid_sum += ft.valid_count().float()
        pixel_err_sum += ft.normalized_pixel_error()
        pos_err_sum += unwrapped._hover_pos_error_norm()
        rew = reward if isinstance(reward, torch.Tensor) else torch.tensor(reward, device=dev)
        reward_sum += rew.view(-1)[:n_envs]
        vel_sum += torch.norm(unwrapped.robot.data.root_lin_vel_b, dim=1)

        if step % 50 == 0:
            print(
                f"[eval] step={step:4d} "
                f"valid_pts={int(ft.valid_count()[0].item())} "
                f"pixel_err={ft.normalized_pixel_error()[0].item():.4f} "
                f"pos_err={unwrapped._hover_pos_error_norm()[0].item():.3f}m "
                f"vel={torch.norm(unwrapped.robot.data.root_lin_vel_b[0]).item():.3f} "
                f"reward={reward[0].item():+.3f}",
                flush=True,
            )

        if (terminated | truncated).any() and step > 10:
            print(f"[eval] episode ended at step={step}", flush=True)
            if use_policy:
                obs, _ = env.reset()
            else:
                with torch.no_grad():
                    env.reset()

    steps = float(args_cli.max_steps)
    tail_pixel = ft.normalized_pixel_error()[0].item()
    tail_valid = int(ft.valid_count()[0].item())
    tail_pos = unwrapped._hover_pos_error_norm()[0].item()
    tail_vel = torch.norm(unwrapped.robot.data.root_lin_vel_b[0]).item()

    mean_valid = (valid_sum / steps).mean().item()
    mean_pixel = (pixel_err_sum / steps).mean().item()
    mean_pos = (pos_err_sum / steps).mean().item()
    mean_reward = (reward_sum / steps).mean().item()
    mean_vel = (vel_sum / steps).mean().item()

    min_valid = int(getattr(env_cfg, "min_valid_features", 2))
    homed = tail_pixel < 0.05 and tail_vel < 0.08 and tail_valid >= min_valid

    print("\n========== Sonar IBVS-DRL Eval Summary ==========")
    print(f"  policy             : {'trained' if use_policy else 'zero-action'}")
    if use_policy:
        print(f"  checkpoint         : {args_cli.checkpoint}")
    print(f"  steps              : {args_cli.max_steps}")
    print(f"  mean valid points  : {mean_valid:.1f}  (min required: {min_valid})")
    print(f"  mean reward        : {mean_reward:+.4f}")
    print(f"  mean pixel error   : {mean_pixel:.4f}")
    print(f"  mean pos error (m) : {mean_pos:.3f}")
    print(f"  mean |vel| (m/s)   : {mean_vel:.3f}")
    print(f"  final pixel error  : {tail_pixel:.4f}")
    print(f"  final valid points : {tail_valid}")
    print(f"  final pos error(m) : {tail_pos:.3f}")
    print(f"  final |vel| (m/s)  : {tail_vel:.3f}")
    print(f"  homing success     : {'YES' if homed else 'NO'}")
    print("=================================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
