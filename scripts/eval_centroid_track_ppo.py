#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless eval for discrete PPO sonar centroid tracking checkpoint."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval PPO checkpoint on triangle target motion.")
parser.add_argument("--task", type=str, default="Isaac-SonarCentroidTrack-Eval-Direct-v0")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=(
        "logs/skrl/sonar_centroid_track/"
        "2026-07-06_18-09-52_ppo_torch_discrete_ppo_roundtrip_image/checkpoints/best_agent.pt"
    ),
)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--algorithm", type=str, default="PPO")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import skrl
from packaging import version

if version.parse(skrl.__version__) < version.parse("1.4.3"):
    raise RuntimeError(f"Unsupported skrl {skrl.__version__}")

from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.direct.sonar_centroid_track.modules.obs_correspondence import correspondence_report
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)
    agent_cfg["seed"] = int(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, agent_cfg)

    ckpt = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    print(f"[eval] Loading checkpoint: {ckpt}", flush=True)
    runner.agent.load(ckpt)
    runner.agent.set_running_mode("eval")

    raw = env.unwrapped
    w = int(raw.cfg.oceansim_sonar_cfg.width)
    h = int(raw.cfg.oceansim_sonar_cfg.height)
    max_ep = int(raw.max_episode_length)

    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    ep_success: list[bool] = []
    all_abs_dcol: list[float] = []
    all_u: list[float] = []

    n_ep = int(args_cli.episodes)
    print(f"\n=== PPO eval: {n_ep} episodes (max_len={max_ep}) ===\n", flush=True)

    obs, _ = env.reset()
    ep_r = 0.0
    ep_len = 0
    ep_idx = 0

    while ep_idx < n_ep and simulation_app.is_running():
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])

        raw.ensure_tracking()
        rep = correspondence_report(
            raw.tracker, env_id=0, prev_action=int(raw._prev_actions[0].item()), polar_width=w, polar_height=h
        )
        if rep["d_col_px"] is not None:
            all_abs_dcol.append(abs(rep["d_col_px"]))
            all_u.append(rep["u_norm"])

        obs, reward, terminated, truncated, _ = env.step(actions)
        ep_r += float(reward[0].item())
        ep_len += 1

        if bool(terminated[0].item()) or bool(truncated[0].item()):
            lost = bool(terminated[0].item())
            success = (not lost) and ep_len >= max_ep - 1
            ep_rewards.append(ep_r)
            ep_lengths.append(ep_len)
            ep_success.append(success)
            dc_s = f"{rep['d_col_px']:+.2f}" if rep["d_col_px"] is not None else "nan"
            print(
                f"ep {ep_idx + 1:2d}: len={ep_len:3d}  R={ep_r:+7.2f}  "
                f"last_d_col={dc_s}px  lost={lost}  success={success}",
                flush=True,
            )
            ep_idx += 1
            ep_r = 0.0
            ep_len = 0
            if ep_idx < n_ep:
                obs, _ = env.reset()

    env.close()

    if ep_rewards:
        sr = 100.0 * np.mean(ep_success)
        print("\n=== Summary ===", flush=True)
        print(f"  episodes       : {len(ep_rewards)}", flush=True)
        print(f"  success_rate   : {sr:.1f}%  (full triangle, no lost terminate)", flush=True)
        print(f"  reward mean    : {np.mean(ep_rewards):+.2f}  std={np.std(ep_rewards):.2f}", flush=True)
        print(f"  ep length mean : {np.mean(ep_lengths):.1f}  (max={max_ep})", flush=True)
        if all_abs_dcol:
            dc = np.array(all_abs_dcol)
            print(f"  |d_col| mean   : {dc.mean():.2f} px  p90={np.percentile(dc, 90):.2f}  max={dc.max():.2f}", flush=True)
            print(f"  |u_norm| mean  : {np.mean(np.abs(all_u)):.4f}", flush=True)
        print(flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
