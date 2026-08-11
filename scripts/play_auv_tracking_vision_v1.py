# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Finite-step headless play for Isaac-AUV-Tracking-Direct-v1 (CustomSAC)."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--log_every", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_vision_env import AUVTrackingVisionEnvCfg


def main():
    cfg = AUVTrackingVisionEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Isaac-AUV-Tracking-Direct-v1", cfg=cfg)
    vec = Sb3VecEnvWrapper(env, fast_variant=True)

    print(f"[play] loading {args_cli.checkpoint}")
    agent = SAC.load(args_cli.checkpoint, env=vec, print_system_info=False)

    obs = vec.reset()
    rews = []
    ep_rets = []
    ep_lens = []
    cur_ret = np.zeros(vec.num_envs, dtype=np.float64)
    cur_len = np.zeros(vec.num_envs, dtype=np.int32)

    for t in range(args_cli.steps):
        actions, _ = agent.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec.step(actions)
        r = np.asarray(rewards, dtype=np.float64)
        rews.append(float(r.mean()))
        cur_ret += r
        cur_len += 1
        done = np.asarray(dones, dtype=bool)
        if done.any():
            for i in np.where(done)[0]:
                ep_rets.append(float(cur_ret[i]))
                ep_lens.append(int(cur_len[i]))
                cur_ret[i] = 0.0
                cur_len[i] = 0
        if t % args_cli.log_every == 0 or t == args_cli.steps - 1:
            print(f"[play] step {t:04d} rew={rews[-1]:.3f} running_ep_ret={float(cur_ret.mean()):.1f}")

    print(
        f"[play] steps={args_cli.steps} mean_step_rew={float(np.mean(rews)):.3f} "
        f"n_episodes={len(ep_rets)}"
    )
    if ep_rets:
        print(
            f"[play] ep_ret mean={float(np.mean(ep_rets)):.1f} "
            f"min={float(np.min(ep_rets)):.1f} max={float(np.max(ep_rets)):.1f} "
            f"ep_len mean={float(np.mean(ep_lens)):.1f}"
        )
    else:
        print(f"[play] no full episode finished; partial_ret={float(cur_ret.mean()):.1f} len={int(cur_len.mean())}")
    env.close()
    print("[play] DONE")


if __name__ == "__main__":
    main()
    simulation_app.close()
