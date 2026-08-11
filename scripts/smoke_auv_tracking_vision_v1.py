# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Smoke: Isaac-AUV-Tracking-Direct-v1 Dict obs (image + state)."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_vision_env import AUVTrackingVisionEnvCfg


def main():
    cfg = AUVTrackingVisionEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Isaac-AUV-Tracking-Direct-v1", cfg=cfg)
    obs, _ = env.reset()
    pol = obs["policy"]
    print("[smoke] image", tuple(pol["image"].shape), pol["image"].dtype)
    print("[smoke] state", tuple(pol["state"].shape), pol["state"].dtype)
    assert pol["image"].shape[-3:] == (3, 64, 64), pol["image"].shape
    assert pol["state"].shape[-1] == 4, pol["state"].shape
    act = torch.zeros(env.unwrapped.num_envs, 2, device=env.unwrapped.device)
    for i in range(args_cli.steps):
        obs, rew, term, trunc, _ = env.step(act)
        pol = obs["policy"]
        print(f"[smoke] step{i} img={tuple(pol['image'].shape)} rew={float(rew.mean()):.3f}")
    env.close()
    print("[smoke] SMOKE_OK")


if __name__ == "__main__":
    main()
    simulation_app.close()
