# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Smoke: Isaac-AUV-Tracking-Direct-v2 Dict obs (flow image + 8D state)."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_flow_env import AUVTrackingFlowEnvCfg


def main():
    cfg = AUVTrackingFlowEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Isaac-AUV-Tracking-Direct-v2", cfg=cfg)
    obs, _ = env.reset()
    pol = obs["policy"]
    print("[smoke] image", tuple(pol["image"].shape), pol["image"].dtype)
    print("[smoke] state", tuple(pol["state"].shape), pol["state"].dtype)
    assert pol["image"].shape[-3:] == (3, 64, 64), pol["image"].shape
    assert pol["state"].shape[-1] == 8, pol["state"].shape
    raw = env.unwrapped
    # First obs after reset: no previous frame → valid_flow = 0
    assert not bool(raw._flow_valid_flow.any()), "reset must zero valid_flow"
    act = torch.zeros(raw.num_envs, 2, device=raw.device)
    for i in range(args_cli.steps):
        obs, rew, term, trunc, extras = env.step(act)
        pol = obs["policy"]
        logdet = float(raw._log_det_cov().mean())
        print(
            f"[smoke] step{i} img={tuple(pol['image'].shape)} "
            f"rew={float(rew.mean()):.3f} logdet={logdet:.3f} "
            f"valid_flow={int(raw._flow_valid_flow.sum())}/{raw.num_envs}"
        )
        assert pol["image"].shape[-3:] == (3, 64, 64)
        assert pol["state"].shape[-1] == 8
    env.close()
    print("[smoke] SMOKE_OK")


if __name__ == "__main__":
    main()
    simulation_app.close()
