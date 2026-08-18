#!/usr/bin/env python3
"""Headless smoke: Direct-v0 with uniform ocean current enabled."""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    cfg = parse_env_cfg("Isaac-AUV-Tracking-Direct-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.ocean_current.enabled = True
    cfg.ocean_current.type = "uniform"
    cfg.ocean_current.strength = 0.5
    cfg.ocean_current.level = None
    cfg.ocean_current.params = {"direction": [1.0, 0.0, 0.0]}
    env = gym.make("Isaac-AUV-Tracking-Direct-v0", cfg=cfg)
    raw = env.unwrapped
    env.reset()
    print(
        f"[smoke-current] field={raw._ocean_current_field is not None} "
        f"enabled={raw.cfg.ocean_current.enabled} strength={raw.cfg.ocean_current.strength}",
        flush=True,
    )

    for step in range(args_cli.steps):
        action = 2 * torch.rand(raw.num_envs, 2, device=raw.device) - 1
        env.step(action)
        if step % 25 == 0 or step == args_cli.steps - 1:
            v = raw.get_ocean_currents()
            nrm = v.norm(dim=-1).mean().item()
            print(
                f"[smoke-current] step={step:03d} ||v_current||_mean={nrm:.4f} "
                f"v0={v[0].tolist()}",
                flush=True,
            )

    v = raw.get_ocean_currents()
    nrm = v.norm(dim=-1).mean().item()
    env.close()
    if abs(nrm - 0.5) > 0.05:
        print(f"[smoke-current] FAIL: expected ||v||≈0.5, got {nrm:.4f}", flush=True)
        raise SystemExit(1)
    print(f"[smoke-current] PASS ||v_current||_mean={nrm:.4f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
