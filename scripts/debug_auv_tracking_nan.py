#!/usr/bin/env python3
"""Diagnose NaN sources in AUV tracking env (headless short run)."""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    cfg = parse_env_cfg("Isaac-AUV-Tracking-Direct-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make("Isaac-AUV-Tracking-Direct-v0", cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset()
    pol = obs["policy"] if isinstance(obs, dict) else obs
    print(f"[diag] obs shape={tuple(pol.shape)} finite={torch.isfinite(pol).all().item()} "
          f"min={pol.min().item():.4f} max={pol.max().item():.4f} mean={pol.mean(0).tolist()}", flush=True)
    print(f"[diag] obs std={pol.std(0).tolist()}", flush=True)

    for step in range(args_cli.steps):
        action = 2 * torch.rand(raw.num_envs, 2, device=raw.device) - 1
        obs, rew, term, trunc, info = env.step(action)
        pol = obs["policy"] if isinstance(obs, dict) else obs
        bad_obs = (~torch.isfinite(pol)).any().item()
        bad_rew = (~torch.isfinite(rew)).any().item()
        bad_act = (~torch.isfinite(action)).any().item()
        if step % 5 == 0 or bad_obs or bad_rew:
            print(
                f"[diag] step={step:03d} bad_obs={bad_obs} bad_rew={bad_rew} "
                f"rew_mean={rew.mean().item():.4f} rew_min={rew.min().item():.4f} "
                f"obs_std={pol.std(0).tolist()} "
                f"logdet_mean={pol[:,2].mean().item():.4f} observed_rate={pol[:,3].mean().item():.3f} "
                f"term={term.float().mean().item():.3f}",
                flush=True,
            )
        if bad_obs or bad_rew:
            print("[diag] FAIL: non-finite tensor detected", flush=True)
            env.close()
            raise SystemExit(1)

    # Simulate RunningStandardScaler zero-std risk on discrete channel
    std = pol.std(0)
    print(f"[diag] final obs std={std.tolist()} near0={(std < 1e-6).tolist()}", flush=True)
    env.close()
    print("[diag] PASS: env tensors finite over random rollout", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
