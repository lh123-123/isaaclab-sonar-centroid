#!/usr/bin/env python3
"""Smoke: v5 env with RAFT estimated flow (actor input + reward + OOV reset)."""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v5-flow")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=15)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs, use_fabric=True)
    assert str(env_cfg.flow_source) == "raft", env_cfg.flow_source
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        raw = env.unwrapped
        obs, _ = env.reset()
        buf = obs["policy"]["image"]
        assert tuple(buf.shape) == (args_cli.num_envs, int(env_cfg.frame_stack) * 2 + 1, 64, 64), tuple(buf.shape)
        assert bool((buf == 0).all()), "reset stack must be zero"
        print(f"[smoke] reset OK shape={tuple(buf.shape)} flow_source={env_cfg.flow_source}", flush=True)
        act = torch.zeros(raw.action_space.shape, device=raw.device)
        act[:, 0] = 0.3
        for step in range(args_cli.steps):
            obs, rew, terminated, truncated, _ = env.step(act)
            b = obs["policy"]["image"]
            assert bool(torch.isfinite(b).all()) and bool((b.abs() <= 1.0).all())
            print(
                f"[smoke] step {step}: absmax={float(b.abs().max()):.4f} "
                f"est_ok={raw._flow_est_ok.tolist()} est_r={[round(float(v),2) for v in raw._flow_est_r]} "
                f"est_a_deg={[round(float(v)*57.3,1) for v in raw._flow_est_a]} "
                f"oov={raw._oov_streak.tolist()} rew={[round(float(v),2) for v in rew]} "
                f"term={terminated.tolist()}",
                flush=True,
            )
        print("[smoke] RESULT: OK", flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    import traceback

    try:
        rc = main()
    except BaseException:
        traceback.print_exc()
        rc = 1
    finally:
        simulation_app.close()
    sys.exit(rc)
