#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke: kinematic ball target synced to BatchKinematicTargetLQR in isaac_auv_tracking."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AUV tracking — kinematic ball + LQR smoke.")
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped

    assert hasattr(raw, "_target"), "missing _target RigidObject"
    assert "target" in raw.scene.rigid_objects, "target not registered in scene"
    assert hasattr(raw, "pid"), "missing PID controller"
    assert int(raw.cfg.action_space) == 2, f"expected action_space=2, got {raw.cfg.action_space}"

    obs, _ = env.reset()
    pos0 = raw._target.data.root_pos_w[:, :2].clone()
    kin0 = raw._target_pos.clone()
    sync0 = torch.norm(pos0 - kin0, dim=-1).max().item()
    print(
        f"[smoke] action_dim={raw.cfg.action_space} pid=on "
        f"after reset: max |entity_xy - kin_xy| = {sync0:.6f} m",
        flush=True,
    )

    # Forward command to exercise PID (HoloOcean scale: a_v * 1.5)
    action = torch.zeros(raw.num_envs, 2, device=raw.device)
    action[:, 0] = 0.5  # → cmd_v = 0.75 m/s
    max_sync = 0.0
    for step in range(args_cli.steps):
        obs, rew, term, trunc, info = env.step(action)
        err = torch.norm(raw._target.data.root_pos_w[:, :2] - raw._target_pos, dim=-1).max().item()
        z_err = (raw._target.data.root_pos_w[:, 2] - raw.cfg.starting_depth).abs().max().item()
        max_sync = max(max_sync, err)
        if step % 10 == 0 or step == args_cli.steps - 1:
            spd = torch.norm(raw._target_state[:, 2:4], dim=-1).mean().item()
            auv_z = raw._robot.data.root_pos_w[:, 2].mean().item()
            auv_spd = torch.norm(raw._robot.data.root_lin_vel_w[:, :2], dim=-1).mean().item()
            pwm_abs = raw._pwm.abs().mean().item()
            print(
                f"[smoke] step={step:03d} sync_xy={err:.6f} ball_spd={spd:.3f} "
                f"auv_z={auv_z:.2f} auv_spd={auv_spd:.3f} pwm={pwm_abs:.3f} "
                f"rew={rew.mean().item():.3f}",
                flush=True,
            )

    pos1 = raw._target.data.root_pos_w[:, :2]
    moved = torch.norm(pos1 - pos0, dim=-1).mean().item()
    print(f"[smoke] mean ball displacement over {args_cli.steps} steps: {moved:.4f} m", flush=True)
    print(f"[smoke] max sync error: {max_sync:.6f} m", flush=True)

    ok = max_sync < 0.5 and moved > 0.05  # non-kinematic ball ~0.25 m drift over decimation
    env.close()
    if not ok:
        raise SystemExit(
            f"FAIL: sync_ok={max_sync < 0.5} moved_ok={moved > 0.05} "
            f"(max_sync={max_sync}, moved={moved})"
        )
    print("[smoke] PASS: kinematic ball tracks LQR state.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
