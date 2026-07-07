#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Debug PPO 4D obs vs Phase-2 green/yellow col convention (headless)."""

from __future__ import annotations

import argparse

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify u_norm ↔ (green_col − yellow_col)/W.")
parser.add_argument("--task", type=str, default="Isaac-SonarCentroidTrack-Discrete-PPO-Direct-v0")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--calib_settle_steps", type=int, default=8)
parser.add_argument(
    "--action_pattern",
    type=str,
    choices=("lr_cycle", "stay", "right"),
    default="lr_cycle",
    help="lr_cycle: 0,1,2,1,... to exercise prev_act.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.sonar_centroid_track.modules.constants import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
)
from isaaclab_tasks.direct.sonar_centroid_track.modules.obs_correspondence import (
    correspondence_report,
    format_correspondence_line,
)
from isaaclab_tasks.utils import parse_env_cfg


def _action_for_step(step: int, pattern: str) -> int:
    if pattern == "stay":
        return ACTION_STAY
    if pattern == "right":
        return ACTION_RIGHT
    seq = (ACTION_LEFT, ACTION_STAY, ACTION_RIGHT, ACTION_STAY)
    return seq[step % len(seq)]


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.print_policy_obs_debug = False
    env_cfg.immediate_lost_terminate = False
    env_cfg.debug_disable_lost_terminate = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    w = int(unwrapped.cfg.oceansim_sonar_cfg.width)
    h = int(unwrapped.cfg.oceansim_sonar_cfg.height)

    print("\n=== PPO obs correspondence debug ===\n", flush=True)
    print(f"  polar W={w} H={h}  (u_norm = d_col_px/W, v_norm = d_row_px/H)", flush=True)
    print(
        "  green=yellow fix  |  u>0 → green RIGHT → action right(2)",
        flush=True,
    )
    print(f"{'step':>5} {'action':>6} {'d_col':>8} {'u_norm':>10} {'valid':>5} {'prev':>8} match", flush=True)

    env.reset()
    stay = torch.tensor([ACTION_STAY], device=unwrapped.device, dtype=torch.long)
    for _ in range(int(args_cli.calib_settle_steps)):
        env.step(stay)

    n_fail = 0
    n_ok = 0
    prev_action = ACTION_STAY

    for step in range(int(args_cli.steps)):
        act_id = _action_for_step(step, args_cli.action_pattern)
        action = torch.tensor([act_id], device=unwrapped.device, dtype=torch.long)
        obs, _, terminated, truncated, _ = env.step(action)
        if bool(terminated.item()) or bool(truncated.item()):
            print(f"  step={step} episode end (terminated={terminated.item()} truncated={truncated.item()})", flush=True)
            env.reset()
            for _ in range(int(args_cli.calib_settle_steps)):
                env.step(stay)
            continue

        unwrapped.ensure_tracking()
        prev_for_obs = int(unwrapped._prev_actions[0].item())
        rep = correspondence_report(
            unwrapped.tracker,
            env_id=0,
            prev_action=prev_for_obs,
            polar_width=w,
            polar_height=h,
        )
        o = obs["policy"][0].detach().cpu().tolist()
        ref = rep["policy_obs_4d"]
        obs_ok = all(abs(o[k] - ref[k]) < 1.0e-5 for k in range(4))
        uv_ok = bool(rep["u_match"] and rep["v_match"])
        ok = obs_ok and uv_ok
        n_ok += int(ok)
        n_fail += int(not ok)

        dc = rep["d_col_px"]
        dc_s = f"{dc:+.2f}" if dc is not None else "nan"
        act_name = ("left", "stay", "right")[act_id]
        prev_name = ("left", "stay", "right")[prev_for_obs]
        flag = "OK" if ok else "FAIL"
        print(
            f"{step:5d} {act_name:>6} {dc_s:>8} {rep['u_norm']:>+10.6f} "
            f"{rep['valid_ratio']:>5.0f} {prev_name:>8} {flag}",
            flush=True,
        )
        if not ok:
            print(f"         {format_correspondence_line(rep)}", flush=True)
            print(f"         obs={o} ref={ref} u_match={rep['u_match']} v_match={rep['v_match']}", flush=True)

    print(f"\n=== Summary: {n_ok} OK / {n_fail} FAIL over {n_ok + n_fail} steps ===\n", flush=True)
    if n_fail == 0:
        print("VERDICT: PASS — u_norm matches (green_col − yellow_col)/W\n", flush=True)
    else:
        print("VERDICT: FAIL — see mismatches above\n", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
