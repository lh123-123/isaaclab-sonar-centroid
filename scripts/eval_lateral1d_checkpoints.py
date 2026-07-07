#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless rollout comparison for Lateral1D checkpoints (+ zero-action baseline)."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare Lateral1D checkpoints headless.")
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Lateral1D-Direct-v0")
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--checkpoints", type=str, nargs="*", default=None)
parser.add_argument("--algorithm", type=str, default="SAC")
parser.add_argument("--print_interval", type=int, default=50, help="Print metrics every N steps (0=summary only).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.direct.sonar_flow_hover.modules.debug import collect_metrics
from isaaclab_tasks.utils.hydra import hydra_task_config

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = f"skrl_{algorithm}_cfg_entry_point"


def _resolve_checkpoints(run_dir: str, names: list[str] | None) -> list[tuple[str, str]]:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if names:
        return [(n, os.path.join(ckpt_dir, n if n.endswith(".pt") else f"{n}.pt")) for n in names]
    candidates = [
        ("best_agent", "best_agent.pt"),
        ("agent_120000", "agent_120000.pt"),
        ("agent_150000", "agent_150000.pt"),
    ]
    return [(label, os.path.join(ckpt_dir, fname)) for label, fname in candidates if os.path.isfile(os.path.join(ckpt_dir, fname))]


def _rollout(env, unwrapped, env_cfg, agent_cfg, label: str, checkpoint: str | None, steps: int) -> dict:
  if checkpoint is not None:
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    cfg = dict(agent_cfg)
    cfg["trainer"] = dict(cfg.get("trainer", {}))
    cfg["trainer"]["close_environment_at_exit"] = False
    cfg["agent"]["experiment"]["write_interval"] = 0
    cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wrapped, cfg)
    runner.agent.load(os.path.abspath(checkpoint))
    runner.agent.set_running_mode("eval")
    obs, _ = wrapped.reset()
    step_env = wrapped
  else:
    runner = None
    obs, _ = env.reset()
    step_env = env

  centroid_lat_errs: list[float] = []
  centroid_dx_px: list[float] = []
  centroid_dy_px: list[float] = []
  robot_sway_m: list[float] = []
  target_dy_m: list[float] = []
  sway_error_m: list[float] = []
  rewards: list[float] = []

  for step in range(steps):
    if runner is not None:
      with torch.inference_mode():
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
    else:
      actions = torch.zeros(env.action_space.shape, device=unwrapped.device)

    obs, rew, terminated, truncated, _ = step_env.step(actions)

    m = collect_metrics(
        pipeline=unwrapped.pipeline,
        robot=unwrapped.robot,
        station_origin_w=unwrapped._station_origin_w,
        actions=unwrapped.actions,
        reference_mode=unwrapped.ref_mgr.mode,
        kinematic_offset_w=unwrapped._kinematic_offset_w,
        target_kinematic_offset_b=unwrapped._target_kinematic_offset_b,
        centroid_err_px=unwrapped._geometric_centroid_errors_px(),
        lateral_1d=True,
        env_id=0,
    )

    centroid_lat_errs.append(float(m["centroid_lat_px"]))
    centroid_dx_px.append(float(m["centroid_dx"]))
    centroid_dy_px.append(float(m["centroid_dy"]))
    robot_sway_m.append(float(m["robot_sway_m"]))
    target_dy_m.append(float(m["target_dy_m"]))
    sway_error_m.append(float(m["sway_error_m"]))
    rewards.append(float(rew[0].item()))

    interval = max(int(args_cli.print_interval), 0)
    if interval > 0 and step % interval == 0:
      print(
          f"  step={step:4d} "
          f"centroid_px=(dx={m['centroid_dx']:+.2f}, dy={m['centroid_dy']:+.2f}) "
          f"ref=({m['ref_centroid_xy'][0]:.1f},{m['ref_centroid_xy'][1]:.1f}) "
          f"track=({m['tracked_centroid_xy'][0]:.1f},{m['tracked_centroid_xy'][1]:.1f}) "
          f"centroid_daz_px={m['centroid_daz_px']:+.2f} "
          f"sway={m['robot_sway_m']:+.3f} target_dy={m['target_dy_m']:+.3f} "
          f"sway_err={m['sway_error_m']:+.3f} "
          f"reward={float(rew[0].item()):+.4f} valid={m['valid_pts']} "
          f"action={m['action'][0]:+.3f}",
          flush=True,
      )

    if bool((terminated | truncated)[0].item()):
      obs, _ = step_env.reset()

  def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

  def _mean_abs(xs: list[float]) -> float:
    return sum(abs(x) for x in xs) / len(xs) if xs else 0.0

  return {
    "label": label,
    "centroid_lat_mean": _mean(centroid_lat_errs),
    "centroid_lat_max": max(centroid_lat_errs) if centroid_lat_errs else 0.0,
    "centroid_dx_mean": _mean(centroid_dx_px),
    "centroid_dy_mean": _mean(centroid_dy_px),
    "robot_sway_mean": _mean(robot_sway_m),
    "target_dy_mean": _mean(target_dy_m),
    "sway_error_mean": _mean(sway_error_m),
    "sway_error_abs_mean": _mean_abs(sway_error_m),
    "sway_error_abs_max": max(abs(x) for x in sway_error_m) if sway_error_m else 0.0,
    "reward_mean": _mean(rewards),
    "valid_n": int(unwrapped.feature_tracker.valid_count()[0].item()),
  }


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    ckpts = _resolve_checkpoints(args_cli.run_dir, args_cli.checkpoints)
    plan: list[tuple[str, str | None]] = [("zero_action", None)] + ckpts

    print("\n======== Checkpoint Comparison ========", flush=True)
    print(f"  run_dir : {args_cli.run_dir}", flush=True)
    print(f"  steps   : {args_cli.steps}", flush=True)
    print("======================================\n", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    results: list[dict] = []

    for label, path in plan:
        print(f"--- eval {label} ---", flush=True)
        if path:
            print(f"  ckpt: {path}", flush=True)
        m = _rollout(env, unwrapped, env_cfg, agent_cfg, label, path, args_cli.steps)
        results.append(m)
        print(
            f"  centroid_daz_px mean={m['centroid_lat_mean']:+.2f} max={m['centroid_lat_max']:+.2f}  "
            f"centroid_px mean=(dx={m['centroid_dx_mean']:+.2f}, dy={m['centroid_dy_mean']:+.2f})",
            flush=True,
        )
        print(
            f"  robot_sway mean={m['robot_sway_mean']:+.3f}  target_dy mean={m['target_dy_mean']:+.3f}  "
            f"sway_err mean={m['sway_error_mean']:+.3f} abs_mean={m['sway_error_abs_mean']:.3f} "
            f"abs_max={m['sway_error_abs_max']:.3f}",
            flush=True,
        )
        print(
            f"  reward_mean={m['reward_mean']:.4f} valid={m['valid_n']}",
            flush=True,
        )

    print("\n======== Summary (lower centroid_lat & |sway_err| is better) ========", flush=True)
    for i, r in enumerate(sorted(results, key=lambda x: (x["centroid_lat_mean"], x["sway_error_abs_mean"])), 1):
        print(
            f"  #{i} {r['label']:14s} "
            f"centroid_lat={r['centroid_lat_mean']:.4f}  "
            f"centroid_px=(dx={r['centroid_dx_mean']:+.2f},dy={r['centroid_dy_mean']:+.2f})  "
            f"sway_err_abs={r['sway_error_abs_mean']:.3f}  "
            f"reward/step={r['reward_mean']:.4f}",
            flush=True,
        )
    print("====================================================\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
