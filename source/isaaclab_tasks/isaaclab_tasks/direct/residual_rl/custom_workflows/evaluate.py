# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch evaluation: PD-only baseline vs SAC residual policy."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate ResidualRL: PD-only vs SAC residual.")
parser.add_argument("--task", type=str, default="Isaac-ResidualRL-Direct-v0", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (recommend 1 for eval).")
parser.add_argument("--episodes", type=int, default=20, help="Episodes per mode.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
parser.add_argument(
    "--stage",
    type=int,
    default=None,
    choices=[1, 2, 3, 4],
    help="Override curriculum stage via task variant.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="SAC checkpoint for residual mode. PD-only always runs with zero actions.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="CSV output path. Default: logs/residual_rl_eval/<timestamp>.csv",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import skrl
import isaaclab.utils.math as math_utils
from packaging import version

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise RuntimeError(f"Unsupported skrl version: {skrl.__version__} (need >={SKRL_VERSION})")

from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _resolve_task() -> str:
    if args_cli.stage is not None:
        if args_cli.stage == 1:
            return "Isaac-ResidualRL-Direct-v0"
        return f"Isaac-ResidualRL-Stage{args_cli.stage}-Direct-v0"
    return args_cli.task


def _aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0].keys() if k not in ("mode", "episode")]
    out: dict[str, float] = {"episodes": float(len(rows))}
    for key in keys:
        out[key] = sum(float(r[key]) for r in rows) / len(rows)
    out["success_rate"] = sum(float(r["success"]) for r in rows) / len(rows)
    return out


def _run_pd_only(env, episodes: int) -> list[dict]:
    rows: list[dict] = []
    obs, _ = env.reset()
    ep = 0
    ep_steps = 0
    ep_pos_sum = 0.0
    ep_ang_sum = 0.0
    ep_reward = 0.0
    ep_success = False

    unwrapped = env.unwrapped
    zero_action = torch.zeros(unwrapped.num_envs, unwrapped.cfg.num_actions, device=unwrapped.device)

    while ep < episodes and simulation_app.is_running():
        with torch.inference_mode():
            obs, reward, terminated, truncated, _ = env.step(zero_action)
        pos_err = float(unwrapped._position_error_norm()[0].item())
        ang_err = float(
            math_utils.quat_error_magnitude(
                unwrapped._goal[0:1], unwrapped._robot.data.root_quat_w[0:1]
            ).item()
        )
        ep_steps += 1
        ep_pos_sum += pos_err
        ep_ang_sum += ang_err
        ep_reward += float(reward[0].item())
        if pos_err < unwrapped.cfg.success_pos_threshold:
            ep_success = True

        done = bool(terminated[0].item() or truncated[0].item())
        if done:
            rows.append(
                {
                    "mode": "pd_only",
                    "episode": ep,
                    "steps": ep_steps,
                    "mean_pos_err_m": ep_pos_sum / max(ep_steps, 1),
                    "mean_ang_err_rad": ep_ang_sum / max(ep_steps, 1),
                    "final_pos_err_m": pos_err,
                    "total_reward": ep_reward,
                    "mean_u_rl_norm": 0.0,
                    "success": float(ep_success),
                }
            )
            ep += 1
            ep_steps = 0
            ep_pos_sum = 0.0
            ep_ang_sum = 0.0
            ep_reward = 0.0
            ep_success = False
            obs, _ = env.reset()

    return rows


def _run_sac(env, runner, episodes: int) -> list[dict]:
    rows: list[dict] = []
    obs, _ = env.reset()
    ep = 0
    ep_steps = 0
    ep_pos_sum = 0.0
    ep_ang_sum = 0.0
    ep_reward = 0.0
    ep_u_rl_sum = 0.0
    ep_success = False

    unwrapped = env.unwrapped
    runner.agent.set_running_mode("eval")

    while ep < episodes and simulation_app.is_running():
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, reward, terminated, truncated, _ = env.step(actions)

        pos_err = float(unwrapped._position_error_norm()[0].item())
        ang_err = float(
            math_utils.quat_error_magnitude(
                unwrapped._goal[0:1], unwrapped._robot.data.root_quat_w[0:1]
            ).item()
        )
        u_rl_norm = float(torch.norm(unwrapped._u_rl[0]).item())
        ep_steps += 1
        ep_pos_sum += pos_err
        ep_ang_sum += ang_err
        ep_reward += float(reward[0].item())
        ep_u_rl_sum += u_rl_norm
        if pos_err < unwrapped.cfg.success_pos_threshold:
            ep_success = True

        done = bool(terminated[0].item() or truncated[0].item())
        if done:
            rows.append(
                {
                    "mode": "residual_sac",
                    "episode": ep,
                    "steps": ep_steps,
                    "mean_pos_err_m": ep_pos_sum / max(ep_steps, 1),
                    "mean_ang_err_rad": ep_ang_sum / max(ep_steps, 1),
                    "final_pos_err_m": pos_err,
                    "total_reward": ep_reward,
                    "mean_u_rl_norm": ep_u_rl_sum / max(ep_steps, 1),
                    "success": float(ep_success),
                }
            )
            ep += 1
            ep_steps = 0
            ep_pos_sum = 0.0
            ep_ang_sum = 0.0
            ep_reward = 0.0
            ep_u_rl_sum = 0.0
            ep_success = False
            obs, _ = env.reset()

    return rows


@hydra_task_config(_resolve_task(), "skrl_sac_cfg_entry_point")
def main(env_cfg, experiment_cfg: dict):
    task_name = _resolve_task()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.eval_mode = True
    env_cfg.print_eval_debug = False
    env_cfg.eval_silent = True
    experiment_cfg["seed"] = args_cli.seed

    env = gym.make(task_name, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    all_rows: list[dict] = []

    print(f"[INFO] Task={task_name} episodes={args_cli.episodes} seed={args_cli.seed}")
    print("[INFO] Running PD-only baseline (zero residual)...")
    all_rows.extend(_run_pd_only(env, args_cli.episodes))

    if args_cli.checkpoint:
        log_root = os.path.abspath(
            os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
        )
        resume_path = os.path.abspath(args_cli.checkpoint)
        if not os.path.isfile(resume_path):
            resume_path = get_checkpoint_path(
                log_root, run_dir=".*_sac_torch", other_dirs=["checkpoints"]
            )
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        runner = Runner(env, experiment_cfg)
        print(f"[INFO] Loading SAC checkpoint: {resume_path}")
        runner.agent.load(resume_path)
        print("[INFO] Running SAC residual policy...")
        all_rows.extend(_run_sac(env, runner, args_cli.episodes))
    else:
        print("[INFO] No --checkpoint provided; skipping SAC residual evaluation.")

    env.close()

    pd_rows = [r for r in all_rows if r["mode"] == "pd_only"]
    sac_rows = [r for r in all_rows if r["mode"] == "residual_sac"]
    pd_agg = _aggregate(pd_rows)
    sac_agg = _aggregate(sac_rows)

    print("\n========== ResidualRL Evaluation Summary ==========")
    if pd_agg:
        print(
            f"PD-only:  mean_pos_err={pd_agg['mean_pos_err_m']:.3f}m "
            f"final_pos_err={pd_agg['final_pos_err_m']:.3f}m "
            f"success_rate={pd_agg['success_rate']:.1%} reward={pd_agg['total_reward']:+.1f}"
        )
    if sac_agg:
        print(
            f"SAC+PD:   mean_pos_err={sac_agg['mean_pos_err_m']:.3f}m "
            f"final_pos_err={sac_agg['final_pos_err_m']:.3f}m "
            f"success_rate={sac_agg['success_rate']:.1%} reward={sac_agg['total_reward']:+.1f} "
            f"mean|u_rl|={sac_agg['mean_u_rl_norm']:.3f}"
        )
    if pd_agg and sac_agg:
        improve = (pd_agg["mean_pos_err_m"] - sac_agg["mean_pos_err_m"]) / max(pd_agg["mean_pos_err_m"], 1e-6)
        print(f"Improvement (mean pos err): {improve * 100:.1f}%")
    print("===================================================\n")

    out_path = args_cli.output
    if out_path is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.abspath(os.path.join("logs", "residual_rl_eval"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ts}_stage{args_cli.stage or 'default'}.csv")
    else:
        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[INFO] Saved per-episode results to: {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
