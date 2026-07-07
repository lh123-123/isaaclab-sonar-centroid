# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug one Test1 episode: log AUV world pose, error, and thruster commands."""

from __future__ import annotations

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Log ResidualRL Test1 AUV trajectory (PD-only by default).")
parser.add_argument("--steps", type=int, default=500, help="Max simulation steps to log.")
parser.add_argument("--interval", type=int, default=10, help="Print/console every N steps.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=str, default=None, help="CSV output path.")
parser.add_argument("--pd-only", action="store_true", default=True, help="Zero SAC residual (default).")
parser.add_argument("--checkpoint", type=str, default=None, help="Optional SAC checkpoint.")
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

if version.parse(skrl.__version__) < version.parse("1.4.3"):
    raise RuntimeError(f"Unsupported skrl: {skrl.__version__}")

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

if args_cli.checkpoint:
    from skrl.utils.runner.torch import Runner

TEST1_TASK = "Isaac-ResidualRL-Test1-Direct-v0"


def _local_pos(env, world_pos: torch.Tensor) -> tuple[float, float, float]:
    origin = env.scene.env_origins[0]
    p = (world_pos - origin).tolist()
    return float(p[0]), float(p[1]), float(p[2])


@hydra_task_config(TEST1_TASK, "skrl_sac_cfg_entry_point")
def main(env_cfg, experiment_cfg: dict):
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.debug_vis = True
    experiment_cfg["seed"] = args_cli.seed

    env = gym.make(TEST1_TASK, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    u = env.unwrapped

    runner = None
    if args_cli.checkpoint:
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        runner = Runner(env, experiment_cfg)
        runner.agent.load(os.path.abspath(args_cli.checkpoint))
        runner.agent.set_running_mode("eval")
        mode = "sac"
    else:
        mode = "pd_only"

    zero = torch.zeros(u.num_envs, u.cfg.num_actions, device=u.device)
    obs, _ = env.reset()

    start = u.cfg.test1_start_pos
    target = u.cfg.test1_target_pos
    dt = env.step_dt

    print("\n========== Trajectory Debug (Test 1) ==========")
    print(f"  Mode:         {mode}")
    print(f"  step_dt:      {dt:.4f}s  (sim_dt={u.sim.cfg.dt:.4f}, decimation={u.cfg.decimation})")
    print(f"  Start (env):  {start}")
    print(f"  Target (env): {target}")
    print(f"  Episode max:  {u.max_episode_length} steps (~{u.max_episode_length * dt:.1f}s)")
    print("  Control: body wrench PD -> thruster allocation (B_pinv)")
    print("  Thrusters: 0,1=drive L/R | 2,3=rear L/R | 4,5=front L/R")
    print("===============================================\n")

    rows: list[dict] = []
    lx, ly, lz = _local_pos(u, u._robot.data.root_pos_w[0])
    tx, ty, tz = _local_pos(u, u._target_pos_w[0])
    print(f"[t=0.000s] spawn local=({lx:.3f},{ly:.3f},{lz:.3f}) target=({tx:.3f},{ty:.3f},{tz:.3f}) pos_err={u._position_error_norm()[0].item():.3f}m")

    for step in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            if runner is None:
                actions = zero
            else:
                out = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = out[-1].get("mean_actions", out[0])
            obs, reward, terminated, truncated, _ = env.step(actions)

        pos_w = u._robot.data.root_pos_w[0]
        vel_w = u._robot.data.root_lin_vel_w[0]
        lx, ly, lz = _local_pos(u, pos_w)
        pos_err = float(u._position_error_norm()[0].item())
        ang_err = float(
            math_utils.quat_error_magnitude(u._goal[0:1], u._robot.data.root_quat_w[0:1]).item()
        )
        t = (step + 1) * dt
        row = {
            "step": step + 1,
            "t_s": round(t, 4),
            "x": lx, "y": ly, "z": lz,
            "vx": float(vel_w[0]), "vy": float(vel_w[1]), "vz": float(vel_w[2]),
            "pos_err_m": pos_err,
            "ang_err_rad": ang_err,
            "reward": float(reward[0]),
            "u_rl_norm": float(torch.norm(u._u_rl[0])),
            "u_base_norm": float(torch.norm(u._u_base[0])),
            "u_total_norm": float(torch.norm(u._actions[0])),
        }
        for i in range(6):
            row[f"u_rl_{i}"] = float(u._u_rl[0, i])
            row[f"u_base_{i}"] = float(u._u_base[0, i])
        rows.append(row)

        if (step + 1) % args_cli.interval == 0 or step == 0:
            print(
                f"[t={t:6.2f}s] local=({lx:+.3f},{ly:+.3f},{lz:+.3f}) "
                f"err={pos_err:.3f}m vel=({vel_w[0]:+.2f},{vel_w[1]:+.2f},{vel_w[2]:+.2f}) "
                f"|u_base|={row['u_base_norm']:.2f} |u_rl|={row['u_rl_norm']:.2f}"
            )

        if bool(terminated[0] or truncated[0]):
            reason = "out_of_bounds" if bool(terminated[0]) else "timeout"
            print(f"\n[Episode end @ step {step+1}] reason={reason} final_pos_err={pos_err:.3f}m")
            break

    env.close()

    out = args_cli.output or os.path.join("logs", "residual_rl_debug", f"trajectory_{mode}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] Trajectory CSV: {os.path.abspath(out)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
    simulation_app.close()
