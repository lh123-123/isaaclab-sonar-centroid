# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test 1 — point reaching: compare PD-only vs SAC residual at fixed target (5, 0, 8)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ResidualRL Test 1: fixed point reaching evaluation.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (use 1).")
parser.add_argument("--episodes", type=int, default=10, help="Episodes per controller.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "SAC"],
    help="RL algorithm used for the checkpoint (default: PPO).",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="RL checkpoint. PD-only always runs first.",
)
parser.add_argument("--output", type=str, default=None, help="CSV path for per-episode rows.")
parser.add_argument("--summary", type=str, default=None, help="JSON path for aggregated Test1 summary.")
parser.add_argument(
    "--print_interval",
    type=int,
    default=0,
    help="Print step debug every N env steps (0=off). Try 30~60 for play/debug.",
)
parser.add_argument(
    "--step_trace",
    action="store_true",
    help="Save per-step CSV traces to logs/residual_rl_debug/ (one file per episode).",
)
parser.add_argument(
    "--delta",
    type=float,
    default=None,
    help="Thruster mode: max |u_rl| per PWM (default 0.15).",
)
parser.add_argument(
    "--delta-wrench",
    type=float,
    default=None,
    help="Wrench mode: max |Δw| per force/torque component (default 0.15).",
)
parser.add_argument(
    "--residual-mode",
    type=str,
    default=None,
    choices=["wrench", "thruster"],
    help="Residual space: wrench (v3) or thruster (v2 legacy checkpoints).",
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

if version.parse(skrl.__version__) < version.parse("1.4.3"):
    raise RuntimeError(f"Unsupported skrl version: {skrl.__version__}")

from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

TEST1_TASK = "Isaac-ResidualRL-Test1-Direct-v0"


def _local_pos(unwrapped, world_pos: torch.Tensor) -> tuple[float, float, float]:
    origin = unwrapped.scene.env_origins[0]
    p = (world_pos - origin).tolist()
    return float(p[0]), float(p[1]), float(p[2])


def _overshoot_x(unwrapped) -> float:
    if not getattr(unwrapped.cfg, "point_reach_mode", False):
        return 0.0
    robot_x = unwrapped._robot.data.root_pos_w[0, 0] - unwrapped.scene.env_origins[0, 0]
    target_x = float(unwrapped.cfg.test1_target_pos[0])
    return float(max(robot_x.item() - target_x, 0.0))


def _step_snapshot(unwrapped, reward: float) -> dict:
    pos_w = unwrapped._robot.data.root_pos_w[0]
    vel_w = unwrapped._robot.data.root_lin_vel_w[0]
    lx, ly, lz = _local_pos(unwrapped, pos_w)
    pos_err = float(unwrapped._position_error_norm()[0].item())
    ang_err = float(
        math_utils.quat_error_magnitude(
            unwrapped._goal[0:1], unwrapped._robot.data.root_quat_w[0:1]
        ).item()
    )
    return {
        "x": lx,
        "y": ly,
        "z": lz,
        "vx": float(vel_w[0]),
        "vy": float(vel_w[1]),
        "vz": float(vel_w[2]),
        "pos_err_m": pos_err,
        "ang_err_rad": ang_err,
        "overshoot_x_m": _overshoot_x(unwrapped),
        "reward": reward,
        "residual_norm": float(unwrapped._residual_norm(0).item()),
        "dw_rl_norm": float(torch.norm(unwrapped._dw_rl[0]).item()),
        "u_rl_norm": float(torch.norm(unwrapped._u_rl[0]).item()),
        "u_base_norm": float(torch.norm(unwrapped._u_base[0]).item()),
        "u_total_norm": float(torch.norm(unwrapped._actions[0]).item()),
        "w_base_norm": float(torch.norm(unwrapped._w_base[0]).item()),
        "residual_mode": unwrapped.cfg.residual_mode,
        **{f"dw_rl_{i}": float(unwrapped._dw_rl[0, i]) for i in range(6)},
        **{f"u_rl_{i}": float(unwrapped._u_rl[0, i]) for i in range(6)},
        **{f"u_base_{i}": float(unwrapped._u_base[0, i]) for i in range(6)},
    }


TRACE_FIELDNAMES = [
    "mode", "episode", "step", "t_s", "x", "y", "z", "vx", "vy", "vz",
    "pos_err_m", "ang_err_rad", "overshoot_x_m", "reward",
    "residual_norm", "dw_rl_norm", "u_rl_norm", "u_base_norm", "u_total_norm",
    "w_base_norm", "residual_mode",
    *(f"dw_rl_{i}" for i in range(6)),
    *(f"u_rl_{i}" for i in range(6)),
    *(f"u_base_{i}" for i in range(6)),
]


def _print_step(mode: str, episode: int, step: int, dt: float, snap: dict) -> None:
    t = step * dt
    print(
        f"[Test1][{mode}][ep{episode} t={t:6.2f}s] "
        f"pos=({snap['x']:+.3f},{snap['y']:+.3f},{snap['z']:+.3f}) "
        f"err={snap['pos_err_m']:.3f}m ang={snap['ang_err_rad']:.3f}rad "
        f"vel=({snap['vx']:+.2f},{snap['vy']:+.2f},{snap['vz']:+.2f}) "
        f"overshoot_x={snap['overshoot_x_m']:.3f}m "
        f"|u_base|={snap['u_base_norm']:.2f} |res|={snap['residual_norm']:.2f} "
        f"|u_rl|={snap['u_rl_norm']:.2f} rew={snap['reward']:+.3f}",
        flush=True,
    )


def _episode_row(
    mode: str,
    episode: int,
    unwrapped,
    ep_steps: int,
    ep_pos_sum: float,
    ep_ang_sum: float,
    ep_reward: float,
    ep_u_rl_sum: float,
    ep_success: bool,
    final_pos_err: float,
    min_pos_err: float,
    first_success_step: int,
    final_vel_norm: float,
    final_overshoot_x: float,
) -> dict:
    target = unwrapped.cfg.test1_target_pos
    start = unwrapped.cfg.test1_start_pos
    return {
        "test": "test1_point_reach",
        "mode": mode,
        "episode": episode,
        "start_x": start[0],
        "start_y": start[1],
        "start_z": start[2],
        "target_x": target[0],
        "target_y": target[1],
        "target_z": target[2],
        "steps": ep_steps,
        "mean_pos_err_m": ep_pos_sum / max(ep_steps, 1),
        "mean_ang_err_rad": ep_ang_sum / max(ep_steps, 1),
        "min_pos_err_m": min_pos_err,
        "final_pos_err_m": final_pos_err,
        "first_success_step": first_success_step,
        "final_vel_norm_mps": final_vel_norm,
        "final_overshoot_x_m": final_overshoot_x,
        "total_reward": ep_reward,
        "mean_u_rl_norm": ep_u_rl_sum / max(ep_steps, 1),
        "success": float(ep_success),
    }


def _run_episodes(env, episodes: int, runner=None, mode: str = "pd_only") -> list[dict]:
    rows: list[dict] = []
    unwrapped = env.unwrapped
    zero_action = torch.zeros(unwrapped.num_envs, unwrapped.cfg.num_actions, device=unwrapped.device)
    dt = float(env.step_dt)
    print_interval = int(args_cli.print_interval)
    save_trace = bool(args_cli.step_trace)

    if runner is not None:
        runner.agent.set_running_mode("eval")

    obs, _ = env.reset()
    ep = 0
    ep_steps = 0
    ep_pos_sum = 0.0
    ep_ang_sum = 0.0
    ep_reward = 0.0
    ep_u_rl_sum = 0.0
    ep_success = False
    ep_min_err = float("inf")
    ep_first_success = -1
    trace_rows: list[dict] = []

    while ep < episodes and simulation_app.is_running():
        with torch.inference_mode():
            if runner is None:
                actions = zero_action
            else:
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, reward, terminated, truncated, extras = env.step(actions)

        rew = float(reward[0].item())
        done = bool(terminated[0].item() or truncated[0].item())
        # DirectRLEnv auto-resets before returning obs; use pre-reset stats on episode end.
        ep_end = extras.get("episode_end") if done else None
        if ep_end is not None:
            snap = {**ep_end, "reward": rew}
        else:
            snap = _step_snapshot(unwrapped, rew)
        pos_err = snap["pos_err_m"]
        ang_err = snap["ang_err_rad"]

        ep_steps += 1
        ep_pos_sum += pos_err
        ep_ang_sum += ang_err
        ep_reward += rew
        ep_min_err = min(ep_min_err, pos_err)
        if runner is not None:
            ep_u_rl_sum += snap["residual_norm"]
        if pos_err < unwrapped.cfg.success_pos_threshold:
            if not ep_success:
                ep_first_success = ep_steps
            ep_success = True

        if save_trace:
            trace_rows.append({"mode": mode, "episode": ep, "step": ep_steps, "t_s": round(ep_steps * dt, 4), **snap})

        if print_interval > 0 and (ep_steps == 1 or ep_steps % print_interval == 0):
            _print_step(mode, ep, ep_steps, dt, snap)

        if done:
            reason = "terminated" if bool(terminated[0].item()) else "timeout"
            final_vel = (snap["vx"] ** 2 + snap["vy"] ** 2 + snap["vz"] ** 2) ** 0.5
            if ep_min_err == float("inf"):
                ep_min_err = pos_err

            print(
                f"[Test1][{mode}][ep{ep} END {reason}] steps={ep_steps} "
                f"min_err={ep_min_err:.3f}m final_err={pos_err:.3f}m "
                f"final_vel={final_vel:.2f}m/s overshoot_x={snap['overshoot_x_m']:.3f}m "
                f"success={ep_success} first_success_step={ep_first_success} "
                f"mean|u_rl|={ep_u_rl_sum / max(ep_steps, 1):.3f} total_rew={ep_reward:+.1f}",
                flush=True,
            )

            if save_trace and trace_rows:
                trace_dir = os.path.join("logs", "residual_rl_debug")
                os.makedirs(trace_dir, exist_ok=True)
                trace_path = os.path.join(trace_dir, f"test1_trace_{mode}_ep{ep}.csv")
                with open(trace_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=TRACE_FIELDNAMES, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(trace_rows)
                print(f"[Test1] step trace: {os.path.abspath(trace_path)}", flush=True)
                trace_rows = []

            rows.append(
                _episode_row(
                    mode, ep, unwrapped, ep_steps, ep_pos_sum, ep_ang_sum,
                    ep_reward, ep_u_rl_sum, ep_success, pos_err,
                    ep_min_err, ep_first_success, final_vel, snap["overshoot_x_m"],
                )
            )
            ep += 1
            ep_steps = 0
            ep_pos_sum = 0.0
            ep_ang_sum = 0.0
            ep_reward = 0.0
            ep_u_rl_sum = 0.0
            ep_success = False
            ep_min_err = float("inf")
            ep_first_success = -1
            obs, _ = env.reset()

    return rows


def _aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    return {
        "episodes": n,
        "mean_pos_err_m": sum(r["mean_pos_err_m"] for r in rows) / n,
        "final_pos_err_m": sum(r["final_pos_err_m"] for r in rows) / n,
        "mean_ang_err_rad": sum(r["mean_ang_err_rad"] for r in rows) / n,
        "success_rate": sum(r["success"] for r in rows) / n,
        "total_reward": sum(r["total_reward"] for r in rows) / n,
        "mean_u_rl_norm": sum(r["mean_u_rl_norm"] for r in rows) / n,
    }


@hydra_task_config(TEST1_TASK, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    experiment_cfg["seed"] = args_cli.seed
    if args_cli.delta is not None:
        env_cfg.delta = args_cli.delta
    if args_cli.delta_wrench is not None:
        env_cfg.delta_wrench = args_cli.delta_wrench
    if args_cli.residual_mode is not None:
        env_cfg.residual_mode = args_cli.residual_mode

    env = gym.make(TEST1_TASK, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    unwrapped = env.unwrapped

    target = unwrapped.cfg.test1_target_pos
    start = unwrapped.cfg.test1_start_pos
    print("\n========== Test 1: Point Reaching ==========")
    print(f"  Start (env):  {start}")
    print(f"  Target (env): {target}  [plan (5,0,-8) → WarpAUV z=8]")
    print(f"  Episodes:     {args_cli.episodes} per mode, seed={args_cli.seed}")
    print(f"  residual_mode: {unwrapped.cfg.residual_mode}")
    if unwrapped.cfg.residual_mode == "wrench":
        print(f"  delta_wrench:  {unwrapped.cfg.delta_wrench}  (max |Δw| per component)")
    else:
        print(f"  delta:         {unwrapped.cfg.delta}  (max |u_rl| per thruster)")
    if args_cli.print_interval > 0:
        print(f"  Debug print:  every {args_cli.print_interval} steps")
    if args_cli.step_trace:
        print("  Step trace:   logs/residual_rl_debug/test1_trace_<mode>_ep<N>.csv")
    print("============================================\n")

    all_rows: list[dict] = []

    print("[Test1] PD-only (zero residual)...")
    all_rows.extend(_run_episodes(env, args_cli.episodes, runner=None, mode="pd_only"))

    rl_agg = {}
    rl_mode = "residual_ppo" if args_cli.algorithm.upper() == "PPO" else "residual_sac"
    run_suffix = "_ppo_torch" if args_cli.algorithm.upper() == "PPO" else "_sac_torch"
    if args_cli.checkpoint:
        log_root = os.path.abspath(
            os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
        )
        resume_path = os.path.abspath(args_cli.checkpoint)
        if not os.path.isfile(resume_path):
            resume_path = get_checkpoint_path(log_root, run_dir=f".*{run_suffix}", other_dirs=["checkpoints"])
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        runner = Runner(env, experiment_cfg)
        print(f"[Test1] Loading {args_cli.algorithm}: {resume_path}")
        runner.agent.load(resume_path)
        print(f"[Test1] {args_cli.algorithm} + PD residual...")
        rl_rows = _run_episodes(env, args_cli.episodes, runner=runner, mode=rl_mode)
        all_rows.extend(rl_rows)
        rl_agg = _aggregate(rl_rows)
    else:
        print("[Test1] No --checkpoint; RL comparison skipped.")

    env.close()

    pd_agg = _aggregate([r for r in all_rows if r["mode"] == "pd_only"])

    print("\n========== Test 1 Results ==========")
    if pd_agg:
        print(
            f"PD-only:  final_pos_err={pd_agg['final_pos_err_m']:.3f}m "
            f"(mean={pd_agg['mean_pos_err_m']:.3f}m) "
            f"success={pd_agg['success_rate']:.0%}"
        )
    if rl_agg:
        print(
            f"{args_cli.algorithm}+PD: final_pos_err={rl_agg['final_pos_err_m']:.3f}m "
            f"(mean={rl_agg['mean_pos_err_m']:.3f}m) "
            f"success={rl_agg['success_rate']:.0%} "
            f"|u_rl|={rl_agg['mean_u_rl_norm']:.3f}"
        )
    if pd_agg and rl_agg:
        delta = pd_agg["final_pos_err_m"] - rl_agg["final_pos_err_m"]
        pct = delta / max(pd_agg["final_pos_err_m"], 1e-6) * 100.0
        print(f"Δ final_pos_err (PD − {args_cli.algorithm}): {delta:+.3f}m ({pct:+.1f}%)")
    print("====================================\n")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = args_cli.output or os.path.join("logs", "residual_rl_eval", f"test1_{ts}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "test": "test1_point_reach",
        "start": list(start),
        "target": list(target),
        "seed": args_cli.seed,
        "episodes": args_cli.episodes,
        "pd_only": pd_agg,
        "residual_rl": rl_agg if rl_agg else None,
        "checkpoint": args_cli.checkpoint,
        "csv": os.path.abspath(csv_path),
    }
    summary_path = args_cli.summary or os.path.join("logs", "residual_rl_eval", f"test1_{ts}.json")
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Test1] CSV:     {os.path.abspath(csv_path)}")
    print(f"[Test1] Summary: {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
