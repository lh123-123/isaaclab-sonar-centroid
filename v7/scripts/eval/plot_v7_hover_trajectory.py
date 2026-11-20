#!/usr/bin/env python3
"""Plot the in-episode hover trajectory, not just the endpoint.

Why this exists
---------------
`disp_spawn_m` (start -> end) is the current bar, and it cannot tell these apart:

  * a tight hover that never left 2 cm,
  * a 40 cm excursion that happened to walk back to 3 cm by the last step,
  * a slow one-way drift of 30 cm that ended wherever the clock ran out.

All three report the same number today. This script records the position every control step
for one full episode (224 steps x 66.7 ms = 14.9 s) and reports the metrics that separate
them: RMS distance, peak excursion, total path length, and the drift rate over the last
third of the episode (a hover that holds station has ~0 late drift even if it sat off-centre).

It also records the visodom estimate next to ground truth, so "the policy drifted" and "the
policy thought it was holding still" are visually distinguishable.

Outputs
-------
  <out>/hover_traj_<tag>.png   panels: XY path | distance vs time | depth vs time
  <out>/hover_traj_<tag>.npz   raw per-step arrays (gt xy, estimate, depth, t)
  stdout: per-env metric table

Usage (headless, server):
  python v7/scripts/eval/plot_v7_hover_trajectory.py \
      --checkpoint <model.zip> --num_envs 4 --action_delay 3 --out v7/logs --tag G150k
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record and plot one 15 s hover episode.")
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v7-flow-hover")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4, help="one episode per env is recorded")
parser.add_argument("--action_delay", type=int, default=3, help="control steps of actuator dead time")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--algorithm", type=str, default="SAC", choices=["PPO", "SAC"])
parser.add_argument("--out", type=str, default="/tmp")
parser.add_argument("--tag", type=str, default="run")
AppLauncher.add_app_launcher_args(parser)
# parse_known_args + rewriting sys.argv is what keeps hydra_task_config from re-parsing our
# own flags: Hydra initialises later and would reject --checkpoint/--tag as unknown.
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg, agent_cfg):
    n = int(args_cli.num_envs)
    env_cfg.scene.num_envs = n
    env_cfg.seed = int(args_cli.seed)
    # The checkpoint must not carry its own delay on top of --action_delay, otherwise two
    # independent lags stack and the curve is unreadable.
    if hasattr(env_cfg, "train_action_delay_max_steps"):
        env_cfg.train_action_delay_max_steps = 0
    if hasattr(env_cfg, "train_action_delay_fixed_steps"):
        env_cfg.train_action_delay_fixed_steps = -1

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    core = gym_env.unwrapped
    while hasattr(core, "unwrapped") and core.unwrapped is not core:
        core = core.unwrapped

    steps = int(core.max_episode_length)
    dt = float(core.cfg.sim.dt) * float(core.cfg.decimation)
    print(f"[traj] episode = {steps} steps x {dt * 1e3:.1f} ms = {steps * dt:.2f} s", flush=True)

    agent_cfg = process_sb3_cfg(agent_cfg, n)
    for _k in ("normalize_input", "normalize_value", "clip_obs"):
        agent_cfg.pop(_k, None)
    env = Sb3VecEnvWrapper(gym_env, fast_variant=True)

    print(f"[traj] loading {args_cli.checkpoint}", flush=True)
    agent = {"PPO": PPO, "SAC": SAC}[args_cli.algorithm.upper()].load(
        args_cli.checkpoint, env, print_system_info=False, buffer_size=256
    )

    gt = np.full((n, steps, 2), np.nan)
    est = np.full((n, steps), np.nan)
    dep = np.full((n, steps), np.nan)
    done = np.zeros(n, dtype=bool)

    obs = env.reset()
    delay = max(0, int(args_cli.action_delay))
    pending: deque = deque(maxlen=delay + 1)
    step = 0
    while step < steps and simulation_app.is_running():
        with torch.inference_mode():
            action, _ = agent.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            if len(pending) == 0:
                pending.extend([np.zeros_like(action)] * delay)
            pending.append(action)
            obs, _rew, _dones, _infos = env.step(pending[0])

        pos = (core._robot.data.root_pos_w[:, :2] - core._spawn_xy).detach().cpu().numpy()
        e = core._visodom_m().detach().cpu().numpy()
        z = (core._robot.data.root_pos_w[:, 2] - float(core.cfg.starting_depth)).detach().cpu().numpy()
        ep_len = core.episode_length_buf.detach().cpu().numpy()

        for j in range(n):
            if done[j]:
                continue
            idx = int(ep_len[j]) - 1
            if idx < 0:
                done[j] = True  # reset happened: the episode already ended
                continue
            if idx < steps:
                gt[j, idx] = pos[j]
                est[j, idx] = float(np.linalg.norm(e[j]))
                dep[j, idx] = z[j]
            if idx + 1 >= steps:
                done[j] = True
        step += 1
        if done.all():
            break

    env.close()

    t = np.arange(steps) * dt
    os.makedirs(args_cli.out, exist_ok=True)
    npz = os.path.join(args_cli.out, f"hover_traj_{args_cli.tag}.npz")
    np.savez(npz, t=t, gt=gt, est=est, depth=dep)

    print(f"\n[traj] ==== metrics per episode ({args_cli.tag}) ====", flush=True)
    header = (
        f"{'env':>4} {'end':>8} {'rms':>8} {'peak':>8} {'path':>9} "
        f"{'drift_late':>11} {'est_end':>8} {'est_err':>8} {'cur':>6} {'turb':>5}"
    )
    print(header + "   (mm, mm/s)", flush=True)
    rows = []
    for j in range(n):
        d = gt[j]
        ok = ~np.isnan(d[:, 0])
        if ok.sum() < 10:
            continue
        d = d[ok]
        dist = np.linalg.norm(d, axis=1)
        k = max(1, len(d) // 3)
        drift_late = float(np.mean(np.linalg.norm(np.diff(d[-k:], axis=0), axis=1))) / dt * 1e3
        path = float(np.sum(np.linalg.norm(np.diff(d, axis=0), axis=1))) * 1e3
        e_ok = est[j][ok]
        est_end = float(np.nan_to_num(e_ok[-1])) * 1e3
        rows.append(
            dict(
                env=j,
                end=float(dist[-1]) * 1e3,
                rms=float(np.sqrt(np.mean(dist**2))) * 1e3,
                peak=float(np.max(dist)) * 1e3,
                path=path,
                drift_late=drift_late,
                est_end=est_end,
                est_err=abs(est_end - float(dist[-1]) * 1e3),
                cur=float(core._dr_cur_speed[j]),
                turb=float(core._turbidity[j]),
            )
        )
        r = rows[-1]
        print(
            f"{j:>4} {r['end']:>8.1f} {r['rms']:>8.1f} {r['peak']:>8.1f} {r['path']:>9.1f} "
            f"{r['drift_late']:>11.2f} {r['est_end']:>8.1f} {r['est_err']:>8.1f} "
            f"{r['cur']:>6.3f} {r['turb']:>5.2f}",
            flush=True,
        )
    if rows:
        print(
            f"{'mean':>4} {np.mean([r['end'] for r in rows]):>8.1f} "
            f"{np.mean([r['rms'] for r in rows]):>8.1f} "
            f"{np.mean([r['peak'] for r in rows]):>8.1f} "
            f"{np.mean([r['path'] for r in rows]):>9.1f} "
            f"{np.mean([r['drift_late'] for r in rows]):>11.2f}",
            flush=True,
        )

    # ---------------- plot ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))
    ax = axes[0]
    for j in range(n):
        d = gt[j]
        ok = ~np.isnan(d[:, 0])
        if ok.sum() < 2:
            continue
        d = d[ok]
        ax.plot(d[:, 0] * 1e3, d[:, 1] * 1e3, lw=1.6, color=colors[j], label=f"env{j} (end {np.linalg.norm(d[-1]) * 1e3:.0f} mm)")
        ax.plot(d[0, 0] * 1e3, d[0, 1] * 1e3, "o", ms=5, color=colors[j], mfc="white")
        ax.plot(d[-1, 0] * 1e3, d[-1, 1] * 1e3, "s", ms=5, color=colors[j])
    for rr in (20, 50, 100):
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, ls="--", lw=0.8, color="0.6"))
        ax.annotate(f"{rr} mm", (rr * 0.71, rr * 0.71), fontsize=7, color="0.45")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x from spawn (mm)")
    ax.set_ylabel("y from spawn (mm)")
    ax.set_title("XY path (o = spawn, □ = end)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    for j in range(n):
        d = gt[j]
        ok = ~np.isnan(d[:, 0])
        if ok.sum() < 2:
            continue
        ax.plot(t[ok], np.linalg.norm(d[ok], axis=1) * 1e3, lw=1.5, color=colors[j], label=f"gt env{j}")
        e = est[j][ok]
        ax.plot(t[ok], np.nan_to_num(e) * 1e3, lw=1.0, ls=":", color=colors[j], alpha=0.85)
    ax.axhline(50, color="crimson", ls="--", lw=1.0)
    ax.annotate("5 cm bar", (t[max(len(t) // 20, 1)], 52), color="crimson", fontsize=8)
    ax.set_xlabel("time in episode (s)")
    ax.set_ylabel("|displacement from spawn| (mm)")
    ax.set_title("distance vs time — solid: truth, dotted: visodom estimate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[2]
    for j in range(n):
        d = dep[j]
        ok = ~np.isnan(d)
        if ok.sum() < 2:
            continue
        ax.plot(t[ok], d[ok] * 1e3, lw=1.4, color=colors[j], label=f"env{j}")
    ax.set_xlabel("time in episode (s)")
    ax.set_ylabel("depth offset from start (mm)")
    ax.set_title("heave hold (depth PI target is the spawn depth)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    fig.suptitle(
        f"hover trajectory, 15 s episode — {args_cli.tag}  ({os.path.basename(args_cli.checkpoint)})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(args_cli.out, f"hover_traj_{args_cli.tag}.png")
    fig.savefig(png, dpi=140)
    print(f"\n[traj] wrote {png}\n[traj] wrote {npz}", flush=True)


if __name__ == "__main__":
    main()
    app_launcher.app.close()
