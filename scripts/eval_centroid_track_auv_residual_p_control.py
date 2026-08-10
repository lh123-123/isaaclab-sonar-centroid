#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sanity-check AUV compensatory residual scales (no PPO): image-P on Δsway/Δsurge."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AUV residual scale sanity — rule-based image compensation.")
parser.add_argument("--task", type=str, default="Isaac-SonarCentroidTrack-AUV-Residual-Eval-Direct-v0")
parser.add_argument("--kp_u", type=float, default=3.5, help="Gain on u_norm → Δsway action")
parser.add_argument("--kp_v", type=float, default=1.5, help="Gain on v_norm → Δsurge action")
parser.add_argument("--max_residual_sway", type=float, default=0.0, help="0=cfg default")
parser.add_argument("--max_residual_surge", type=float, default=0.0, help="0=cfg default")
parser.add_argument("--tune", action="store_true", help="Sweep kp_u with zero residual baseline")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.headless:
    args_cli.sonar_ui = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _policy_obs(obs) -> torch.Tensor:
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def _run_episode(env, *, kp_u: float, kp_v: float, verbose: bool) -> dict:
    unwrapped = env.unwrapped
    obs, _ = env.reset()
    pol = _policy_obs(obs)
    d_cols: list[float] = []
    sway_errs: list[float] = []
    step_dt = float(unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)
    total = max(int(round(float(unwrapped.cfg.lateral_target_period_s) / step_dt)), 1)

    for step in range(total):
        u = float(pol[0, 0].item())
        v = float(pol[0, 1].item())
        act_u = float(np.clip(kp_u * u, -1.0, 1.0))
        act_v = float(np.clip(-kp_v * v, -1.0, 1.0))
        action = torch.tensor([[act_u, act_v]], device=unwrapped.device, dtype=torch.float32)
        obs, _, _, _, _ = env.step(action)
        pol = _policy_obs(obs)

        frame = unwrapped.get_centroid_frame(0)
        p_fixed = unwrapped.fixed.p_fixed_abs_env(0)
        if frame.p_sonar_abs is not None and p_fixed is not None:
            d_cols.append(abs(frame.p_sonar_abs.col - p_fixed.col))
        sway = float(unwrapped._robot_sway_m()[0].item())
        tgt = float(unwrapped._target_dy_m()[0].item())
        sway_errs.append(abs(sway - tgt))

        if verbose and (step % max(total // 8, 1) == 0 or step == total - 1):
            dc = d_cols[-1] if d_cols else float("nan")
            print(f"    step={step:4d} u={u:+.3f} act=({act_u:+.2f},{act_v:+.2f}) |d_col|={dc:.2f}", flush=True)

    dc = np.array(d_cols, dtype=np.float64)
    se = np.array(sway_errs, dtype=np.float64)
    return {
        "d_col_mean": float(dc.mean()) if len(dc) else float("nan"),
        "d_col_p90": float(np.percentile(dc, 90)) if len(dc) else float("nan"),
        "d_col_max": float(dc.max()) if len(dc) else float("nan"),
        "sway_err_mean": float(se.mean()) if len(se) else float("nan"),
        "sway_err_max": float(se.max()) if len(se) else float("nan"),
    }


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.eval_mode = True
    env_cfg.spawn_radius = 0.0
    env_cfg.eval_spawn_radius = 0.0
    env_cfg.debug_disable_lost_terminate = True
    env_cfg.print_centroid_debug = False
    if float(args_cli.max_residual_sway) > 0.0:
        env_cfg.auv_max_residual_sway = float(args_cli.max_residual_sway)
    if float(args_cli.max_residual_surge) > 0.0:
        env_cfg.auv_max_residual_surge = float(args_cli.max_residual_surge)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    print(
        f"\n=== AUV residual sanity (base={getattr(env_cfg, 'auv_base_law', '?')}) ===",
        flush=True,
    )
    print(
        f"  Δsway max={unwrapped.cfg.auv_max_residual_sway:.2f} m  "
        f"Δsurge max={unwrapped.cfg.auv_max_residual_surge:.2f} m\n",
        flush=True,
    )

    if args_cli.tune:
        grid = [0.0, 1.5, 2.5, 3.5, 5.0, 7.0]
        print(f"{'kp_u':>5} |d_col| mean  p90   sway_err", flush=True)
        for k in grid:
            m = _run_episode(env, kp_u=k, kp_v=float(args_cli.kp_v), verbose=False)
            print(
                f"{k:5.1f} {m['d_col_mean']:8.2f} {m['d_col_p90']:5.2f}  {m['sway_err_mean']:.3f}",
                flush=True,
            )
    else:
        print(f"  kp_u={args_cli.kp_u}  kp_v={args_cli.kp_v}\n", flush=True)
        m0 = _run_episode(env, kp_u=0.0, kp_v=0.0, verbose=False)
        m1 = _run_episode(env, kp_u=float(args_cli.kp_u), kp_v=float(args_cli.kp_v), verbose=True)
        print("\n=== Summary ===", flush=True)
        print(
            f"  zero residual: |d_col| mean={m0['d_col_mean']:.2f}  p90={m0['d_col_p90']:.2f}  "
            f"max={m0['d_col_max']:.2f}",
            flush=True,
        )
        print(
            f"  image residual: |d_col| mean={m1['d_col_mean']:.2f}  p90={m1['d_col_p90']:.2f}  "
            f"max={m1['d_col_max']:.2f}",
            flush=True,
        )
        print(
            f"  sway_err mean {m0['sway_err_mean']:.3f} → {m1['sway_err_mean']:.3f} m\n",
            flush=True,
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
