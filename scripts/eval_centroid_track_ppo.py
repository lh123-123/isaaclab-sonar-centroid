#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Eval PPO checkpoint (headless or GUI with 6D obs debug)."""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval PPO checkpoint on triangle target motion.")
parser.add_argument("--task", type=str, default="Isaac-SonarCentroidTrack-Eval-Direct-v0")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=(
        "logs/skrl/sonar_centroid_track/"
        "2026-07-06_18-09-52_ppo_torch_discrete_ppo_roundtrip_image/checkpoints/best_agent.pt"
    ),
)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--algorithm", type=str, default="PPO")
parser.add_argument("--step_sleep", type=float, default=0.0, help="Pause between steps (GUI).")
parser.add_argument(
    "--sonar_ui",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable OceanSim sonar viewport UI.",
)
parser.add_argument(
    "--obs_debug_interval",
    type=int,
    default=0,
    help="Print 6D policy obs + step delta every N steps (0=off).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
if not args_cli.sonar_ui:
    args_cli.headless = True
else:
    args_cli.sonar_ui = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import skrl
from packaging import version

if version.parse(skrl.__version__) < version.parse("1.4.3"):
    raise RuntimeError(f"Unsupported skrl {skrl.__version__}")

from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.direct.sonar_centroid_track.modules.obs_correspondence import correspondence_report
from isaaclab_tasks.utils.hydra import hydra_task_config

OBS6D_LABELS = ("u_norm", "v_norm", "valid", "d_range", "prev_dsway", "prev_dsurge")


def _policy_obs(obs) -> torch.Tensor:
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)
    env_cfg.debug_disable_lost_terminate = True
    if "AUV-Residual" in args_cli.task:
        # Full sine period for |d_col| comparison (training terminates on range drift).
        env_cfg.range_accept_m = 999.0
    if args_cli.sonar_ui and hasattr(env_cfg, "oceansim_sonar_cfg"):
        env_cfg.oceansim_sonar_cfg.sonar_initialize_viewport = True
    agent_cfg["seed"] = int(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, agent_cfg)

    ckpt = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    print(f"[eval] Loading checkpoint: {ckpt}", flush=True)
    runner.agent.load(ckpt)
    runner.agent.set_running_mode("eval")

    raw = env.unwrapped
    w = int(raw.cfg.oceansim_sonar_cfg.width)
    h = int(raw.cfg.oceansim_sonar_cfg.height)
    max_ep = int(raw.max_episode_length)

    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    ep_success: list[bool] = []
    all_abs_dcol: list[float] = []
    all_u: list[float] = []
    all_sway_err: list[float] = []

    n_ep = int(args_cli.episodes)
    print(f"\n=== PPO eval: {n_ep} episodes (max_len={max_ep}) ===\n", flush=True)

    obs, _ = env.reset()
    ep_r = 0.0
    ep_len = 0
    ep_idx = 0
    prev_obs6d: list[float] | None = None
    obs_interval = max(int(args_cli.obs_debug_interval), 0)

    while ep_idx < n_ep and simulation_app.is_running():
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])

        raw.ensure_tracking()
        pa = raw._prev_actions[0]
        prev_action = float(pa[0].item()) if pa.numel() > 1 else int(pa.item())
        rep = correspondence_report(
            raw.tracker,
            env_id=0,
            prev_action=prev_action,
            polar_width=w,
            polar_height=h,
            continuous_actions=pa.numel() > 1 or bool(getattr(raw.cfg, "continuous_actions", False)),
        )
        if rep["d_col_px"] is not None:
            all_abs_dcol.append(abs(rep["d_col_px"]))
            all_u.append(rep["u_norm"])
        if hasattr(raw, "_robot_sway_m") and hasattr(raw, "_target_dy_m"):
            sway_err = abs(float(raw._robot_sway_m()[0].item()) - float(raw._target_dy_m()[0].item()))
            all_sway_err.append(sway_err)
        else:
            sway_err = float("nan")

        obs, reward, terminated, truncated, _ = env.step(actions)
        ep_r += float(reward[0].item())
        ep_len += 1

        if obs_interval > 0 and ep_len % obs_interval == 0:
            o6 = _policy_obs(obs)[0].detach().cpu().tolist()
            d6 = [0.0] * len(o6) if prev_obs6d is None else [a - b for a, b in zip(o6, prev_obs6d)]
            prev_obs6d = o6
            env_act = raw.actions[0].detach().cpu().tolist()
            dc = rep["d_col_px"]
            dc_s = f"{dc:+.2f}" if dc is not None else "nan"
            se_s = f"{sway_err:.3f}" if all_sway_err else "nan"
            names = OBS6D_LABELS[: len(o6)]
            obs_s = " ".join(f"{n}={v:+.4f}" for n, v in zip(names, o6))
            dlt_s = " ".join(f"Δ{n}={v:+.4f}" for n, v in zip(names, d6))
            act_s = (
                f"act=({env_act[0]:+.3f},{env_act[1]:+.3f})" if len(env_act) >= 2 else f"act={env_act[0]:+.3f}"
            )
            # Station layer diagnostics (depth / attitude) — task should look like 1-DoF follow.
            z_err = float(raw.robot.data.root_pos_w[0, 2].item() - raw._station_origin_w[0, 2].item())
            from isaaclab.utils.math import euler_xyz_from_quat

            r, p, _ = euler_xyz_from_quat(raw.robot.data.root_quat_w[0:1])
            att_s = f"z_err={z_err:+.3f}m roll={float(r[0]):+.2f} pitch={float(p[0]):+.2f}"
            print(
                f"[obs6d] ep={ep_idx + 1} step={ep_len:4d}  {obs_s}  |  {dlt_s}  |  {act_s}  "
                f"d_col={dc_s}px  sway_err={se_s}m  {att_s}",
                flush=True,
            )

        if args_cli.step_sleep > 0.0:
            time.sleep(float(args_cli.step_sleep))

        if bool(terminated[0].item()) or bool(truncated[0].item()):
            lost = bool(terminated[0].item())
            success = (not lost) and ep_len >= max_ep - 1
            ep_rewards.append(ep_r)
            ep_lengths.append(ep_len)
            ep_success.append(success)
            dc_s = f"{rep['d_col_px']:+.2f}" if rep["d_col_px"] is not None else "nan"
            print(
                f"ep {ep_idx + 1:2d}: len={ep_len:3d}  R={ep_r:+7.2f}  "
                f"last_d_col={dc_s}px  lost={lost}  success={success}",
                flush=True,
            )
            ep_idx += 1
            ep_r = 0.0
            ep_len = 0
            prev_obs6d = None
            if ep_idx < n_ep:
                obs, _ = env.reset()

    env.close()

    if ep_rewards:
        sr = 100.0 * np.mean(ep_success)
        print("\n=== Summary ===", flush=True)
        print(f"  episodes       : {len(ep_rewards)}", flush=True)
        print(f"  success_rate   : {sr:.1f}%  (full triangle, no lost terminate)", flush=True)
        print(f"  reward mean    : {np.mean(ep_rewards):+.2f}  std={np.std(ep_rewards):.2f}", flush=True)
        print(f"  ep length mean : {np.mean(ep_lengths):.1f}  (max={max_ep})", flush=True)
        if all_abs_dcol:
            dc = np.array(all_abs_dcol)
            print(f"  |d_col| mean   : {dc.mean():.2f} px  p90={np.percentile(dc, 90):.2f}  max={dc.max():.2f}", flush=True)
            print(f"  |u_norm| mean  : {np.mean(np.abs(all_u)):.4f}", flush=True)
        if all_sway_err:
            se = np.array(all_sway_err)
            print(f"  |sway−target| mean: {se.mean():.3f} m  max={se.max():.3f} m", flush=True)
        print(flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
