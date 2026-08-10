#!/usr/bin/env python3
"""Headless PD-only station hold: depth / attitude / lateral (no PPO)."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PD-only AUV station diagnostics (no PPO).")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SonarCentroidTrack-AUV-Residual-Eval-Direct-v0",
)
parser.add_argument("--print_every", type=int, default=25)
parser.add_argument(
    "--gt_blend",
    type=float,
    default=None,
    help="Override auv_gt_lateral_blend (1=GT, 0=image). Default: keep env cfg.",
)
parser.add_argument("--algorithm", type=str, default="PPO")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
args_cli.sonar_ui = False
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    del agent_cfg
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    # Keep residual env (position_lead PD + station layer); actions stay zero → no PPO.
    env_cfg.auv_residual_ppo = True
    # PD stress uses Step B noise even on Eval task (eval play keeps noise off).
    env_cfg.auv_station_obs_noise = True
    env_cfg.auv_noise_depth_m = 0.01
    env_cfg.auv_noise_att_rad = 0.01
    env_cfg.auv_noise_vz = 0.01
    env_cfg.auv_noise_ang_vel = 0.01
    env_cfg.auv_noise_delay_steps = 0
    if args_cli.gt_blend is not None:
        env_cfg.auv_gt_lateral_blend = float(args_cli.gt_blend)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    raw = env.unwrapped

    obs, _ = env.reset()
    del obs
    max_ep = int(raw.max_episode_length)
    pe = max(int(args_cli.print_every), 1)
    action_dim = int(getattr(raw.cfg, "auv_residual_action_dim", 2))

    print("\n=== PD-only (zero residual) station diagnostics ===", flush=True)
    print(
        f"max_len={max_ep}  station_z={float(raw._station_origin_w[0, 2]):.3f}  "
        f"kp_depth={raw.cfg.auv_kp_depth} ki={getattr(raw.cfg, 'auv_ki_depth', 0)}  "
        f"kp_surge={raw.cfg.auv_kp_surge}  inhibit_fwd={getattr(raw.cfg, 'auv_inhibit_surge_forward', False)}  "
        f"kp_rng_fwd={getattr(raw.cfg, 'auv_kp_range_fwd', None)}  "
        f"range_ref={getattr(raw.cfg, 'sonar_range_ref_mode', None)}  "
        f"gate={getattr(raw.cfg, 'auv_station_gate', False)}  "
        f"alloc_pri={getattr(raw.cfg, 'auv_alloc_priority', False)}  "
        f"pri_att={getattr(raw.cfg, 'auv_alloc_priority_attitude', True)}",
        flush=True,
    )
    print(
        f"station_noise={getattr(raw.cfg, 'auv_station_obs_noise', False)}  "
        f"sig_z={getattr(raw.cfg, 'auv_noise_depth_m', 0)}  "
        f"sig_att={getattr(raw.cfg, 'auv_noise_att_rad', 0)}  "
        f"sig_vz={getattr(raw.cfg, 'auv_noise_vz', 0)}  "
        f"sig_w={getattr(raw.cfg, 'auv_noise_ang_vel', 0)}  "
        f"delay={getattr(raw.cfg, 'auv_noise_delay_steps', 0)}  "
        f"gt_blend={getattr(raw.cfg, 'auv_gt_lateral_blend', 1.0)}",
        flush=True,
    )
    print(
        f"{'step':>5} {'z_err':>7} {'roll':>7} {'pitch':>7} {'yaw_e':>7} "
        f"{'sway_e':>7} {'d_col':>7} {'surge':>7} {'r_err':>7}",
        flush=True,
    )

    z_errs: list[float] = []
    rolls: list[float] = []
    pitches: list[float] = []
    sway_errs: list[float] = []
    surges: list[float] = []
    range_errs: list[float] = []
    d_cols: list[float] = []

    ep_len = 0
    while simulation_app.is_running() and ep_len < max_ep:
        actions = torch.zeros(1, action_dim, device=raw.device)
        _, _, term, trunc, _ = env.step(actions)
        ep_len += 1

        z_err = float(raw.robot.data.root_pos_w[0, 2] - raw._station_origin_w[0, 2])
        r, p, y = euler_xyz_from_quat(raw.robot.data.root_quat_w[0:1])
        roll = float(r[0].item())
        pitch = float(p[0].item())
        yaw_e = float((y[0] - raw._fixed_robot_yaw_w[0]).item())
        # wrap-ish display
        if yaw_e > np.pi:
            yaw_e -= 2 * np.pi
        if yaw_e < -np.pi:
            yaw_e += 2 * np.pi

        sway_e = float(raw._robot_sway_m()[0] - raw._target_dy_m()[0])
        surge = float(raw._robot_surge_m()[0])
        r_meas, r_valid = raw._sonar_range_meas_m()
        r_ref = float(raw.range_ref_for_control_m()[0])
        r_err = float((r_meas[0] - r_ref) * r_valid[0]) if float(r_valid[0]) >= 0.5 else float("nan")

        d_col = float("nan")
        frame = raw.get_centroid_frame(0)
        p_fixed = raw.fixed.p_fixed_abs_env(0)
        if frame.p_sonar_abs is not None and p_fixed is not None:
            d_col = float(frame.p_sonar_abs.col - p_fixed.col)

        z_errs.append(z_err)
        rolls.append(roll)
        pitches.append(pitch)
        sway_errs.append(abs(sway_e))
        surges.append(surge)
        if r_err == r_err:
            range_errs.append(r_err)
        if d_col == d_col:
            d_cols.append(abs(d_col))

        if ep_len % pe == 0 or ep_len == 1:
            dc_s = f"{d_col:+7.2f}" if d_col == d_col else "    nan"
            re_s = f"{r_err:+7.3f}" if r_err == r_err else "    nan"
            print(
                f"{ep_len:5d} {z_err:+7.3f} {roll:+7.3f} {pitch:+7.3f} {yaw_e:+7.3f} "
                f"{sway_e:+7.3f} {dc_s} {surge:+7.3f} {re_s}",
                flush=True,
            )

        if bool(term[0].item()) or bool(trunc[0].item()):
            break

    def _summ(name: str, arr: list[float]) -> None:
        a = np.asarray(arr, dtype=np.float64)
        if len(a) == 0:
            print(f"  {name}: no samples", flush=True)
            return
        print(
            f"  {name}: mean={a.mean():+.4f}  std={a.std():.4f}  "
            f"min={a.min():+.4f}  max={a.max():+.4f}  |mean|={np.abs(a).mean():.4f}",
            flush=True,
        )

    print(f"\n--- PD-only summary (len={ep_len}) ---", flush=True)
    _summ("z_err (m)", z_errs)
    _summ("roll (rad)", rolls)
    _summ("pitch (rad)", pitches)
    _summ("|sway_err| (m)", sway_errs)
    _summ("surge (m)", surges)
    _summ("range_err (m)", range_errs)
    _summ("|d_col| (px)", d_cols)
    print(
        f"  |z_err|<0.10m: {100.0 * np.mean(np.abs(z_errs) < 0.10):.1f}%  "
        f"|range_err|<0.15m: {100.0 * np.mean(np.abs(range_errs) < 0.15) if range_errs else 0.0:.1f}%  "
        f"|surge|<0.30m: {100.0 * np.mean(np.abs(surges) < 0.30):.1f}%  "
        f"|roll|<0.08: {100.0 * np.mean(np.abs(rolls) < 0.08):.1f}%  "
        f"|pitch|<0.08: {100.0 * np.mean(np.abs(pitches) < 0.08):.1f}%",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
