#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless PPO residual eval: compensation magnitude + GT/beam distance tracking."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AUV residual PPO metrics (distance + compensation).")
parser.add_argument("--task", type=str, default="Isaac-SonarCentroidTrack-AUV-Residual-Eval-Direct-v0")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=(
        "logs/skrl/sonar_centroid_track/"
        "2026-07-13_17-53-29_ppo_torch_auv_residual_ppo_sine_beamrecalib_gtreward/checkpoints/best_agent.pt"
    ),
)
parser.add_argument("--baseline_pd", action="store_true", help="Also run zero-residual PD-only episode.")
parser.add_argument("--print_every", type=int, default=50)
parser.add_argument("--algorithm", type=str, default="PPO")
parser.add_argument(
    "--gt_blend",
    type=float,
    default=-1.0,
    help="Override auv_gt_lateral_blend (α). <0 = keep cfg default.",
)
parser.add_argument(
    "--max_residual_sway",
    type=float,
    default=-1.0,
    help="Override auv_max_residual_sway (m). <0 = keep cfg.",
)
parser.add_argument(
    "--max_residual_surge",
    type=float,
    default=-1.0,
    help="Override auv_max_residual_surge (m). <0 = keep cfg.",
)
parser.add_argument("--tag", type=str, default="", help="Optional label printed in summary.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
args_cli.sonar_ui = False
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from packaging import version

import skrl

if version.parse(skrl.__version__) < version.parse("1.4.3"):
    raise RuntimeError(f"Unsupported skrl {skrl.__version__}")

from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.direct.sonar_centroid_track.modules.coordinates import gt_sonar_beacon_horiz_range_m
from isaaclab_tasks.utils.hydra import hydra_task_config


def _policy_obs(obs) -> torch.Tensor:
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def _obs_row(obs, idx: int = 0) -> list[float]:
    pol = _policy_obs(obs)
    if pol.ndim == 1:
        return pol.detach().cpu().tolist()
    return pol[idx].detach().cpu().tolist()


def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def _summarize_dist(name: str, err: np.ndarray, accept_m: float) -> None:
    if len(err) == 0:
        print(f"  {name}: no samples", flush=True)
        return
    in_band = int((np.abs(err) <= accept_m).sum())
    print(
        f"  {name}: mean={err.mean():+.3f}  std={err.std():.3f}  "
        f"min={err.min():+.3f}  max={err.max():+.3f}  p90|err|={_pct(np.abs(err), 90):.3f}",
        flush=True,
    )
    print(
        f"    |err|<={accept_m}m: {in_band}/{len(err)} ({100 * in_band / len(err):.1f}%)",
        flush=True,
    )


def _run_episode(
    env,
    runner,
    *,
    label: str,
    use_ppo: bool,
    print_every: int,
) -> dict:
    raw = env.unwrapped
    obs, _ = env.reset()
    accept_m = float(getattr(raw.cfg, "range_accept_m", 0.5))
    max_dsway = float(getattr(raw.cfg, "auv_max_residual_sway", 0.25))
    max_dsurge = float(getattr(raw.cfg, "auv_max_residual_surge", 0.40))
    action_dim = 2
    ref_gt_reset = float(raw._range_ref_gt_m[0].item())
    ref_hold = float(raw._range_ref_hold_m[0].item())
    ref_ctrl = float(raw.range_ref_for_control_m()[0].item())
    max_ep = int(raw.max_episode_length)
    p_fixed = raw.fixed.p_fixed_abs_env(0)

    if use_ppo:
        runner.agent.set_running_mode("eval")

    print(f"\n=== {label} (max_len={max_ep}) ===", flush=True)
    print(
        f"  ref_gt_reset={ref_gt_reset:.3f}m  hold_meas={ref_hold:.3f}m  "
        f"ctrl_ref={ref_ctrl:.3f}m  accept=±{accept_m:.2f}m",
        flush=True,
    )
    print(
        f"  residual scales: Δsway=±{max_dsway:.2f}m  Δsurge=±{max_dsurge:.2f}m  "
        f"α={float(getattr(raw.cfg, 'auv_gt_lateral_blend', 1.0)):.2f}",
        flush=True,
    )
    print(
        f"{'step':>5} {'rng_gt':>7} {'rng_bm':>7} {'e_hold':>7} {'e_gt0':>7} "
        f"{'surge':>6} {'|dsw|':>6} {'|dsu|':>6} {'act':>11} {'d_col':>6} {'sway':>6} {'d_rng':>6}",
        flush=True,
    )

    gt_ranges: list[float] = []
    beam_ranges: list[float] = []
    err_hold: list[float] = []
    err_gt0: list[float] = []
    d_cols: list[float] = []
    sway_errs: list[float] = []
    abs_dsw: list[float] = []
    abs_dsu: list[float] = []
    act_raw: list[tuple[float, float]] = []
    sat_count = 0

    ep_len = 0
    terminated = False

    while simulation_app.is_running() and ep_len < max_ep:
        with torch.inference_mode():
            if use_ppo:
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
            else:
                actions = torch.zeros(1, action_dim, device=raw.device)

        obs, _, term, trunc, _ = env.step(actions)
        ep_len += 1

        rng_gt = gt_sonar_beacon_horiz_range_m(
            raw._beacon_pos_w,
            raw.robot.data.root_pos_w,
            raw.robot.data.root_quat_w,
            raw._sonar_mount_offset_b,
            env_id=0,
        )
        rng_bm_t, _, _, v_bm = raw.beam_peak_range_meas_m()
        beam_ok = float(v_bm[0].item()) >= 0.5
        rng_bm = float(rng_bm_t[0].item()) if beam_ok else float("nan")

        e_hold = rng_gt - ref_ctrl
        e_gt0 = rng_gt - ref_gt_reset
        gt_ranges.append(rng_gt)
        if beam_ok:
            beam_ranges.append(rng_bm)
            err_hold.append(rng_bm - ref_ctrl)
        err_gt0.append(e_gt0)

        act = raw.actions[0].detach().cpu().tolist()
        dsw_m = abs(float(act[0]) * max_dsway)
        dsu_m = abs(float(act[1]) * max_dsurge)
        act_raw.append((float(act[0]), float(act[1])))
        if abs(act[0]) > 0.95 or abs(act[1]) > 0.95:
            sat_count += 1
        abs_dsw.append(dsw_m)
        abs_dsu.append(dsu_m)

        d_col = float("nan")
        frame = raw.get_centroid_frame(0)
        if frame.p_sonar_abs is not None and p_fixed is not None:
            d_col = abs(frame.p_sonar_abs.col - p_fixed.col)
            d_cols.append(d_col)

        sway_err = abs(float((raw._target_dy_m() - raw._robot_sway_m())[0].item()))
        sway_errs.append(sway_err)

        pol = _obs_row(obs)
        d_rng = float(pol[3]) if len(pol) > 3 else float("nan")
        surge = float(raw._robot_surge_m()[0].item())

        pe = max(int(print_every), 1)
        if ep_len % pe == 0 or ep_len == 1 or bool(term[0].item()) or bool(trunc[0].item()):
            rng_bm_s = f"{rng_bm:7.3f}" if beam_ok else "    nan"
            e_hold_s = f"{rng_gt - ref_ctrl:+7.3f}"
            dc_s = f"{d_col:6.2f}" if np.isfinite(d_col) else "   nan"
            print(
                f"{ep_len:5d} {rng_gt:7.3f} {rng_bm_s} {e_hold_s} {e_gt0:+7.3f} "
                f"{surge:+6.3f} {dsw_m:6.3f} {dsu_m:6.3f} "
                f"({act[0]:+.2f},{act[1]:+.2f}) {dc_s} {sway_err:6.3f} {d_rng:+6.3f}",
                flush=True,
            )

        if bool(term[0].item()) or bool(trunc[0].item()):
            terminated = bool(term[0].item())
            break

    gt = np.array(gt_ranges, dtype=np.float64)
    e0 = np.array(err_gt0, dtype=np.float64)
    eh = np.array(err_hold, dtype=np.float64) if err_hold else np.array([])
    eb = np.array(beam_ranges, dtype=np.float64) if beam_ranges else np.array([])
    dc = np.array(d_cols, dtype=np.float64) if d_cols else np.array([])
    dsw = np.array(abs_dsw, dtype=np.float64)
    dsu = np.array(abs_dsu, dtype=np.float64)

    print(f"\n--- {label} summary ---", flush=True)
    print(f"  episode_len={ep_len}  terminated={terminated}  timeout={not terminated}", flush=True)
    print(
        f"  GT range (actual): mean={gt.mean():.3f}  std={gt.std():.3f}  "
        f"min={gt.min():.3f}  max={gt.max():.3f}  drift_from_reset={gt[-1]-gt[0]:+.3f}m",
        flush=True,
    )
    _summarize_dist("GT − hold_ref (actual vs control target)", gt - ref_ctrl, accept_m)
    _summarize_dist("GT − reset_gt (actual vs spawn GT)", e0, accept_m)
    if len(eh):
        _summarize_dist("beam − hold_ref", eh, accept_m)
    if len(eb):
        print(
            f"  beam−GT_horiz: mean={(eb - gt[: len(eb)]).mean():+.3f}  "
            f"max|err|={np.abs(eb - gt[: len(eb)]).max():.3f}m",
            flush=True,
        )
    if len(dc):
        print(
            f"  |d_col| px: mean={dc.mean():.2f}  p90={_pct(dc, 90):.2f}  max={dc.max():.2f}",
            flush=True,
        )
    se = np.array(sway_errs, dtype=np.float64)
    print(
        f"  |sway_err| m: mean={se.mean():.3f}  p90={_pct(se, 90):.3f}  max={se.max():.3f}",
        flush=True,
    )
    print(
        f"  |Δsway| m: mean={dsw.mean():.3f}  max={dsw.max():.3f}  "
        f"|Δsurge| m: mean={dsu.mean():.3f}  max={dsu.max():.3f}",
        flush=True,
    )
    print(
        f"  action saturate (|a|>0.95): {sat_count}/{ep_len} steps ({100*sat_count/max(ep_len,1):.1f}%)",
        flush=True,
    )

    return {
        "label": label,
        "ep_len": ep_len,
        "gt_mean": float(gt.mean()),
        "gt_std": float(gt.std()),
        "gt_min": float(gt.min()),
        "gt_max": float(gt.max()),
        "gt_err_hold_mean_abs": float(np.abs(gt - ref_ctrl).mean()),
        "gt_err_hold_p90_abs": _pct(np.abs(gt - ref_ctrl), 90),
        "gt_in_band_pct": 100.0 * float((np.abs(gt - ref_ctrl) <= accept_m).mean()),
        "d_col_mean": float(dc.mean()) if len(dc) else float("nan"),
        "sway_err_mean": float(se.mean()),
        "sway_err_p90": _pct(se, 90),
        "dsw_mean": float(dsw.mean()),
        "dsu_mean": float(dsu.mean()),
        "sat_pct": 100.0 * sat_count / max(ep_len, 1),
    }


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    # Match training: do not disable lost/range terminate overrides from eval script.
    if hasattr(env_cfg, "debug_disable_lost_terminate"):
        env_cfg.debug_disable_lost_terminate = False
    if float(args_cli.gt_blend) >= 0.0:
        env_cfg.auv_gt_lateral_blend = float(args_cli.gt_blend)
    if float(args_cli.max_residual_sway) > 0.0:
        env_cfg.auv_max_residual_sway = float(args_cli.max_residual_sway)
    if float(args_cli.max_residual_surge) > 0.0:
        env_cfg.auv_max_residual_surge = float(args_cli.max_residual_surge)
    agent_cfg["seed"] = 0
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(env, agent_cfg)

    ckpt = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    tag = str(args_cli.tag).strip()
    print(f"[eval] checkpoint: {ckpt}", flush=True)
    if tag:
        print(f"[eval] tag: {tag}", flush=True)
    print(
        f"[eval] α={float(env_cfg.auv_gt_lateral_blend):.2f}  "
        f"Δsway=±{float(env_cfg.auv_max_residual_sway):.2f}  "
        f"Δsurge=±{float(env_cfg.auv_max_residual_surge):.2f}",
        flush=True,
    )
    runner.agent.load(ckpt)

    results: list[dict] = []
    if args_cli.baseline_pd:
        results.append(
            _run_episode(
                env,
                runner,
                label="PD-only (zero residual)",
                use_ppo=False,
                print_every=int(args_cli.print_every),
            )
        )
    results.append(
        _run_episode(
            env,
            runner,
            label="PPO residual",
            use_ppo=True,
            print_every=int(args_cli.print_every),
        )
    )

    if len(results) == 2:
        pd, ppo = results
        print("\n=== PD vs PPO (distance = GT horiz vs hold_ref) ===", flush=True)
        print(
            f"  |GT−hold| mean: PD={pd['gt_err_hold_mean_abs']:.3f}m  "
            f"PPO={ppo['gt_err_hold_mean_abs']:.3f}m  "
            f"Δ={pd['gt_err_hold_mean_abs']-ppo['gt_err_hold_mean_abs']:+.3f}m",
            flush=True,
        )
        print(
            f"  in ±{float(getattr(env.unwrapped, 'cfg').range_accept_m):.2f}m band: "
            f"PD={pd['gt_in_band_pct']:.1f}%  PPO={ppo['gt_in_band_pct']:.1f}%",
            flush=True,
        )
        print(
            f"  |d_col| mean px: PD={pd['d_col_mean']:.2f}  PPO={ppo['d_col_mean']:.2f}",
            flush=True,
        )
        print(
            f"  |sway_err| mean: PD={pd['sway_err_mean']:.3f}m  PPO={ppo['sway_err_mean']:.3f}m",
            flush=True,
        )
        print(
            f"  |Δsurge| mean: PD={pd['dsu_mean']:.3f}m  PPO={ppo['dsu_mean']:.3f}m",
            flush=True,
        )
        print(
            f"  |Δsway| mean: PD={pd['dsw_mean']:.3f}m  PPO={ppo['dsw_mean']:.3f}m  "
            f"sat PPO={ppo['sat_pct']:.1f}%",
            flush=True,
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
