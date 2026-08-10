#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify AUV sonar ranging: peak-beam range vs GT (reset + per-step)."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Compare peak-beam sonar range vs GT; optional image-row baseline."
)
parser.add_argument("--task", type=str, default="Isaac-SonarCentroidTrack-AUV-Residual-Eval-Direct-v0")
parser.add_argument("--steps", type=int, default=0, help="0 = one sine period")
parser.add_argument("--range_accept_m", type=float, default=0.5)
parser.add_argument("--print_every", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.sonar_centroid_track.modules.coordinates import (
    green_sonar_range_m,
    gt_sonar_beacon_horiz_range_m,
    gt_sonar_beacon_slant_range_m,
    range_err_norm,
)
from isaaclab_tasks.utils import parse_env_cfg


def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def _summarize(name: str, err: np.ndarray, accept_m: float) -> None:
    if len(err) == 0:
        print(f"  {name}: no valid samples", flush=True)
        return
    in_band = int((np.abs(err) <= accept_m).sum())
    print(
        f"  {name}: mean={err.mean():+.3f}  std={err.std():.3f}  "
        f"min={err.min():+.3f}  max={err.max():+.3f}  "
        f"p90|err|={_pct(np.abs(err), 90):.3f}",
        flush=True,
    )
    print(
        f"    |err| <= {accept_m}m: {in_band}/{len(err)} ({100 * in_band / len(err):.1f}%)",
        flush=True,
    )


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.range_accept_m = float(args_cli.range_accept_m)
    env_cfg.debug_disable_lost_terminate = True
    env_cfg.immediate_lost_terminate = False
    env_cfg.debug_disable_too_close_terminate = True
    env_cfg.print_centroid_debug = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    ocfg = unwrapped.cfg.oceansim_sonar_cfg
    min_r = float(ocfg.min_range)
    max_r = float(ocfg.max_range)
    polar_h = int(ocfg.height)
    accept_m = float(args_cli.range_accept_m)
    step_dt = float(unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)

    obs, _ = env.reset()
    ref_gt = float(unwrapped._range_ref_gt_m[0].item())
    ref_hold = float(unwrapped._range_ref_hold_m[0].item())
    ref_ctrl = float(unwrapped.range_ref_for_control_m()[0].item())
    bias = float(unwrapped._range_bias_m[0].item())
    period_s = float(unwrapped.cfg.lateral_target_period_s)
    total = int(args_cli.steps) if int(args_cli.steps) > 0 else max(int(round(period_s / step_dt)), 1)

    print("\n=== AUV sonar beam range verify ===\n", flush=True)
    print(f"  range_accept_m = {accept_m:.2f} m", flush=True)
    print(
        f"  gt_reset={ref_gt:.3f} m  hold_ref={ref_hold:.3f} m  "
        f"ctrl_ref={ref_ctrl:.3f} m  bias={bias:+.3f} m",
        flush=True,
    )
    print(f"  steps={total}  dt={step_dt:.3f}s  sine T={period_s:.1f}s", flush=True)
    print(
        f"  A+B+C: calib={getattr(unwrapped.cfg, 'sonar_range_reset_calib', True)}  "
        f"gate_deg={getattr(unwrapped.cfg, 'sonar_beam_azimuth_gate_deg', 0.0)}  "
        f"ref_mode={getattr(unwrapped.cfg, 'sonar_range_ref_mode', 'hold_meas')}  "
        f"peak_min={getattr(unwrapped.cfg, 'sonar_beam_peak_valid_min', 0.5)}  "
        f"ema={getattr(unwrapped.cfg, 'sonar_range_ema_alpha', 0.85)}",
        flush=True,
    )
    prior_az = float(unwrapped._beam_prior_az_rad_reset[0].item())
    print(f"  teach_prior_az={np.degrees(prior_az):+.1f} deg\n", flush=True)
    print(
        f"{'step':>5} {'rng_bm':>7} {'rng_gt':>7} {'rng_sl':>7} {'e_ctrl':>7} {'e_gt':>7} "
        f"{'surge':>7} {'sway':>7} {'az°':>6} {'peak':>5} {'obs3':>7} {'v':>2}",
        flush=True,
    )

    zero_act = torch.zeros(1, 2, device=unwrapped.device)
    err_beam_ref: list[float] = []
    err_beam_now: list[float] = []
    err_beam_slant: list[float] = []
    err_img_ref: list[float] = []
    err_img_now: list[float] = []
    valid_steps = 0
    mismatches = 0

    for step in range(total):
        obs_t, _, terminated, truncated, _ = env.step(zero_act)
        o = obs_t["policy"][0].detach().cpu().tolist()

        rng_bm, az, peak_i, v_bm = unwrapped.beam_peak_range_meas_m()
        rng_beam = float(rng_bm[0].item())
        beam_valid = float(v_bm[0].item()) >= 0.5
        rng_raw, _, peak_raw, v_raw = unwrapped._calibrated_range_raw()
        rng_raw_f = float(rng_raw[0].item())

        rng_img = float("nan")
        p_abs = unwrapped.tracker.detected_abs(0)
        if p_abs is not None:
            rng_img = green_sonar_range_m(
                p_abs.row, min_range=min_r, max_range=max_r, polar_height=polar_h
            )

        rng_gt_now = gt_sonar_beacon_horiz_range_m(
            unwrapped._beacon_pos_w,
            unwrapped.robot.data.root_pos_w,
            unwrapped.robot.data.root_quat_w,
            unwrapped._sonar_mount_offset_b,
            env_id=0,
        )
        rng_gt_slant = gt_sonar_beacon_slant_range_m(
            unwrapped._beacon_pos_w,
            unwrapped.robot.data.root_pos_w,
            unwrapped.robot.data.root_quat_w,
            unwrapped._sonar_mount_offset_b,
            env_id=0,
        )

        e_bm_r = rng_beam - ref_ctrl if beam_valid else float("nan")
        e_bm_n = rng_beam - rng_gt_now if beam_valid else float("nan")
        e_bm_s = rng_beam - rng_gt_slant if beam_valid else float("nan")
        e_im_r = rng_img - ref_gt if np.isfinite(rng_img) else float("nan")
        d_ref = range_err_norm(rng_beam, ref_ctrl, accept_m=accept_m) if beam_valid else float("nan")
        obs3 = float(o[3])
        ok = abs(obs3 - d_ref) < 1.0e-4 if np.isfinite(d_ref) else False
        if np.isfinite(d_ref) and not ok:
            mismatches += 1

        if beam_valid:
            valid_steps += 1
        if beam_valid and np.isfinite(e_bm_r):
            err_beam_ref.append(e_bm_r)
            err_beam_now.append(e_bm_n)
            err_beam_slant.append(e_bm_s)
        if np.isfinite(e_im_r):
            err_img_ref.append(e_im_r)
            err_img_now.append(rng_img - rng_gt_now)

        pe = int(args_cli.print_every)
        if step % pe == 0 or step == total - 1 or terminated or truncated:
            surge = float(unwrapped._robot_surge_m()[0].item())
            sway = float(unwrapped._robot_sway_m()[0].item())
            az_deg = float(np.degrees(float(az[0].item())))
            peak = float(peak_i[0].item())
            print(
                f"{step:5d} {rng_beam:7.3f} {rng_gt_now:7.3f} {rng_gt_slant:7.3f} "
                f"{e_bm_r:+7.3f} {e_bm_n:+7.3f} {surge:+7.3f} {sway:+7.3f} "
                f"{az_deg:6.1f} {peak:5.3f} {obs3:+7.3f} {int(beam_valid):2d}",
                flush=True,
            )
            if np.isfinite(rng_img):
                print(
                    f"       raw={rng_raw_f:.3f} ema={rng_beam:.3f} peak={peak:5.3f} "
                    f"img={rng_img:.3f}m  e_img_vs_gt={rng_img - rng_gt_now:+.3f}m",
                    flush=True,
                )
            if beam_valid and abs(e_bm_r) > accept_m:
                side = "CLOSER" if e_bm_r < 0 else "FARTHER"
                print(f"       |e_ctrl|={abs(e_bm_r):.3f}m > {accept_m}m ({side} vs hold)", flush=True)

        if terminated or truncated:
            print(f"\n  episode ended at step={step} term={bool(terminated.item())}", flush=True)
            break

    err_bm_r = np.array(err_beam_ref, dtype=np.float64)
    err_bm_n = np.array(err_beam_now, dtype=np.float64)
    err_bm_s = np.array(err_beam_slant, dtype=np.float64)
    err_im_r = np.array(err_img_ref, dtype=np.float64)
    err_im_n = np.array(err_img_now, dtype=np.float64)

    print("\n=== Summary (beam = gated+calib; img = green row baseline) ===", flush=True)
    _summarize("beam − ctrl_ref (hold)", err_bm_r, accept_m)
    _summarize("beam − gt_horiz_now", err_bm_n, accept_m)
    _summarize("beam − gt_slant_now", err_bm_s, accept_m)
    _summarize("image − gt_reset", err_im_r, accept_m)
    _summarize("image − gt_now", err_im_n, accept_m)
    if len(err_bm_r) and len(err_im_r):
        delta = err_bm_r - err_im_r[: len(err_bm_r)]
        print(
            f"  beam vs image (vs reset): mean delta={delta.mean():+.3f}m  "
            f"std={delta.std():.3f}m",
            flush=True,
        )
    print(
        f"  valid beam steps (peak>={getattr(unwrapped.cfg, 'sonar_beam_peak_valid_min', 0.5)}): "
        f"{valid_steps}/{total} ({100 * valid_steps / max(total, 1):.1f}%)",
        flush=True,
    )
    if len(err_bm_r):
        print(
            f"  |e_ctrl| mean={np.abs(err_bm_r).mean():.3f}  max={np.abs(err_bm_r).max():.3f}",
            flush=True,
        )
    print(f"  obs[3] mismatches vs (beam−hold) d_norm: {mismatches}", flush=True)
    print(f"  range_gt_reset fixed at {ref_gt:.3f} m\n", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
