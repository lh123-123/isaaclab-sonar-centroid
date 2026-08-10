#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Debug: GT (r, α) vs OceanSim beam-peak, optionally belief-gated association.

Modes:
  (default)           env beam peak (teach prior / cfg gate)
  --moving            sine-moving ball, robot hover
  --belief_gate       EKF predicts (α,r); peak only in gate around prediction
                      (answers: how do we know which return is the target)

Usage:
  MOVING=1 BELIEF_GATE=1 ./scripts/debug_auv_sonar_vs_gt_meas.sh
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AUV sonar meas vs GT (r, α) debug.")
parser.add_argument("--task", type=str, default="Isaac-SonarCentroidTrack-AUV-Debug-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--log_every", type=int, default=5)
parser.add_argument(
    "--gate_deg",
    type=float,
    default=None,
    help="Azimuth gate half-width (deg). With --belief_gate default 20; else cfg.",
)
parser.add_argument(
    "--peak_threshold",
    type=float,
    default=None,
    help="Override sonar_beam_peak_threshold (None = cfg default).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="CSV + summary output directory.",
)
parser.add_argument(
    "--moving",
    action="store_true",
    help="Enable sine lateral target motion (robot stays hover; ball moves).",
)
parser.add_argument("--amp_m", type=float, default=1.3, help="Sine amplitude (m) when --moving.")
parser.add_argument("--vy_peak", type=float, default=0.28, help="Sine peak |vy| (m/s) when --moving.")
parser.add_argument(
    "--belief_gate",
    action="store_true",
    help="Associate detections with Nonar2 EKF predicted (α,r) gate.",
)
parser.add_argument(
    "--detector",
    type=str,
    choices=("beam", "centroid"),
    default="beam",
    help="beam=intensity argmax on beams; centroid=bright-spot on preprocessed polar (sub-bin).",
)
parser.add_argument(
    "--range_bias_m",
    type=float,
    default=None,
    help="Add to sonar range before compare/KF (surface→center). "
    "Default: beacon sphere radius when detector=centroid, else 0.",
)
parser.add_argument(
    "--no_range_bias",
    action="store_true",
    help="Disable surface→center range correction.",
)
parser.add_argument(
    "--viz_dir",
    type=str,
    default="",
    help="If set, save annotated polar overlays here (GT / pred / detect).",
)
parser.add_argument("--viz_every", type=int, default=20, help="Save overlay every N steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.belief_gate and args_cli.detector == "beam" and "--detector" not in __import__("sys").argv:
    # Belief association works better with polar centroid (default when belief_gate).
    args_cli.detector = "centroid"
if not str(args_cli.output_dir).strip():
    tag = []
    if args_cli.belief_gate:
        tag.append("belief")
    if args_cli.moving:
        tag.append("moving")
    tag.append(args_cli.detector)
    args_cli.output_dir = "logs/centroid_track/auv_sonar_vs_gt_" + "_".join(tag)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.nonar2.modules.belief_estimator import BeliefEstimator
from isaaclab_tasks.direct.sonar_centroid_track.modules.coordinates import (
    polar_col_to_azimuth_rad,
    polar_range_m_to_row,
    polar_row_to_range_m,
)
from isaaclab_tasks.direct.sonar_centroid_track.modules.sonar_beam_range import peak_beam_range_from_beams
from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import (
    beacon_bearing_range,
    euler_xyz_from_quat,
    sonar_pos_w,
)
from isaaclab_tasks.direct.sonar_flow_hover.shared_platform import TARGET_SPHERE_RADIUS
from isaaclab_tasks.utils import parse_env_cfg

import cv2
import numpy as np


def _wrap_pi(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def _polar_to_u8(polar: np.ndarray) -> np.ndarray:
    x = polar.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi > lo:
        x = (x - lo) / (hi - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def _save_meas_overlay(
    raw,
    *,
    out_path: Path,
    step: int,
    r_gt: float,
    a_gt_rad: float,
    r_det_raw: float,
    a_det_rad: float,
    a_pred_deg: float,
    range_bias: float,
    gate_deg: float,
    valid: bool,
) -> None:
    """Annotate polar: blue=GT center, cyan=GT surface, yellow=pred/gate, green=detect."""
    ocfg = raw.cfg.oceansim_sonar_cfg
    h = int(ocfg.height)
    w = int(ocfg.width)
    fov = math.radians(float(ocfg.hori_fov))
    min_r = float(ocfg.min_range)
    max_r = float(ocfg.max_range)

    gray = _polar_to_u8(raw.pipeline.polar_image[0].detach().cpu().numpy())
    scale = max(4, 512 // max(w, 1))
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    def _px(az_rad: float, rng: float) -> tuple[int, int]:
        col = _azimuth_to_col(az_rad, w, fov)
        row = polar_range_m_to_row(rng, min_range=min_r, max_range=max_r, polar_height=h)
        return int(round(col * scale)), int(round(row * scale))

    if math.isfinite(a_pred_deg):
        r_pred_surf = max(r_gt - range_bias, min_r)
        cx, cy = _px(math.radians(a_pred_deg), r_pred_surf)
        fov_deg = max(math.degrees(fov), 1e-3)
        rx = max(int(gate_deg / fov_deg * (w - 1) * scale), 4)
        span = max(max_r - min_r, 1e-3)
        ry = max(int(0.6 / span * (h - 1) * scale), 4)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (0, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.drawMarker(img, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

    cv2.circle(img, _px(a_gt_rad, r_gt), 4, (255, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, _px(a_gt_rad, max(r_gt - range_bias, min_r)), 3, (255, 255, 0), 1, lineType=cv2.LINE_AA)

    if valid:
        cv2.circle(img, _px(a_det_rad, r_det_raw), 5, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    label = (
        f"step={step} valid={int(valid)} "
        f"GT(a={math.degrees(a_gt_rad):+.1f} r={r_gt:.2f}) "
        f"det(a={math.degrees(a_det_rad):+.1f} r={r_det_raw:.2f})"
    )
    cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        "blue=GT center  cyan ring=GT surface  yellow=pred/gate  green=detect",
        (8, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[sonar_vs_gt] viz {out_path.resolve()}", flush=True)

def _gt_bearing_range(raw) -> tuple[torch.Tensor, torch.Tensor]:
    sonar_pw = sonar_pos_w(
        raw.robot.data.root_pos_w,
        raw.robot.data.root_quat_w,
        raw._sonar_mount_offset_b,
    )
    return beacon_bearing_range(raw._beacon_pos_w, sonar_pw, raw.robot.data.root_quat_w)


def _sonar_pose_xy_yaw(raw) -> tuple[torch.Tensor, torch.Tensor]:
    sonar_pw = sonar_pos_w(
        raw.robot.data.root_pos_w,
        raw.robot.data.root_quat_w,
        raw._sonar_mount_offset_b,
    )
    _, _, yaw = euler_xyz_from_quat(raw.robot.data.root_quat_w)
    return sonar_pw[:, :2], yaw


def _predict_polar(belief: BeliefEstimator, agent_xy: torch.Tensor, yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rel_w = belief.mean_xy - agent_xy
    ca, sa = torch.cos(yaw), torch.sin(yaw)
    rel_bx = ca * rel_w[:, 0] + sa * rel_w[:, 1]
    rel_by = -sa * rel_w[:, 0] + ca * rel_w[:, 1]
    az = torch.atan2(rel_by, rel_bx)
    rng = torch.sqrt(rel_bx * rel_bx + rel_by * rel_by + 1.0e-8)
    return az, rng


def _azimuth_to_col(az_rad: float, width: int, hori_fov_rad: float) -> float:
    """Body azimuth → polar image column (matches ``beacon_polar_rois``: +az → lower col)."""
    fov_deg = max(math.degrees(float(hori_fov_rad)), 1.0e-6)
    az_deg = math.degrees(float(az_rad))
    w = float(width)
    col = w / 2.0 - az_deg * w / fov_deg
    return max(0.0, min(w - 1.0, col))


def _col_to_azimuth_rad(col: float, width: int, hori_fov_rad: float) -> float:
    """Polar image column → body azimuth (inverse of ``beacon_polar_rois``)."""
    fov_deg = max(math.degrees(float(hori_fov_rad)), 1.0e-6)
    w = float(width)
    az_deg = (w / 2.0 - float(col)) * fov_deg / w
    return math.radians(az_deg)


def _beam_meas(
    raw,
    *,
    peak_threshold: float,
    prior_azimuth: torch.Tensor | None,
    prior_gate_deg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw._ensure_pipeline()
    return peak_beam_range_from_beams(
        raw.oceansim_sonar.get_beam_data(),
        peak_threshold=peak_threshold,
        prior_azimuth=prior_azimuth,
        prior_gate_deg=prior_gate_deg,
    )


def _centroid_meas(
    raw,
    *,
    prior_azimuth: torch.Tensor | None,
    prior_range: torch.Tensor | None,
    prior_gate_deg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intensity-weighted bright-spot on preprocessed polar → (r, α)."""
    raw._ensure_pipeline()
    ocfg = raw.cfg.oceansim_sonar_cfg
    h = int(ocfg.height)
    w = int(ocfg.width)
    fov = math.radians(float(ocfg.hori_fov))
    min_r = float(ocfg.min_range)
    max_r = float(ocfg.max_range)
    device = raw.device

    r_out = torch.zeros(raw.num_envs, device=device)
    a_out = torch.zeros(raw.num_envs, device=device)
    peak_out = torch.zeros(raw.num_envs, device=device)
    valid_out = torch.zeros(raw.num_envs, device=device)

    # Gate half-width in polar pixels from azimuth degrees.
    fov_deg = max(math.degrees(fov), 1.0e-3)
    search_col = max(float(prior_gate_deg) / fov_deg * (w - 1), 4.0)
    # ~0.6 m range window in pixels.
    span = max(max_r - min_r, 1.0e-3)
    search_row = max(0.6 / span * (h - 1), 6.0)

    for i in range(raw.num_envs):
        polar = raw.pipeline.polar_image[i].detach().cpu().numpy()
        center = None
        if prior_azimuth is not None and prior_range is not None:
            az = float(prior_azimuth[i].item())
            rng = float(prior_range[i].item())
            col = _azimuth_to_col(az, w, fov)
            row = polar_range_m_to_row(rng, min_range=min_r, max_range=max_r, polar_height=h)
            center = (col, row)
            pt = raw.gt.detect_abs_roi(
                polar,
                center,
                search_radius_col=search_col,
                search_radius_row=search_row,
            )
        else:
            pt = raw.gt.detect_abs(polar)
        if pt is None:
            continue
        a_out[i] = _col_to_azimuth_rad(pt.col, w, fov)
        r_out[i] = polar_row_to_range_m(pt.row, min_range=min_r, max_range=max_r, polar_height=h)
        # Peak proxy: local max in a small window for confidence logging.
        rr = int(round(pt.row))
        cc = int(round(pt.col))
        r0, r1 = max(0, rr - 2), min(h, rr + 3)
        c0, c1 = max(0, cc - 2), min(w, cc + 3)
        peak_out[i] = float(polar[r0:r1, c0:c1].max()) if polar.size else 0.0
        valid_out[i] = 1.0
    return r_out, a_out, peak_out, valid_out


def _detect(
    raw,
    *,
    detector: str,
    peak_threshold: float,
    prior_azimuth: torch.Tensor | None,
    prior_range: torch.Tensor | None,
    prior_gate_deg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if detector == "centroid":
        return _centroid_meas(
            raw,
            prior_azimuth=prior_azimuth,
            prior_range=prior_range,
            prior_gate_deg=prior_gate_deg,
        )
    return _beam_meas(
        raw,
        peak_threshold=peak_threshold,
        prior_azimuth=prior_azimuth,
        prior_gate_deg=prior_gate_deg,
    )


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.peak_threshold is not None:
        env_cfg.sonar_beam_peak_threshold = float(args_cli.peak_threshold)
    if args_cli.moving:
        amp = float(args_cli.amp_m)
        vy = float(args_cli.vy_peak)
        period_s = 2.0 * math.pi * amp / max(vy, 1.0e-6)
        env_cfg.lateral_target_moving = True
        env_cfg.lateral_target_mode = "sine"
        env_cfg.lateral_target_amp_m = amp
        env_cfg.lateral_target_vy = vy
        env_cfg.lateral_target_period_s = period_s
        env_cfg.max_lateral_velocity = vy * 1.25
        env_cfg.auv_hover_only = True

    # Belief gate owns the prior; disable env teach gate to avoid double logic.
    if args_cli.belief_gate:
        env_cfg.sonar_beam_azimuth_gate_deg = 0.0
        env_cfg.sonar_beam_prior_use_teach = False

    if args_cli.gate_deg is not None:
        gate_deg = float(args_cli.gate_deg)
    elif args_cli.belief_gate:
        gate_deg = 20.0
    else:
        gate_deg = float(getattr(env_cfg, "sonar_beam_azimuth_gate_deg", 15.0))

    if not args_cli.belief_gate and args_cli.gate_deg is not None:
        env_cfg.sonar_beam_azimuth_gate_deg = gate_deg

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "meas_compare.csv"
    step_dt = float(raw.cfg.sim.dt * raw.cfg.decimation)
    peak_thr = float(getattr(raw.cfg, "sonar_beam_peak_threshold", 0.08))
    if args_cli.no_range_bias:
        range_bias = 0.0
    elif args_cli.range_bias_m is not None:
        range_bias = float(args_cli.range_bias_m)
    elif args_cli.detector == "centroid":
        # Intensity centroid sits between near-surface and center; full radius (0.5m)
        # over-corrects (~+0.21m). Empirically ~0.29m on this beacon/sphere setup.
        range_bias = 0.29
    else:
        range_bias = 0.0

    belief = None
    belief_inited = False
    if args_cli.belief_gate:
        belief = BeliefEstimator(raw.num_envs, raw.device, process_noise=0.05)

    obs, _ = env.reset()
    print(
        f"[sonar_vs_gt] task={args_cli.task} num_envs={raw.num_envs} "
        f"moving={bool(getattr(raw.cfg, 'lateral_target_moving', False))} "
        f"belief_gate={args_cli.belief_gate} detector={args_cli.detector} gate_deg={gate_deg} "
        f"range_bias={range_bias:.3f}m step_dt={step_dt:.4f}s peak_thr={peak_thr}",
        flush=True,
    )

    zero = torch.zeros(env.action_space.shape, device=raw.device)
    rows: list[dict] = []
    n_valid = 0
    sum_abs_r = 0.0
    sum_abs_a = 0.0
    max_abs_a_gt = 0.0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "t_s",
                "dy_m",
                "r_gt",
                "alpha_gt_deg",
                "r_sonar_raw",
                "r_sonar",
                "alpha_sonar_deg",
                "alpha_pred_deg",
                "peak_I",
                "valid",
                "err_r_m",
                "err_alpha_deg",
            ],
        )
        writer.writeheader()

        for step in range(int(args_cli.steps)):
            obs, _, terminated, truncated, _ = env.step(zero)

            bearing_gt, range_gt = _gt_bearing_range(raw)
            agent_xy, yaw = _sonar_pose_xy_yaw(raw)
            a_pred = float("nan")

            if args_cli.belief_gate:
                assert belief is not None
                belief.predict(step_dt)
                if belief_inited:
                    az_pred, r_pred = _predict_polar(belief, agent_xy, yaw)
                    a_pred = math.degrees(float(az_pred[0].item()))
                    # ROI is on the acoustic surface; belief state is sphere center.
                    r_roi = torch.clamp(r_pred - range_bias, min=1.0e-3)
                    r_s, a_s, peak_i, valid = _detect(
                        raw,
                        detector=args_cli.detector,
                        peak_threshold=peak_thr,
                        prior_azimuth=az_pred,
                        prior_range=r_roi,
                        prior_gate_deg=gate_deg,
                    )
                else:
                    r_s, a_s, peak_i, valid = _detect(
                        raw,
                        detector=args_cli.detector,
                        peak_threshold=peak_thr,
                        prior_azimuth=None,
                        prior_range=None,
                        prior_gate_deg=0.0,
                    )
                # Correct surface range → center before KF update (matches GT center).
                r_s_corr = r_s + range_bias
                belief.update(
                    a_s,
                    r_s_corr,
                    peak_i,
                    valid >= 0.5,
                    agent_pos_xy=agent_xy,
                    agent_yaw=yaw,
                    conf_threshold=0.05 if args_cli.detector == "centroid" else peak_thr,
                    dt=step_dt,
                )
                if (valid >= 0.5).any():
                    belief_inited = True
            else:
                r_s, a_s, peak_i, valid = _detect(
                    raw,
                    detector=args_cli.detector,
                    peak_threshold=peak_thr,
                    prior_azimuth=None,
                    prior_range=None,
                    prior_gate_deg=float(getattr(raw.cfg, "sonar_beam_azimuth_gate_deg", 0.0)),
                )
                r_s_corr = r_s + range_bias

            r_gt = float(range_gt[0].item())
            a_gt = float(bearing_gt[0].item())
            r_sonar_raw = float(r_s[0].item())
            r_sonar = float(r_s_corr[0].item())
            a_sonar = float(a_s[0].item())
            peak = float(peak_i[0].item())
            is_valid = float(valid[0].item()) >= 0.5
            err_r = r_sonar - r_gt if is_valid else float("nan")
            err_a = math.degrees(_wrap_pi(a_sonar - a_gt)) if is_valid else float("nan")
            dy = float(raw._target_dy_m()[0].item()) if hasattr(raw, "_target_dy_m") else float("nan")
            t_s = step * step_dt
            max_abs_a_gt = max(max_abs_a_gt, abs(math.degrees(a_gt)))

            row = {
                "step": step,
                "t_s": f"{t_s:.3f}",
                "dy_m": f"{dy:.4f}",
                "r_gt": f"{r_gt:.4f}",
                "alpha_gt_deg": f"{math.degrees(a_gt):.3f}",
                "r_sonar_raw": f"{r_sonar_raw:.4f}",
                "r_sonar": f"{r_sonar:.4f}",
                "alpha_sonar_deg": f"{math.degrees(a_sonar):.3f}",
                "alpha_pred_deg": f"{a_pred:.3f}" if math.isfinite(a_pred) else "",
                "peak_I": f"{peak:.4f}",
                "valid": int(is_valid),
                "err_r_m": f"{err_r:.4f}" if is_valid else "",
                "err_alpha_deg": f"{err_a:.3f}" if is_valid else "",
            }
            writer.writerow(row)
            rows.append(row)

            if is_valid:
                n_valid += 1
                sum_abs_r += abs(err_r)
                sum_abs_a += abs(err_a)

            viz_dir = str(args_cli.viz_dir).strip()
            if viz_dir and step % max(int(args_cli.viz_every), 1) == 0:
                _save_meas_overlay(
                    raw,
                    out_path=Path(viz_dir) / f"polar_step{step:04d}.png",
                    step=step,
                    r_gt=r_gt,
                    a_gt_rad=a_gt,
                    r_det_raw=r_sonar_raw,
                    a_det_rad=a_sonar,
                    a_pred_deg=a_pred,
                    range_bias=range_bias,
                    gate_deg=gate_deg,
                    valid=is_valid,
                )

            if step % int(args_cli.log_every) == 0:
                pred_s = f" predα={a_pred:+.1f}°" if math.isfinite(a_pred) else ""
                if is_valid:
                    print(
                        f"[sonar_vs_gt] step={step:4d} t={t_s:5.2f}s dy={dy:+.3f}m "
                        f"GT(r={r_gt:.3f}m α={math.degrees(a_gt):+.2f}°) "
                        f"SONAR(raw={r_sonar_raw:.3f}→{r_sonar:.3f}m α={math.degrees(a_sonar):+.2f}°){pred_s} "
                        f"err(Δr={err_r:+.3f}m Δα={err_a:+.2f}°)",
                        flush=True,
                    )
                else:
                    print(
                        f"[sonar_vs_gt] step={step:4d} t={t_s:5.2f}s dy={dy:+.3f}m "
                        f"GT(r={r_gt:.3f}m α={math.degrees(a_gt):+.2f}°) "
                        f"SONAR miss{pred_s} (I={peak:.3f})",
                        flush=True,
                    )

            if bool(terminated.any()) or bool(truncated.any()):
                print(f"[sonar_vs_gt] episode done at step={step}", flush=True)
                obs, _ = env.reset()
                if belief is not None:
                    belief.reset(torch.arange(raw.num_envs, device=raw.device))
                    belief_inited = False

    n = max(len(rows), 1)
    mean_abs_r = sum_abs_r / max(n_valid, 1)
    mean_abs_a = sum_abs_a / max(n_valid, 1)
    hit_rate = n_valid / n
    summary = (
        f"moving={args_cli.moving} belief_gate={args_cli.belief_gate} detector={args_cli.detector} "
        f"range_bias={range_bias:.3f}m gate_deg={gate_deg} steps={n} valid={n_valid} "
        f"hit_rate={hit_rate:.2%} mean_|Δr|={mean_abs_r:.4f}m mean_|Δα|={mean_abs_a:.3f}deg "
        f"max_|α_gt|={max_abs_a_gt:.2f}deg"
    )
    (out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"[sonar_vs_gt] {summary}", flush=True)
    print(f"[sonar_vs_gt] wrote {csv_path.resolve()}", flush=True)

    ok = hit_rate >= 0.7 and (n_valid == 0 or (mean_abs_r < 1.0 and mean_abs_a < 25.0))
    if args_cli.moving:
        ok = ok and max_abs_a_gt > 5.0
    print(f"[sonar_vs_gt] PASS={ok}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
