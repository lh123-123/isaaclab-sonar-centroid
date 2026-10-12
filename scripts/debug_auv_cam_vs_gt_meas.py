#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Debug: GT (r, α) vs camera luma-centroid, optionally belief-gated — under a trained policy.

Mirrors ``debug_auv_sonar_vs_gt_meas.py``. Default checkpoint is the morning penalty-best
650k (``2026-09-21_10-36-23/model_650000_steps.zip``).

Usage:
  ./scripts/debug_auv_cam_vs_gt_meas.sh
  BELIEF_GATE=1 ./scripts/debug_auv_cam_vs_gt_meas.sh
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AUV cam meas vs GT (r, α) under policy play.")
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v4-cam")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/sb3/Isaac-AUV-Tracking-Direct-v4-cam/2026-09-21_10-36-23/model_650000_steps.zip",
)
parser.add_argument(
    "--gate_deg",
    type=float,
    default=None,
    help="Azimuth gate half-width (deg). With --belief_gate default 20; else unused.",
)
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument(
    "--belief_gate",
    action="store_true",
    help="Associate detections with BeliefEstimator predicted (α,r) gate.",
)
parser.add_argument("--viz_dir", type=str, default="", help="Optional annotated luma overlays.")
parser.add_argument("--viz_every", type=int, default=40)
parser.add_argument("--dark_thr", type=int, default=200, help="Luma threshold (pixels darker = target).")
parser.add_argument("--min_area", type=int, default=8)
parser.add_argument("--seed", type=int, default=66)
parser.add_argument(
    "--no_policy",
    action="store_true",
    help="Zero actions (sonar-style open loop). Default: load checkpoint and play.",
)
parser.add_argument(
    "--s1",
    action="store_true",
    help="S1 replace-GT path: teacher-lock + NCC appearance + alpha-only update.",
)
parser.add_argument("--ncc_tau", type=float, default=0.45, help="NCC threshold for S1 association.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if not str(args_cli.output_dir).strip():
    tag = []
    if args_cli.s1:
        tag.append("s1")
    tag.append("belief" if args_cli.belief_gate or args_cli.s1 else "global")
    args_cli.output_dir = "logs/auv_tracking/cam_vs_gt_" + "_".join(tag)
# S1 implies belief gating.
if args_cli.s1:
    args_cli.belief_gate = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from pxr import Usd, UsdGeom
from stable_baselines3 import SAC

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.direct.isaac_auv_tracking.agents.cam_encoder import MidCNNExtractor  # noqa: F401
from isaaclab_tasks.direct.isaac_auv_tracking.agents.custom_sac_policy import CustomSACPolicy  # noqa: F401
from isaaclab_tasks.direct.isaac_auv_tracking.cam_meas import (
    azimuth_to_col,
    azimuth_to_col_cam,
    col_to_azimuth_cam,
    col_to_azimuth_rad,
    detect_centroid_luma,
    detect_s1_luma,
    detect_s1_rgb,
    extract_patch,
    ncc,
    orange_affinity,
    teacher_lock_patch,
    teacher_lock_rgb,
)
from isaaclab_tasks.direct.isaac_auv_tracking.sonar_meas import body_polar_from_xy
from isaaclab_tasks.direct.nonar2.modules.belief_estimator import BeliefEstimator
from isaaclab_tasks.direct.sonar_centroid_track.auv_platform import AUV_SONAR_MOUNT_OFFSET_LOCAL
from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import euler_xyz_from_quat, sonar_pos_w
from isaaclab_tasks.utils import parse_env_cfg


def _dbg(hypothesis_id, location, message, data=None):
    # #region agent log
    try:
        import json

        payload = {
            "sessionId": "4d71f8",
            "runId": "appear",
            "hypothesisId": str(hypothesis_id),
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
            "id": f"log_{int(time.time() * 1000)}",
        }
        with Path("/home/lh/IsaacLab/.cursor/debug-4d71f8.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # #endregion


def _wrap_pi(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def _rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
    """(N,H,W,C) uint8/float RGB → (N,H,W) uint8 luma."""
    f = rgb[..., :3].to(torch.float32)
    y = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).round().clamp_(0, 255)
    return y.to(torch.uint8)


def _cam_xy_yaw(raw) -> tuple[torch.Tensor, torch.Tensor]:
    off = torch.tensor(AUV_SONAR_MOUNT_OFFSET_LOCAL, device=raw.device, dtype=torch.float32)
    cam_w = sonar_pos_w(raw._robot.data.root_pos_w, raw._robot.data.root_quat_w, off)
    _, _, yaw = euler_xyz_from_quat(raw._robot.data.root_quat_w)
    return cam_w[:, :2], yaw


def _target_length_m(raw) -> float:
    """Characteristic target size for bbox→range.

    USD local AABB Z can be ~0.3 m (thin hull); the silhouette that drives bbox height is
    closer to the longest axis (~0.7 m unscaled). Prefer max(AABB) but never below the
    known WarpAUV length floor used by the motion probe (~0.68 m * target_scale).
    """
    stage = raw.sim.stage
    prim = stage.GetPrimAtPath("/World/envs/env_0/Target")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeLocalBound(prim).ComputeAlignedRange()
    lo = np.array(rng.GetMin(), dtype=np.float64)
    hi = np.array(rng.GetMax(), dtype=np.float64)
    extents = np.maximum(hi - lo, 1.0e-6)
    scale = float(getattr(raw.cfg, "target_scale", 1.0))
    from_usd = float(extents.max() * scale)
    floor = 0.68 * scale  # motion_probe: AABB ~0.68 m long before scale
    return max(from_usd, floor, 0.15)


def _predict_polar(belief: BeliefEstimator, agent_xy: torch.Tensor, yaw: torch.Tensor):
    rel_w = belief.x[:, :2] - agent_xy
    ca, sa = torch.cos(yaw), torch.sin(yaw)
    rel_bx = ca * rel_w[:, 0] + sa * rel_w[:, 1]
    rel_by = -sa * rel_w[:, 0] + ca * rel_w[:, 1]
    az = torch.atan2(rel_by, rel_bx)
    rng = torch.sqrt(rel_bx * rel_bx + rel_by * rel_by + 1.0e-8)
    return az, rng


def _save_overlay(
    luma: np.ndarray,
    *,
    out_path: Path,
    step: int,
    r_gt: float,
    a_gt: float,
    r_det: float,
    a_det: float,
    a_pred: float,
    hfov_deg: float,
    valid: bool,
    gate_deg: float,
) -> None:
    try:
        import cv2
    except ImportError:
        return
    h, w = luma.shape
    img = cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR)
    scale = max(2, 512 // max(w, 1))
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    fov = math.radians(hfov_deg)

    def _px(az: float, _rng: float) -> tuple[int, int]:
        col = azimuth_to_col(az, w, fov)
        row = 0.5 * (h - 1)
        return int(round(col * scale)), int(round(row * scale))

    if math.isfinite(a_pred):
        cv2.drawMarker(img, _px(math.radians(a_pred), 0.0), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
        half = max(int(gate_deg / max(hfov_deg, 1e-3) * (w - 1) * scale), 4)
        cx, cy = _px(math.radians(a_pred), 0.0)
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (0, 255, 255), 1)
    cv2.circle(img, _px(a_gt, r_gt), 5, (255, 0, 0), -1)
    if valid:
        cv2.circle(img, _px(a_det, r_det), 6, (0, 255, 0), 2)
    label = (
        f"step={step} valid={int(valid)} "
        f"GT(a={math.degrees(a_gt):+.1f} r={r_gt:.2f}) "
        f"det(a={math.degrees(a_det):+.1f} r={r_det:.2f})"
    )
    cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def _px_az(az_rad: float, width: int, height: int, hfov_rad: float, scale: int) -> tuple[int, int]:
    col = azimuth_to_col_cam(az_rad, width, math.degrees(hfov_rad))
    row = 0.5 * (height - 1)
    return int(round(col * scale)), int(round(row * scale))


def _callout(
    bgr: np.ndarray,
    xy: tuple[int, int],
    *,
    text: str,
    color: tuple[int, int, int],
    kind: str,
    slot: str,
) -> None:
    """Draw a fat marker plus a labeled chip so overlapping bearings stay readable."""
    import cv2

    x, y = int(xy[0]), int(xy[1])
    h, w = bgr.shape[:2]
    if kind == "gt":
        cv2.drawMarker(bgr, (x, y), color, cv2.MARKER_DIAMOND, 28, 3)
        cv2.drawMarker(bgr, (x, y), (255, 255, 255), cv2.MARKER_DIAMOND, 16, 1)
    elif kind == "depth":
        cv2.circle(bgr, (x, y), 14, color, 3)
        cv2.circle(bgr, (x, y), 5, color, -1)
    elif kind == "prior":
        cv2.drawMarker(bgr, (x, y), color, cv2.MARKER_CROSS, 26, 3)
    else:
        cv2.drawMarker(bgr, (x, y), color, cv2.MARKER_STAR, 26, 3)

    dx, dy = {"ul": (-78, -36), "ur": (18, -36), "dl": (-78, 22), "dr": (18, 22)}[slot]
    tx, ty = x + dx, y + dy
    tx = int(np.clip(tx, 4, w - 86))
    ty = int(np.clip(ty, 18, h - 8))
    cv2.line(bgr, (x, y), (tx + 20, ty - 6), color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    pad = 4
    x0, y0 = tx - pad, ty - th - pad
    x1, y1 = tx + tw + pad, ty + pad
    cv2.rectangle(bgr, (x0, y0), (x1, y1), color, -1)
    cv2.putText(bgr, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def _save_rgb_debug(
    rgb: np.ndarray,
    *,
    out_path: Path,
    step: int,
    a_gt: float,
    a_det: float,
    a_pred: float,
    a_orange: float,
    hfov_deg: float,
    valid: bool,
    extra: str = "",
    uv_orange: tuple[float, float] | None = None,
    uv_det: tuple[float, float] | None = None,
    uv_gt: tuple[float, float] | None = None,
    uv_pred: tuple[float, float] | None = None,
) -> None:
    """Save RGB with labeled GT / depth / prior / chroma callouts on the target."""
    try:
        import cv2
    except ImportError:
        return
    img = np.asarray(rgb[..., :3])
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = img[:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    scale = max(2, 512 // max(w, 1))
    bgr = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    fov = math.radians(hfov_deg)
    v_ref = float(uv_orange[1]) if uv_orange is not None else 0.5 * (h - 1)

    def _pt(uv: tuple[float, float] | None, az: float | None) -> tuple[int, int]:
        if uv is not None:
            return int(round(uv[0] * scale)), int(round(uv[1] * scale))
        if az is None or not math.isfinite(az):
            return _px_az(0.0, w, h, fov, scale)
        col = azimuth_to_col_cam(az, w, hfov_deg)
        return int(round(col * scale)), int(round(v_ref * scale))

    _callout(bgr, _pt(uv_gt, a_gt), text="GT", color=(255, 90, 40), kind="gt", slot="ul")
    if math.isfinite(a_pred) or uv_pred is not None:
        _callout(
            bgr,
            _pt(uv_pred, math.radians(a_pred) if math.isfinite(a_pred) else None),
            text="prior",
            color=(255, 220, 0),
            kind="prior",
            slot="ur",
        )
    if valid:
        _callout(bgr, _pt(uv_det, a_det), text="depth", color=(40, 220, 40), kind="depth", slot="dl")
    if uv_orange is not None or math.isfinite(a_orange):
        _callout(
            bgr,
            _pt(uv_orange, a_orange if math.isfinite(a_orange) else None),
            text="ncc",
            color=(240, 40, 255),
            kind="ncc",
            slot="dr",
        )
    bar = f"s={step}  " + (extra if extra else "")
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(bgr, bar[:88], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)


def _probe_chroma_vs_luma(
    rgb: np.ndarray,
    luma: np.ndarray,
    template: np.ndarray | None,
    *,
    a_gt: float,
    a_pred: float,
    a_cam: float,
    valid: bool,
    hfov_deg: float,
) -> dict:
    """Compare dark-luma lock vs orange-chroma peak vs GT column (design check, not a detector)."""
    aff = orange_affinity(rgb)
    h, w = luma.shape
    v_mid = 0.5 * (h - 1)
    u_gt = float(azimuth_to_col_cam(a_gt, w, hfov_deg))
    u_det = float(azimuth_to_col_cam(a_cam, w, hfov_deg)) if valid else float("nan")
    u_pred = (
        float(azimuth_to_col_cam(math.radians(a_pred), w, hfov_deg)) if math.isfinite(a_pred) else float("nan")
    )
    v_pk, u_pk = np.unravel_index(int(np.argmax(aff)), aff.shape)
    a_or = float(col_to_azimuth_cam(float(u_pk), w, hfov_deg))
    ui_gt = int(np.clip(round(u_gt), 0, w - 1))
    vi_mid = int(np.clip(round(v_mid), 0, h - 1))
    aff_gt = float(aff[vi_mid, ui_gt])
    luma_gt = float(luma[vi_mid, ui_gt])
    aff_det = float("nan")
    luma_det = float("nan")
    if valid and math.isfinite(u_det):
        ui_d = int(np.clip(round(u_det), 0, w - 1))
        aff_det = float(aff[vi_mid, ui_d])
        luma_det = float(luma[vi_mid, ui_d])
    ncc_gt = -1.0
    if template is not None:
        src_img = rgb if getattr(template, "ndim", 2) == 3 else luma
        v_use = float(v_pk)
        patch_gt = extract_patch(src_img, u_gt, v_use)
        if patch_gt is not None and patch_gt.size == np.asarray(template).size:
            ncc_gt = float(ncc(patch_gt, template))
    d_or_gt = abs(math.degrees(_wrap_pi(a_or - a_gt)))
    d_det_gt = abs(math.degrees(_wrap_pi(a_cam - a_gt))) if valid else float("nan")
    return {
        "u_gt": round(u_gt, 1),
        "u_det": None if not math.isfinite(u_det) else round(u_det, 1),
        "u_pred": None if not math.isfinite(u_pred) else round(u_pred, 1),
        "u_orange": int(u_pk),
        "v_orange": int(v_pk),
        "a_orange_deg": round(math.degrees(a_or), 2),
        "d_orange_gt_deg": round(d_or_gt, 2),
        "d_det_gt_deg": None if not math.isfinite(d_det_gt) else round(d_det_gt, 2),
        "aff_max": round(float(aff.max()), 2),
        "aff_gt": round(aff_gt, 2),
        "aff_det": None if not math.isfinite(aff_det) else round(aff_det, 2),
        "luma_gt": round(luma_gt, 1),
        "luma_det": None if not math.isfinite(luma_det) else round(luma_det, 1),
        "ncc_at_gt_col": round(ncc_gt, 3),
        "a_orange": a_or,
    }


def main() -> None:
    use_policy = not bool(args_cli.no_policy)
    ckpt = Path(args_cli.checkpoint)
    if use_policy and not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt.resolve()}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    # Detector must not depend on paint: lock the first-photo appearance instead.
    env_cfg.target_tint = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    print(
        f"[cam_vs_gt] env ready kf_measurement={getattr(raw.cfg, 'kf_measurement', None)} "
        f"has_cam={hasattr(raw, '_tiled_camera')}",
        flush=True,
    )
    agent = None
    vec = None
    if use_policy:
        vec = Sb3VecEnvWrapper(env, fast_variant=True)
        print(f"[cam_vs_gt] Loading checkpoint: {ckpt.resolve()}", flush=True)
        agent = SAC.load(str(ckpt), vec, print_system_info=False)
        obs = vec.reset()
    else:
        print("[cam_vs_gt] --no_policy: zero actions (open-loop like sonar_vs_gt)", flush=True)
        obs, _ = env.reset()
        zero = torch.zeros(env.action_space.shape, device=raw.device)

    gate_deg = float(args_cli.gate_deg) if args_cli.gate_deg is not None else (20.0 if args_cli.belief_gate else 15.0)
    length_m = _target_length_m(raw)
    hfov = float(getattr(raw.cfg, "cam_hfov_deg", getattr(raw.cfg, "sensor_fov_deg", 90.0)))
    sensor_range = float(getattr(raw.cfg, "sensor_range", 4.0))
    step_dt = float(raw.cfg.sim.dt * raw.cfg.decimation)
    fov_half = math.radians(hfov * 0.5)

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "meas_compare.csv"

    belief = BeliefEstimator(raw.num_envs, raw.device, process_noise=0.05) if args_cli.belief_gate else None
    belief_inited = False
    templates: list[np.ndarray | None] = [None] * raw.num_envs
    teacher_locked = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)

    # obs already reset above
    print(
        f"[cam_vs_gt] task={args_cli.task} s1={args_cli.s1} belief_gate={args_cli.belief_gate} "
        f"gate_deg={gate_deg} ncc_tau={args_cli.ncc_tau} length_m={length_m:.3f} hfov={hfov} "
        f"dark_thr={args_cli.dark_thr} steps={args_cli.steps} "
        f"policy={use_policy} ckpt={ckpt.name if use_policy else 'none'}",
        flush=True,
    )
    # #region agent log
    _dbg(
        "setup",
        "debug_auv_cam_vs_gt_meas.py:main",
        "run_config",
        {
            "s1": bool(args_cli.s1),
            "belief_gate": bool(args_cli.belief_gate),
            "gate_deg": gate_deg,
            "ncc_tau": float(args_cli.ncc_tau),
            "length_m": length_m,
            "hfov": hfov,
            "checkpoint": str(ckpt),
            "steps": int(args_cli.steps),
        },
    )
    # #endregion

    rows: list[dict] = []
    n_valid = 0
    n_gt_in_fov = 0
    n_hit_when_visible = 0
    n_false_when_oob = 0
    sum_abs_r = 0.0
    sum_abs_a = 0.0
    abs_a_list: list[float] = []
    abs_r_list: list[float] = []

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "t_s",
                "r_gt",
                "alpha_gt_deg",
                "gt_in_fov",
                "r_cam",
                "alpha_cam_deg",
                "alpha_pred_deg",
                "valid",
                "err_r_m",
                "err_alpha_deg",
            ],
        )
        writer.writeheader()

        print(f"[cam_vs_gt] entering step loop (n={args_cli.steps})...", flush=True)
        # #region agent log
        _dbg("H1-H5", "debug_auv_cam_vs_gt_meas.py:loop", "entering_loop", {"steps": int(args_cli.steps)})
        # #endregion
        for step in range(int(args_cli.steps)):
            # #region agent log
            t0 = time.perf_counter()
            if step == 0:
                print("[cam_vs_gt] step0: act...", flush=True)
            # #endregion
            if use_policy:
                assert agent is not None and vec is not None
                with torch.inference_mode():
                    actions, _ = agent.predict(obs, deterministic=True)
                # #region agent log
                t_pred = time.perf_counter() - t0
                t1 = time.perf_counter()
                if step == 0:
                    print(f"[cam_vs_gt] step0: predict done {t_pred*1000:.0f}ms, env.step...", flush=True)
                # #endregion
                with torch.inference_mode():
                    obs, _rew, dones, _infos = vec.step(actions)
                done_any = bool(np.asarray(dones).any())
            else:
                t_pred = 0.0
                t1 = time.perf_counter()
                if step == 0:
                    print("[cam_vs_gt] step0: env.step...", flush=True)
                obs, _rew, term, trunc, _info = env.step(zero)
                dones = term | trunc
                done_any = bool(dones.any().item() if hasattr(dones, "any") else bool(dones))
            # #region agent log
            t_step = time.perf_counter() - t1
            t2 = time.perf_counter()
            if step == 0:
                print(f"[cam_vs_gt] step0: env.step done {t_step*1000:.0f}ms, read luma...", flush=True)
                _dbg("H2", "debug_auv_cam_vs_gt_meas.py:step0", "before_luma", {"step_ms": round(t_step * 1000, 1)})
            # #endregion

            # Always use tiled RGB for measurement + photos. Actor obs is grayscale 64x64.
            rgb = raw._tiled_camera.data.output["rgb"]
            if step == 0:
                print(f"[cam_vs_gt] step0: tiled rgb shape={tuple(rgb.shape)} dtype={rgb.dtype}", flush=True)
            if rgb.dtype != torch.uint8:
                rgb = rgb.clamp(0, 255).to(torch.uint8)
            luma = _rgb_to_luma(rgb)
            src = "tiled"
            if step == 0:
                print(f"[cam_vs_gt] step0: luma src={src} -> cpu...", flush=True)
            luma_cpu = luma.detach().cpu() if hasattr(luma, "detach") else luma
            rgb_cpu = rgb.detach().cpu() if hasattr(rgb, "detach") else rgb
            rgb0 = rgb_cpu[0].numpy() if hasattr(rgb_cpu, "numpy") else np.asarray(rgb_cpu[0])
            if step == 0:
                print(f"[cam_vs_gt] step0: luma on host shape={tuple(luma_cpu.shape)} "
                      f"mean={float(luma_cpu.float().mean()):.1f} ({(time.perf_counter()-t2)*1000:.0f}ms)",
                      flush=True)
                _dbg(
                    "D",
                    "debug_auv_cam_vs_gt_meas.py:step0",
                    "after_luma",
                    {
                        "src": src,
                        "shape": list(luma_cpu.shape),
                        "rgb_shape": list(rgb0.shape),
                        "mean": float(luma_cpu.float().mean()),
                        "ms": round((time.perf_counter() - t2) * 1000, 1),
                    },
                )

            if step == 0:
                print("[cam_vs_gt] step0: GT pose...", flush=True)
            # Use the env's own 2D pose helper (same as reward / KF GT path). Avoids an extra
            # mount-offset quat path that was stalling after the first camera read.
            agent_xy, yaw = raw._agent_pose_2d()
            a_gt_t, r_gt_t = body_polar_from_xy(raw._target_pos, agent_xy, yaw)
            r_gt = float(r_gt_t[0].item())
            a_gt = float(a_gt_t[0].item())
            gt_in_fov = bool(r_gt <= sensor_range and abs(a_gt) <= fov_half)
            if gt_in_fov:
                n_gt_in_fov += 1
            if step == 0:
                print(f"[cam_vs_gt] step0: GT r={r_gt:.3f} a={math.degrees(a_gt):+.2f} in_fov={gt_in_fov}", flush=True)
                _dbg("H6", "debug_auv_cam_vs_gt_meas.py:step0", "after_gt", {"r_gt": r_gt, "a_gt_deg": math.degrees(a_gt), "in_fov": gt_in_fov})

            a_pred = float("nan")
            ncc_val = float("nan")
            u_det_px = float("nan")
            v_det_px = float("nan")
            t_det0 = time.perf_counter()
            lum0 = luma_cpu[0].numpy() if hasattr(luma_cpu, "numpy") else np.asarray(luma_cpu[0])

            if args_cli.s1:
                assert belief is not None
                # Teacher-lock once while GT is visible (sim only).
                if (not bool(teacher_locked[0])) and gt_in_fov:
                    d0 = None
                    depth_t = raw._tiled_camera.data.output.get("distance_to_image_plane")
                    if depth_t is not None:
                        dnp = depth_t.detach().cpu().numpy()
                        d0 = dnp[0]
                        if d0.ndim == 3:
                            d0 = d0[..., 0]
                    locked = teacher_lock_rgb(
                        rgb0, alpha_gt_rad=a_gt, hori_fov_deg=hfov, depth=d0
                    )
                    if locked is not None:
                        templates[0], _u_l, _v_l = locked
                        teacher_locked[0] = True
                        # #region agent log
                        _u_gt = float(azimuth_to_col_cam(a_gt, rgb0.shape[1], hfov))
                        _dbg(
                            "H1",
                            "debug_auv_cam_vs_gt_meas.py:teacher_lock",
                            "teacher_lock_uv",
                            {
                                "step": step,
                                "assoc": "appear",
                                "u_lock": round(float(_u_l), 1),
                                "v_lock": round(float(_v_l), 1),
                                "u_gt": round(_u_gt, 1),
                                "du_px": round(float(_u_l) - _u_gt, 1),
                                "a_gt_deg": round(math.degrees(a_gt), 2),
                                "has_depth": d0 is not None,
                            },
                        )
                        # #endregion
                        # Seed belief with GT so the first prior is correct.
                        a_gt_t_dev = torch.tensor([a_gt], device=raw.device)
                        r_gt_t_dev = torch.tensor([r_gt], device=raw.device)
                        peak = torch.ones(1, device=raw.device)
                        belief.update(
                            a_gt_t_dev,
                            r_gt_t_dev,
                            peak,
                            torch.ones(1, dtype=torch.bool, device=raw.device),
                            agent_pos_xy=agent_xy,
                            agent_yaw=yaw,
                            conf_threshold=0.05,
                            dt=step_dt,
                        )
                        belief_inited = True
                        print(
                            f"[cam_vs_gt] teacher-lock @ step={step} "
                            f"GT(a={math.degrees(a_gt):+.1f}° r={r_gt:.2f})",
                            flush=True,
                        )

                belief.predict(step_dt)
                if belief_inited and bool(teacher_locked[0]):
                    az_pred, r_pred = _predict_polar(belief, agent_xy, yaw)
                    a_pred = math.degrees(float(az_pred[0].item()))
                    r_s, a_s, valid, ncc_t, u_t, v_t = detect_s1_rgb(
                        rgb_cpu,
                        templates=templates,
                        prior_azimuth=az_pred,
                        prior_range=r_pred,
                        prior_gate_deg=gate_deg,
                        hori_fov_deg=hfov,
                        ncc_tau=float(args_cli.ncc_tau),
                        alpha_only=False,
                        length_m=length_m,
                        depth_image=raw._tiled_camera.data.output.get("distance_to_image_plane"),
                    )
                    ncc_val = float(ncc_t[0].item())
                    u_det_px = float(u_t[0].item()) if float(valid[0].item()) > 0.5 else float("nan")
                    v_det_px = float(v_t[0].item()) if float(valid[0].item()) > 0.5 else float("nan")
                else:
                    r_s = torch.zeros(raw.num_envs, device=raw.device)
                    a_s = torch.zeros(raw.num_envs, device=raw.device)
                    valid = torch.zeros(raw.num_envs, device=raw.device)

                t_det = time.perf_counter() - t_det0
                t3 = time.perf_counter()
                if a_s.device != raw.device:
                    a_s = a_s.to(raw.device)
                    r_s = r_s.to(raw.device)
                    valid = valid.to(raw.device)
                peak = torch.ones(raw.num_envs, device=raw.device)
                belief.update(
                    a_s,
                    r_s,
                    peak,
                    valid >= 0.5,
                    agent_pos_xy=agent_xy,
                    agent_yaw=yaw,
                    conf_threshold=0.05,
                    dt=step_dt,
                )
                t_bel = time.perf_counter() - t3
            elif args_cli.belief_gate:
                assert belief is not None
                belief.predict(step_dt)
                if belief_inited:
                    az_pred, r_pred = _predict_polar(belief, agent_xy, yaw)
                    a_pred = math.degrees(float(az_pred[0].item()))
                    r_s, a_s, valid = detect_centroid_luma(
                        luma_cpu,
                        prior_azimuth=az_pred,
                        prior_range=r_pred,
                        prior_gate_deg=gate_deg,
                        hori_fov_deg=hfov,
                        length_m=length_m,
                        min_range=0.15,
                        max_range=max(sensor_range * 2.0, 8.0),
                        dark_thr=int(args_cli.dark_thr),
                        min_area=int(args_cli.min_area),
                    )
                else:
                    r_s, a_s, valid = detect_centroid_luma(
                        luma_cpu,
                        prior_azimuth=None,
                        prior_range=None,
                        prior_gate_deg=0.0,
                        hori_fov_deg=hfov,
                        length_m=length_m,
                        min_range=0.15,
                        max_range=max(sensor_range * 2.0, 8.0),
                        dark_thr=int(args_cli.dark_thr),
                        min_area=int(args_cli.min_area),
                    )
                t_det = time.perf_counter() - t_det0
                t3 = time.perf_counter()
                peak = torch.ones(raw.num_envs, device=raw.device)
                if a_s.device != raw.device:
                    a_s = a_s.to(raw.device)
                    r_s = r_s.to(raw.device)
                    valid = valid.to(raw.device)
                belief.update(
                    a_s,
                    r_s,
                    peak,
                    valid >= 0.5,
                    agent_pos_xy=agent_xy,
                    agent_yaw=yaw,
                    conf_threshold=0.05,
                    dt=step_dt,
                )
                t_bel = time.perf_counter() - t3
                if (valid >= 0.5).any():
                    belief_inited = True
            else:
                r_s, a_s, valid = detect_centroid_luma(
                    luma_cpu,
                    prior_azimuth=None,
                    prior_range=None,
                    prior_gate_deg=0.0,
                    hori_fov_deg=hfov,
                    length_m=length_m,
                    min_range=0.15,
                    max_range=max(sensor_range * 2.0, 8.0),
                    dark_thr=int(args_cli.dark_thr),
                    min_area=int(args_cli.min_area),
                )
                t_det = time.perf_counter() - t_det0
                t_bel = 0.0

            # #region agent log
            if step < 5 or step % max(int(args_cli.log_every), 1) == 0:
                print(
                    f"[cam_vs_gt] timing step={step} pred={t_pred*1000:.0f}ms "
                    f"env={t_step*1000:.0f}ms det={t_det*1000:.0f}ms bel={t_bel*1000:.0f}ms "
                    f"src={src} ncc={ncc_val if math.isfinite(ncc_val) else float('nan'):.2f}",
                    flush=True,
                )
            # #endregion

            r_cam = float(r_s[0].item())
            a_cam = float(a_s[0].item())
            is_valid = float(valid[0].item()) >= 0.5
            err_r = r_cam - r_gt if is_valid else float("nan")
            err_a = math.degrees(_wrap_pi(a_cam - a_gt)) if is_valid else float("nan")
            t_s = step * step_dt

            if gt_in_fov and is_valid:
                n_hit_when_visible += 1
            if (not gt_in_fov) and is_valid:
                n_false_when_oob += 1
            if is_valid:
                n_valid += 1
                sum_abs_r += abs(err_r)
                sum_abs_a += abs(err_a)
                abs_a_list.append(abs(err_a))
                abs_r_list.append(abs(err_r))

            row = {
                "step": step,
                "t_s": f"{t_s:.3f}",
                "r_gt": f"{r_gt:.4f}",
                "alpha_gt_deg": f"{math.degrees(a_gt):.3f}",
                "gt_in_fov": int(gt_in_fov),
                "r_cam": f"{r_cam:.4f}",
                "alpha_cam_deg": f"{math.degrees(a_cam):.3f}",
                "alpha_pred_deg": f"{a_pred:.3f}" if math.isfinite(a_pred) else "",
                "valid": int(is_valid),
                "err_r_m": f"{err_r:.4f}" if is_valid else "",
                "err_alpha_deg": f"{err_a:.3f}" if is_valid else "",
            }
            writer.writerow(row)
            rows.append(row)
            f.flush()

            # #region agent log
            snap_steps = {0, 1, 2, 10, 20, 40, 60, 79}
            probe = None
            if step in snap_steps or step % max(int(args_cli.log_every), 1) == 0:
                probe = _probe_chroma_vs_luma(
                    rgb0,
                    lum0,
                    templates[0],
                    a_gt=a_gt,
                    a_pred=a_pred,
                    a_cam=a_cam,
                    valid=is_valid,
                    hfov_deg=hfov,
                )
                _dbg(
                    "A",
                    "debug_auv_cam_vs_gt_meas.py:step",
                    "chroma_vs_luma",
                    {
                        "step": step,
                        "gt_in_fov": gt_in_fov,
                        "valid": is_valid,
                        "r_gt": r_gt,
                        "a_gt_deg": round(math.degrees(a_gt), 2),
                        "a_cam_deg": None if not is_valid else round(math.degrees(a_cam), 2),
                        "a_pred_deg": None if not math.isfinite(a_pred) else round(a_pred, 2),
                        "ncc_det": None if not math.isfinite(ncc_val) else round(ncc_val, 3),
                        "ncc_at_gt_col": probe["ncc_at_gt_col"],
                        "d_orange_gt_deg": probe["d_orange_gt_deg"],
                        "d_det_gt_deg": probe["d_det_gt_deg"],
                        "aff_max": probe["aff_max"],
                        "aff_gt": probe["aff_gt"],
                        "aff_det": probe["aff_det"],
                        "luma_gt": probe["luma_gt"],
                        "luma_det": probe["luma_det"],
                        "u_gt": probe["u_gt"],
                        "u_det": probe["u_det"],
                        "u_pred": probe["u_pred"],
                        "u_orange": probe["u_orange"],
                        "v_orange": probe["v_orange"],
                        "a_orange_deg": probe["a_orange_deg"],
                        "assoc": "appear",
                        "u_ncc": None if not math.isfinite(u_det_px) else round(u_det_px, 1),
                        "v_ncc": None if not math.isfinite(v_det_px) else round(v_det_px, 1),
                    },
                )
            # #endregion

            frame_dir = Path(args_cli.output_dir) / "frames"
            if step in snap_steps:
                a_or = float(probe["a_orange"]) if probe is not None else float("nan")
                if is_valid:
                    extra = (
                        f"rGT={r_gt:.2f}m rCam={r_cam:.2f}m dR={err_r:+.2f}m "
                        f"aGT={math.degrees(a_gt):+.1f} aCam={math.degrees(a_cam):+.1f}"
                    )
                else:
                    extra = f"rGT={r_gt:.2f}m CAM=miss aGT={math.degrees(a_gt):+.1f}"
                v_row = v_det_px if math.isfinite(v_det_px) else (
                    float(probe["v_orange"]) if probe is not None else None
                )
                u_gt = float(probe["u_gt"]) if probe is not None else None
                u_det = u_det_px if math.isfinite(u_det_px) else (
                    probe["u_det"] if probe is not None else None
                )
                u_pred = probe["u_pred"] if probe is not None else None
                uv_ncc = (u_det_px, v_det_px) if math.isfinite(u_det_px) and math.isfinite(v_det_px) else None
                _save_rgb_debug(
                    rgb0,
                    out_path=frame_dir / f"rgb_step{step:04d}.png",
                    step=step,
                    a_gt=a_gt,
                    a_det=a_cam,
                    a_pred=a_pred,
                    a_orange=a_or,
                    hfov_deg=hfov,
                    valid=is_valid,
                    extra=extra,
                    uv_orange=uv_ncc,
                    uv_gt=(u_gt, v_row) if u_gt is not None and v_row is not None else None,
                    uv_det=(float(u_det), v_row) if u_det is not None and v_row is not None else None,
                    uv_pred=(float(u_pred), v_row) if u_pred is not None and v_row is not None else None,
                )
                print(f"[cam_vs_gt] wrote {frame_dir / f'rgb_step{step:04d}.png'}", flush=True)

            viz_dir = str(args_cli.viz_dir).strip()
            if viz_dir and step % max(int(args_cli.viz_every), 1) == 0:
                _save_overlay(
                    lum0,
                    out_path=Path(viz_dir) / f"luma_step{step:04d}.png",
                    step=step,
                    r_gt=r_gt,
                    a_gt=a_gt,
                    r_det=r_cam,
                    a_det=a_cam,
                    a_pred=a_pred,
                    hfov_deg=hfov,
                    valid=is_valid,
                    gate_deg=gate_deg,
                )

            if step % int(args_cli.log_every) == 0:
                pred_s = f" predα={a_pred:+.1f}°" if math.isfinite(a_pred) else ""
                ncc_s = f" ncc={ncc_val:.2f}" if math.isfinite(ncc_val) else ""
                if is_valid:
                    print(
                        f"[cam_vs_gt] step={step:4d} t={t_s:5.2f}s "
                        f"GT(r={r_gt:.3f}m α={math.degrees(a_gt):+.2f}° in_fov={int(gt_in_fov)}) "
                        f"CAM(r={r_cam:.3f}m α={math.degrees(a_cam):+.2f}°){pred_s}{ncc_s} "
                        f"err(Δr={err_r:+.3f}m Δα={err_a:+.2f}°)",
                        flush=True,
                    )
                else:
                    print(
                        f"[cam_vs_gt] step={step:4d} t={t_s:5.2f}s "
                        f"GT(r={r_gt:.3f}m α={math.degrees(a_gt):+.2f}° in_fov={int(gt_in_fov)}) "
                        f"CAM miss{pred_s}{ncc_s}",
                        flush=True,
                    )

            if done_any:
                print(f"[cam_vs_gt] episode done at step={step}", flush=True)
                if belief is not None:
                    belief.reset(torch.arange(raw.num_envs, device=raw.device))
                    belief_inited = False
                templates = [None] * raw.num_envs
                teacher_locked[:] = False

    n = max(len(rows), 1)
    mean_abs_r = sum_abs_r / max(n_valid, 1)
    mean_abs_a = sum_abs_a / max(n_valid, 1)
    med_a = float(np.median(abs_a_list)) if abs_a_list else float("nan")
    med_r = float(np.median(abs_r_list)) if abs_r_list else float("nan")
    p95_a = float(np.percentile(abs_a_list, 95)) if abs_a_list else float("nan")
    p95_r = float(np.percentile(abs_r_list, 95)) if abs_r_list else float("nan")
    hit_rate = n_valid / n
    recall = n_hit_when_visible / max(n_gt_in_fov, 1)
    false_rate = n_false_when_oob / max(n - n_gt_in_fov, 1)
    fov_frac = n_gt_in_fov / n

    # Soft PASS bars aligned with sonar_vs_gt (hit≥0.7, |Δr|<1, |Δα|<25).
    ok = hit_rate >= 0.7 and (n_valid == 0 or (mean_abs_r < 1.0 and mean_abs_a < 25.0))
    summary = (
        f"belief_gate={args_cli.belief_gate} gate_deg={gate_deg} length_m={length_m:.3f} "
        f"steps={n} valid={n_valid} hit_rate={hit_rate:.2%} "
        f"gt_in_fov={fov_frac:.2%} recall={recall:.2%} false_oob={false_rate:.2%} "
        f"mean_|Δr|={mean_abs_r:.4f}m med_|Δr|={med_r:.4f}m p95_|Δr|={p95_r:.4f}m "
        f"mean_|Δα|={mean_abs_a:.3f}deg med_|Δα|={med_a:.3f}deg p95_|Δα|={p95_a:.3f}deg"
    )
    (out_dir / "summary.txt").write_text(summary + "\n" + f"PASS={ok}\n", encoding="utf-8")
    print(f"[cam_vs_gt] {summary}", flush=True)
    print(f"[cam_vs_gt] PASS={ok}", flush=True)
    print(f"[cam_vs_gt] wrote {csv_path.resolve()}", flush=True)

    # #region agent log
    _dbg(
        "H5",
        "debug_auv_cam_vs_gt_meas.py:main",
        "final_metrics",
        {
            "belief_gate": bool(args_cli.belief_gate),
            "hit_rate": hit_rate,
            "fov_frac": fov_frac,
            "recall": recall,
            "false_oob": false_rate,
            "mean_abs_r": mean_abs_r,
            "mean_abs_a": mean_abs_a,
            "med_abs_r": med_r,
            "med_abs_a": med_a,
            "p95_abs_r": p95_r,
            "p95_abs_a": p95_a,
            "pass_sonar_bar": ok,
            "assoc": "appear",
            "target_tint": None,
            "hypA_alpha_ok": bool(abs_a_list) and med_a <= 15.0,
            "hypB_range_ok": bool(abs_r_list) and med_r <= 1.0,
            "hypC_recall_ok": recall >= 0.70,
            "hypE_fov_ok": fov_frac >= 0.50,
            "H5_pass": ok,
        },
    )
    # #endregion

    if vec is not None:
        vec.close()
    else:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
