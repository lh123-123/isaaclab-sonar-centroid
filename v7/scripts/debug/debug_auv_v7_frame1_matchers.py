#!/usr/bin/env python3
"""Local probe: other ways to compare frame 1 with the current frame.

Farneback(frame_1, now) locked onto a nearby texture peak. This run keeps the
server untouched and, on one local env, drives sway then surge while measuring
the same pair of images three other ways:

  phase   cv2.phaseCorrelate on the whole frame (global translation peak)
  phaseZ  the same, after warping the current frame back to frame-1 scale
          with the ground-truth range to the plane
  ecc     findTransformECC, affine (translation + scale + rotation)
  orb     ORB matches + partial-affine RANSAC

Template-match u from the env is printed on the same rows. Nothing here is
written into the reward or the observation.

    ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_frame1_matchers.py --headless
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 frame-1 matcher probe.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import cv2
import numpy as np
import torch

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)


def _corr_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan"), float("nan")
    den = float(np.dot(x, x))
    slope = float(np.dot(x, y) / den) if den > 1e-9 else float("nan")
    if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
        return slope, float("nan")
    return slope, float(np.corrcoef(x, y)[0, 1])


def _read(env):
    origin = env.scene.env_origins[0]
    pos = (env._robot.data.root_pos_w[0] - origin).detach()
    _roll, pitch, yaw = euler_xyz_from_quat(env._robot.data.root_quat_w)
    return pos, float(yaw[0]), float(pitch[0])


def _body(pos, pos0, yaw):
    dx = float(pos[0] - pos0[0])
    dy = float(pos[1] - pos0[1])
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _act(env, surge: float, sway: float) -> torch.Tensor:
    scale = float(env.cfg.cmd_vel_scale)
    raw = [surge / scale, sway / scale, 0.0]
    return torch.tensor([raw], dtype=torch.float32, device=env.device).clamp(-1.0, 1.0)


def _gray(env, fh: int, fw: int) -> np.ndarray:
    rgb = env._camera_rgb()[0].detach().cpu().numpy()
    small = cv2.resize(rgb, (fw, fh), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def _phase(a: np.ndarray, b: np.ndarray, window: np.ndarray) -> tuple[float, float, float]:
    shift, response = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), window)
    return float(shift[0]), float(shift[1]), float(response)


def _unscale(gray: np.ndarray, scale: float) -> np.ndarray:
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 0.0, float(scale))
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR)


def _ecc(first: np.ndarray, curr: np.ndarray) -> tuple[float, float, float]:
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
    try:
        _cc, warp = cv2.findTransformECC(
            first.astype(np.float32) / 255.0,
            curr.astype(np.float32) / 255.0,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error:
        return float("nan"), float("nan"), float("nan")
    scale = math.hypot(float(warp[0, 0]), float(warp[1, 0]))
    return float(warp[0, 2]), float(warp[1, 2]), scale


def _orb(first: np.ndarray, curr: np.ndarray, orb, matcher) -> tuple[float, float, float, int]:
    k1, d1 = orb.detectAndCompute(first, None)
    k2, d2 = orb.detectAndCompute(curr, None)
    if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
        return float("nan"), float("nan"), float("nan"), 0
    pairs = matcher.match(d1, d2)
    if len(pairs) < 8:
        return float("nan"), float("nan"), float("nan"), len(pairs)
    pairs = sorted(pairs, key=lambda m: m.distance)[:80]
    p1 = np.float32([k1[m.queryIdx].pt for m in pairs])
    p2 = np.float32([k2[m.trainIdx].pt for m in pairs])
    mat, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if mat is None:
        return float("nan"), float("nan"), float("nan"), 0
    n_in = int(inliers.sum()) if inliers is not None else 0
    scale = math.hypot(float(mat[0, 0]), float(mat[1, 0]))
    return float(mat[0, 2]), float(mat[1, 2]), scale, n_in


def main() -> None:
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    env.reset()
    fh = int(cfg.flow_render_height)
    fw = int(cfg.flow_render_width)
    window = cv2.createHanningWindow((fw, fh), cv2.CV_32F)
    orb = cv2.ORB_create(600)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    z0 = float(cfg.plane_distance)

    first = None
    pos0 = None
    sway_gt, surge_gt = [], []
    sway_phase, sway_phase_z, sway_ecc, sway_orb, sway_tmpl = [], [], [], [], []
    surge_ecc_s, surge_orb_s = [], []

    for step in range(70):
        if step < 8:
            vs, vy = 0.0, 0.0
        elif step < 28:
            vs, vy = 0.0, 0.5
        elif step < 36:
            vs, vy = 0.0, 0.0
        else:
            vs, vy = 0.5, 0.0
        env.step(_act(env, vs, vy))
        gray = _gray(env, fh, fw)
        pos, yaw, _pitch = _read(env)
        if first is None:
            if float(gray.std()) < 1.0:
                continue
            first = gray.copy()
            pos0 = pos.clone()
            continue
        gs, gy = _body(pos, pos0, yaw)
        z = max(0.3, z0 - float(pos[0] - pos0[0]))
        px, py, resp = _phase(first, gray, window)
        zx, zy, zresp = _phase(first, _unscale(gray, z / z0), window)
        ex, ey, escale = _ecc(first, gray)
        ox, oy, oscale, nin = _orb(first, gray, orb, matcher)
        tmpl_u = float(env._anchor_off_px[0, 0])
        if 8 <= step < 36:
            sway_gt.append(gy)
            sway_phase.append(px)
            sway_phase_z.append(zx)
            sway_ecc.append(ex)
            sway_orb.append(ox)
            sway_tmpl.append(tmpl_u)
        if step >= 36:
            surge_gt.append(gs)
            surge_ecc_s.append(escale)
            surge_orb_s.append(oscale)
        if step in (15, 27, 35, 50, 69):
            print(
                f"[match] step={step:3d} gt=({gs:+.3f},{gy:+.3f})  "
                f"phase=({px:+.1f},{py:+.1f}) r={resp:.2f}  "
                f"phaseZ=({zx:+.1f},{zy:+.1f}) r={zresp:.2f}  "
                f"ecc=({ex:+.1f},{ey:+.1f}) s={escale:.3f}  "
                f"orb=({ox:+.1f},{oy:+.1f}) s={oscale:.3f} in={nin}  "
                f"tmpl_u={tmpl_u:+.1f}",
                flush=True,
            )

    def _report(title: str, gt: list[float], est: list[float], unit: str) -> None:
        slope, corr = _corr_slope(np.asarray(gt), np.asarray(est))
        print(f"[match] {title:16} slope={slope:+8.2f} {unit}  corr={corr:+.3f}", flush=True)

    print("[match] --- sway segment: horizontal shift vs gt sway ---", flush=True)
    _report("phase dx", sway_gt, sway_phase, "px/m")
    _report("phaseZ dx", sway_gt, sway_phase_z, "px/m")
    _report("ecc tx", sway_gt, sway_ecc, "px/m")
    _report("orb tx", sway_gt, sway_orb, "px/m")
    _report("template u", sway_gt, sway_tmpl, "px/m")
    print("[match] --- surge segment: scale vs gt surge (1 = no zoom) ---", flush=True)
    _report("ecc scale", surge_gt, surge_ecc_s, "1/m")
    _report("orb scale", surge_gt, surge_orb_s, "1/m")
    env.close()
    print("[match] done", flush=True)


if __name__ == "__main__":
    main()
