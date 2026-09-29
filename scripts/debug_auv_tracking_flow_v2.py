# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize Direct-v2 residual + reference / centroid overlays.

  # Isaac-free synthetic blobs:
  python scripts/debug_auv_tracking_flow_v2.py --synthetic \\
      --out_dir logs/auv_tracking/flow_v2_debug

  # Live OceanSim + optional 500k policy, fan PPI dumps:
  ./scripts/debug_auv_tracking_flow_v2.sh --headless --checkpoint PATH --steps 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _channel_stats(name: str, ch: np.ndarray) -> str:
    return (
        f"{name}: min={int(ch.min())} max={int(ch.max())} mean={float(ch.mean()):.1f} "
        f"std={float(ch.std()):.1f}"
    )


def _residual_bgr(ch1_u8: np.ndarray) -> np.ndarray:
    """Signed residual: 128=gray (no change), >128 red (brightened), <128 blue (darkened)."""
    import cv2

    signed = ch1_u8.astype(np.int16) - 128
    vis = np.zeros((*ch1_u8.shape, 3), dtype=np.uint8)
    vis[:, :] = (40, 40, 40)
    pos = np.clip(signed, 0, 127).astype(np.float32) / 127.0
    neg = np.clip(-signed, 0, 127).astype(np.float32) / 127.0
    vis[:, :, 2] = np.clip(40 + 215 * pos, 0, 255).astype(np.uint8)
    vis[:, :, 0] = np.clip(40 + 215 * neg, 0, 255).astype(np.uint8)
    vis[:, :, 1] = np.clip(40 + 40 * (1.0 - pos - neg), 0, 255).astype(np.uint8)
    return vis


def _fan_scale(out_size: int, max_range: float, hori_fov_deg: float) -> float:
    """Pixels per metre so the FOV fan fits in a square (sensor at bottom-center)."""
    half_w = float(max_range) * float(np.sin(0.5 * np.radians(hori_fov_deg)))
    return min((out_size - 1) / max(float(max_range), 1e-6), (out_size - 1) / max(2.0 * half_w, 1e-6))


def _polar_to_fan(
    polar: np.ndarray,
    *,
    min_range: float,
    max_range: float,
    hori_fov_deg: float,
    out_size: int = 256,
    fill: float = 0.0,
) -> np.ndarray:
    """Warp polar (range × azimuth) to a PPI fan: vehicle at bottom-center, forward up."""
    src = polar.astype(np.float32)
    h, w = src.shape
    fov = float(np.radians(hori_fov_deg))
    scale = _fan_scale(out_size, max_range, hori_fov_deg)
    yy, xx = np.mgrid[0:out_size, 0:out_size]
    cx = 0.5 * (out_size - 1)
    x_m = (xx - cx) / scale
    y_m = (out_size - 1 - yy) / scale
    rng = np.hypot(x_m, y_m)
    az = np.arctan2(x_m, y_m)
    span = max(float(max_range) - float(min_range), 1e-6)
    row = (rng - float(min_range)) / span * (h - 1)
    col = (float(w) / 2.0) - (np.degrees(az) * float(w) / max(float(hori_fov_deg), 1e-6))
    in_b = (rng >= float(min_range)) & (rng <= float(max_range)) & (np.abs(az) <= 0.5 * fov)
    r0 = np.clip(np.floor(row).astype(np.int32), 0, h - 1)
    c0 = np.clip(np.floor(col).astype(np.int32), 0, w - 1)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)
    wr = (row - np.floor(row)).astype(np.float32)
    wc = (col - np.floor(col)).astype(np.float32)
    sampled = (
        (1 - wr) * (1 - wc) * src[r0, c0]
        + (1 - wr) * wc * src[r0, c1]
        + wr * (1 - wc) * src[r1, c0]
        + wr * wc * src[r1, c1]
    )
    out = np.full((out_size, out_size), float(fill), dtype=np.float32)
    out[in_b] = sampled[in_b]
    if polar.dtype == np.uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out


def _ra_to_fan_uv(
    r: float,
    alpha: float,
    *,
    max_range: float,
    hori_fov_deg: float,
    out_size: int,
) -> tuple[int, int]:
    scale = _fan_scale(out_size, max_range, hori_fov_deg)
    cx = 0.5 * (out_size - 1)
    u = cx + float(r) * float(np.sin(alpha)) * scale
    v = (out_size - 1) - float(r) * float(np.cos(alpha)) * scale
    return int(round(u)), int(round(v))


def _intensity_centroid_hw(img: np.ndarray, *, threshold_ratio: float = 0.35) -> tuple[float, float] | None:
    """Weighted centroid → (col, row) on a 2D intensity image."""
    arr = np.asarray(img, dtype=np.float32)
    peak = float(arr.max()) if arr.size else 0.0
    if peak <= 1.0e-6:
        return None
    mask = arr >= peak * float(threshold_ratio)
    if int(mask.sum()) < 2:
        return None
    ys, xs = np.nonzero(mask)
    w = arr[mask]
    return float(np.average(xs, weights=w)), float(np.average(ys, weights=w))


def _small_dot(vis: np.ndarray, xy: tuple[float, float], color: tuple[int, int, int], radius: int = 2) -> None:
    import cv2

    x, y = int(round(xy[0])), int(round(xy[1]))
    h, w = vis.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(vis, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)


def _draw_fan_guides(
    vis: np.ndarray,
    *,
    max_range: float,
    hori_fov_deg: float,
    r_star: float,
    draw_ref: bool = False,
) -> None:
    import cv2

    h, w = vis.shape[:2]
    assert h == w
    scale = _fan_scale(h, max_range, hori_fov_deg)
    cx = 0.5 * (w - 1)
    cy = h - 1
    fov = float(np.radians(hori_fov_deg))
    for rng in (1.0, 2.0, 3.0, float(max_range)):
        rad = int(round(rng * scale))
        if rad <= 0:
            continue
        cv2.ellipse(
            vis,
            (int(round(cx)), int(round(cy))),
            (rad, rad),
            0,
            270.0 - 0.5 * float(hori_fov_deg),
            270.0 + 0.5 * float(hori_fov_deg),
            (70, 70, 70),
            1,
        )
    for sign in (-1.0, 1.0):
        az = sign * 0.5 * fov
        u = int(round(cx + max_range * np.sin(az) * scale))
        v = int(round(cy - max_range * np.cos(az) * scale))
        cv2.line(vis, (int(round(cx)), int(round(cy))), (u, v), (70, 70, 70), 1)
    us, vs = _ra_to_fan_uv(r_star, 0.0, max_range=max_range, hori_fov_deg=hori_fov_deg, out_size=h)
    if draw_ref:
        cv2.drawMarker(vis, (us, vs), (0, 165, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=1)


def _montage_fan(ch0: np.ndarray, ch1: np.ndarray, ch2: np.ndarray, titles: list[str]) -> np.ndarray:
    import cv2

    panels = []
    for ch, title in zip((ch0, ch1, ch2), titles, strict=True):
        if "ch1" in title:
            bgr = _residual_bgr(ch)
        else:
            bgr = cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)
        cv2.putText(bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(bgr)
    return np.concatenate(panels, axis=1)


def _draw_overlay(
    polar_u8: np.ndarray,
    *,
    row: float | None,
    col: float | None,
    row_star: float,
    col_star: float,
    color_cur: tuple[int, int, int] = (0, 255, 0),
    color_ref: tuple[int, int, int] = (0, 165, 255),
) -> np.ndarray:
    import cv2

    if polar_u8.ndim == 2:
        vis = cv2.cvtColor(polar_u8, cv2.COLOR_GRAY2BGR)
    else:
        vis = polar_u8.copy()
    h, w = vis.shape[:2]
    cx, cy = int(round(col_star)), int(round(row_star))
    cv2.drawMarker(vis, (cx, cy), color_ref, markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=1)
    if row is not None and col is not None:
        x, y = int(round(col)), int(round(row))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis, (x, y), 3, color_cur, thickness=1)
            cv2.arrowedLine(vis, (cx, cy), (x, y), (255, 255, 0), 1, tipLength=0.25)
    return vis


def _load_flow_obs():
    """Load flow_obs.py without importing isaaclab_tasks (avoids Isaac Sim)."""
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "direct"
        / "isaac_auv_tracking"
        / "flow_obs.py"
    )
    spec = importlib.util.spec_from_file_location("auv_tracking_flow_obs", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic(out_dir: Path) -> None:
    import cv2
    import torch

    fo = _load_flow_obs()
    intensity_centroid_polar = fo.intensity_centroid_polar
    make_gaussian_blob = fo.make_gaussian_blob
    normalize_polar_fixed = fo.normalize_polar_fixed
    pack_flow_image = fo.pack_flow_image
    photometric_residual = fo.photometric_residual
    reference_polar_pixel = fo.reference_polar_pixel

    h, w = 70, 61
    min_r, max_r, fov = 0.5, 4.0, 120.0
    row0, col0 = 28.0, 30.0
    prev = make_gaussian_blob(height=h, width=w, row=row0, col=col0)
    curr = make_gaussian_blob(height=h, width=w, row=row0 + 5.0, col=col0 - 4.0)
    curr_01 = normalize_polar_fixed(curr.unsqueeze(0))
    prev_01 = normalize_polar_fixed(prev.unsqueeze(0))
    residual = photometric_residual(curr_01, prev_01, torch.ones(1, dtype=torch.bool))
    packed = pack_flow_image(curr_01, residual, size=(64, 64))[0].cpu().numpy()
    r, a, valid = intensity_centroid_polar(
        curr.unsqueeze(0), min_range=min_r, max_range=max_r, hori_fov_deg=fov
    )
    row_star, col_star = reference_polar_pixel(
        r_star=2.0, min_range=min_r, max_range=max_r, polar_height=h, polar_width=w
    )
    polar_u8 = (curr_01[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    h_max = float(h - 1)
    span = max_r - min_r
    row_c = float((float(r[0]) - min_r) / span * h_max)
    col_c = float(w) / 2.0 - float(a[0]) * (180.0 / np.pi) * w / fov
    overlay = _draw_overlay(polar_u8, row=row_c, col=col_c, row_star=row_star, col_star=col_star)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "synthetic_polar.png"), overlay)
    cv2.imwrite(str(out_dir / "synthetic_residual_ch.png"), packed[1])
    print(f"[synthetic] valid={bool(valid[0])} r={float(r[0]):.3f} a={float(a[0]):.3f}")
    print(f"[synthetic] wrote {out_dir / 'synthetic_polar.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Direct-v2 flow residual / reference.")
    parser.add_argument("--synthetic", action="store_true", help="Isaac-free synthetic blobs.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=2, help="Save PNG every N steps.")
    parser.add_argument("--out_dir", type=str, default="logs/auv_tracking/flow_v2_centroid_debug")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/sb3/Isaac-AUV-Tracking-Direct-v2/2026-09-04_17-55-26/model_500000_steps.zip",
        help="SAC zip. Empty string = zero actions.",
    )
    args, unknown = parser.parse_known_args()
    if args.synthetic:
        _synthetic(Path(args.out_dir))
        return

    import sys

    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]] + unknown
    live_parser = argparse.ArgumentParser()
    live_parser.add_argument("--num_envs", type=int, default=args.num_envs)
    live_parser.add_argument("--steps", type=int, default=args.steps)
    live_parser.add_argument("--save_every", type=int, default=args.save_every)
    live_parser.add_argument("--out_dir", type=str, default=args.out_dir)
    live_parser.add_argument("--checkpoint", type=str, default=args.checkpoint)
    AppLauncher.add_app_launcher_args(live_parser)
    live_args = live_parser.parse_args()
    live_args.enable_cameras = True
    app_launcher = AppLauncher(live_args)
    simulation_app = app_launcher.app

    import cv2
    import gymnasium as gym
    import torch
    from stable_baselines3 import SAC

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
    from isaaclab_tasks.direct.isaac_auv_tracking.agents.flow_sac_policy import FlowSACPolicy  # noqa: F401
    from isaaclab_tasks.direct.isaac_auv_tracking.agents.sonar_encoder import (  # noqa: F401
        FlowSonarFeaturesExtractor,
    )
    from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_flow_env import AUVTrackingFlowEnvCfg

    from isaaclab_tasks.direct.isaac_auv_tracking.sonar_meas import azimuth_to_col, body_polar_from_xy, col_to_azimuth_rad
    from isaaclab_tasks.direct.sonar_centroid_track.modules.coordinates import polar_range_m_to_row, polar_row_to_range_m

    cfg = AUVTrackingFlowEnvCfg()
    cfg.scene.num_envs = live_args.num_envs
    env = gym.make("Isaac-AUV-Tracking-Direct-v2", cfg=cfg)
    vec = Sb3VecEnvWrapper(env, fast_variant=True)
    raw = env.unwrapped
    ckpt = str(getattr(live_args, "checkpoint", "") or "").strip()
    agent = None
    if ckpt:
        print(f"[live] loading policy {ckpt}", flush=True)
        agent = SAC.load(ckpt, env=vec, print_system_info=False)
    obs = vec.reset()
    out_dir = Path(live_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ocfg = cfg.oceansim_sonar_cfg
    min_r = float(ocfg.min_range)
    max_r = float(ocfg.max_range)
    fov_deg = float(ocfg.hori_fov)
    r_star = float(cfg.flow_ref_range_m)
    a_star = float(cfg.flow_ref_bearing_rad)
    fan_size = 256
    save_every = max(int(getattr(live_args, "save_every", 2)), 1)
    print(
        "[live] centroid debug: polar + PPI, small dots  green=centroid  cyan=GT  magenta=peak  orange=p*\n"
        f"  p* = ({r_star:.1f}m, {a_star:.2f}rad)  fov={fov_deg:.0f}deg  max_range={max_r:.1f}m",
        flush=True,
    )
    for i in range(live_args.steps):
        if agent is None:
            act = np.zeros((vec.num_envs, 2), dtype=np.float32)
        else:
            act, _ = agent.predict(obs, deterministic=True)
            act = np.asarray(act, dtype=np.float32).reshape(vec.num_envs, -1)
        obs, _, _, _ = vec.step(act)
        polar = raw.pipeline.polar_image[0].detach().cpu().numpy()
        img = raw._flow_image[0].detach().cpu().numpy()
        ch0, ch1, ch2 = img[0], img[1], img[2]
        st = raw._flow_state[0].detach().cpu().numpy()
        r = float(raw._flow_r[0])
        a = float(raw._flow_a[0])
        valid = bool(raw._flow_valid[0])
        peak = float(polar.max()) if polar.size else 0.0
        h, w = polar.shape
        fov_rad = float(np.radians(fov_deg))
        sonar_xy, sonar_yaw = raw._sonar_pose_xy_yaw()
        gt_a_t, gt_r_t = body_polar_from_xy(raw._target_pos[:, :2], sonar_xy, sonar_yaw)
        gt_r = float(gt_r_t[0])
        gt_a = float(gt_a_t[0])
        cen_row = polar_range_m_to_row(r, min_range=min_r, max_range=max_r, polar_height=h) if valid else None
        cen_col = azimuth_to_col(a, w, fov_rad) if valid else None
        gt_row = polar_range_m_to_row(gt_r, min_range=min_r, max_range=max_r, polar_height=h)
        gt_col = azimuth_to_col(gt_a, w, fov_rad)
        star_row = polar_range_m_to_row(r_star, min_range=min_r, max_range=max_r, polar_height=h)
        star_col = azimuth_to_col(a_star, w, fov_rad)
        peak_idx = np.unravel_index(int(np.argmax(polar)), polar.shape)
        peak_row, peak_col = float(peak_idx[0]), float(peak_idx[1])
        peak_r = polar_row_to_range_m(peak_row, min_range=min_r, max_range=max_r, polar_height=h)
        peak_a = col_to_azimuth_rad(peak_col, w, fov_rad)
        polar_u8 = np.clip(polar / max(float(np.log1p(1.0)), 1e-6) * 255.0, 0, 255).astype(np.uint8)
        fan_native = _polar_to_fan(
            polar_u8, min_range=min_r, max_range=max_r, hori_fov_deg=fov_deg, out_size=fan_size, fill=0.0
        )
        fan_cent = _intensity_centroid_hw(fan_native)
        uc = vc = None
        if valid:
            uc, vc = _ra_to_fan_uv(r, a, max_range=max_r, hori_fov_deg=fov_deg, out_size=fan_size)
        ug, vg = _ra_to_fan_uv(gt_r, gt_a, max_range=max_r, hori_fov_deg=fov_deg, out_size=fan_size)
        up, vp = _ra_to_fan_uv(peak_r, peak_a, max_range=max_r, hori_fov_deg=fov_deg, out_size=fan_size)
        us, vs = _ra_to_fan_uv(r_star, a_star, max_range=max_r, hori_fov_deg=fov_deg, out_size=fan_size)
        print(
            f"[live] step={i:02d} act=[{float(act[0,0]):+.2f},{float(act[0,1]):+.2f}] "
            f"cen r={r:.3f} a={a:.3f}  GT r={gt_r:.3f} a={gt_a:.3f}  "
            f"dR={r-gt_r:+.3f} dA={a-gt_a:+.3f} valid={int(valid)}",
            flush=True,
        )
        last = i == live_args.steps - 1
        do_save = last or (i % save_every == 0)
        if do_save:
            px = 6
            polar_bgr = cv2.cvtColor(polar_u8, cv2.COLOR_GRAY2BGR)
            polar_big = cv2.resize(polar_bgr, (w * px, h * px), interpolation=cv2.INTER_NEAREST)
            def _pd(col, row, color):
                _small_dot(polar_big, (col * px, row * px), color, radius=2)
            _pd(star_col, star_row, (0, 165, 255))
            _pd(gt_col, gt_row, (255, 255, 0))
            _pd(peak_col, peak_row, (255, 0, 255))
            if cen_col is not None and cen_row is not None:
                _pd(cen_col, cen_row, (0, 255, 0))
            cv2.putText(polar_big, "polar 70x61  g=cen y=GT m=peak o=p*", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            ppi = cv2.cvtColor(fan_native, cv2.COLOR_GRAY2BGR)
            _draw_fan_guides(ppi, max_range=max_r, hori_fov_deg=fov_deg, r_star=r_star, draw_ref=False)
            _small_dot(ppi, (us, vs), (0, 165, 255), 2)
            _small_dot(ppi, (ug, vg), (255, 255, 0), 2)
            _small_dot(ppi, (up, vp), (255, 0, 255), 2)
            if uc is not None:
                _small_dot(ppi, (uc, vc), (0, 255, 0), 2)
            if fan_cent is not None:
                _small_dot(ppi, fan_cent, (0, 0, 255), 2)  # red: centroid computed on PPI pixels
            cv2.putText(ppi, "PPI  g=cen y=GT m=peak o=p*  r=PPI-cent", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # pad polar to fan height for side-by-side
            pad_h = max(fan_size - polar_big.shape[0], 0)
            polar_pad = cv2.copyMakeBorder(polar_big, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if polar_pad.shape[0] > fan_size:
                polar_pad = polar_pad[:fan_size]
            compare = np.concatenate([polar_pad, ppi], axis=1)
            stem = out_dir / f"step{i:02d}"
            cv2.imwrite(str(stem) + "_polar.png", polar_big)
            cv2.imwrite(str(stem) + "_ppi.png", ppi)
            cv2.imwrite(str(stem) + "_compare.png", compare)
            cv2.imwrite(str(stem) + "_fan_ch0.png", ch0)
            print(f"       wrote {stem}_polar.png {stem}_ppi.png {stem}_compare.png", flush=True)
    print(f"[live] all dumps in {out_dir.resolve()}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
