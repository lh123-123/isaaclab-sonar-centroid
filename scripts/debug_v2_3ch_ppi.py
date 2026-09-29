# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dump Direct-v2 flow image 3 channels as PPI fan views (headless).

  ./isaaclab.sh -p scripts/debug_v2_3ch_ppi.py \
      --checkpoint logs/sb3/Isaac-AUV-Tracking-Direct-v2/2026-09-07_15-41-43/model_600000_steps.zip \
      --steps 12 --headless
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--out_dir", type=str, default="logs/auv_tracking/flow_v2_3ch_ppi")
AppLauncher_args = None

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import cv2
import torch
from stable_baselines3 import SAC
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.direct.isaac_auv_tracking.agents.flow_sac_policy import FlowSACPolicy  # noqa: F401
from isaaclab_tasks.direct.isaac_auv_tracking.agents.sonar_encoder import FlowSonarFeaturesExtractor  # noqa: F401
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_flow_env import AUVTrackingFlowEnvCfg


def _fan_scale(out_size, max_range, hori_fov_deg):
    half_w = float(max_range) * float(np.sin(0.5 * np.radians(hori_fov_deg)))
    return min((out_size - 1) / max(float(max_range), 1e-6),
               (out_size - 1) / max(2.0 * half_w, 1e-6))


def polar_to_fan(polar, *, min_range, max_range, hori_fov_deg, out_size=256, fill=0.0):
    """Warp polar (range × azimuth) → PPI fan: sensor at bottom-center, forward up."""
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
    sampled = ((1 - wr) * (1 - wc) * src[r0, c0] + (1 - wr) * wc * src[r0, c1]
               + wr * (1 - wc) * src[r1, c0] + wr * wc * src[r1, c1])
    out = np.full((out_size, out_size), float(fill), dtype=np.float32)
    out[in_b] = sampled[in_b]
    if polar.dtype == np.uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out


def residual_bgr(ch1_u8):
    """Signed residual: 128=gray, >128 red (brightened), <128 blue (darkened). Black base."""
    signed = ch1_u8.astype(np.int16) - 128
    vis = np.zeros((*ch1_u8.shape, 3), dtype=np.uint8)  # black base, not (40,40,40)
    pos = np.clip(signed, 0, 127).astype(np.float32) / 127.0
    neg = np.clip(-signed, 0, 127).astype(np.float32) / 127.0
    vis[:, :, 2] = np.clip(215 * pos, 0, 255).astype(np.uint8)
    vis[:, :, 0] = np.clip(215 * neg, 0, 255).astype(np.uint8)
    vis[:, :, 1] = np.clip(40 * pos * neg, 0, 255).astype(np.uint8)
    return vis


def main():
    cfg = AUVTrackingFlowEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.ocean_current.enabled = False
    # Increase sonar resolution: range_res 0.05→0.025, angular_res 2.0→1.0
    # height = (max-min)/range_res = (4.0-0.5)/0.025 = 140
    # width  = hori_fov/angular_res + 1 = 120/1.0 + 1 = 121
    cfg.oceansim_sonar_cfg.range_res = 0.025
    cfg.oceansim_sonar_cfg.angular_res = 1.0
    cfg.oceansim_sonar_cfg.height = 140
    cfg.oceansim_sonar_cfg.width = 121
    env = gym.make("Isaac-AUV-Tracking-Direct-v2", cfg=cfg)
    vec = Sb3VecEnvWrapper(env, fast_variant=True)
    raw = env.unwrapped
    ocfg = cfg.oceansim_sonar_cfg
    min_r, max_r, fov_deg = float(ocfg.min_range), float(ocfg.max_range), float(ocfg.hori_fov)

    print(f"[3ch-ppi] loading {args_cli.checkpoint}", flush=True)
    agent = SAC.load(args_cli.checkpoint, env=vec, print_system_info=False)

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fan_size = 256

    # Fan mask: 1 inside sonar sector, 0 outside → black background
    _s = _fan_scale(fan_size, max_r, fov_deg)
    _yy, _xx = np.mgrid[0:fan_size, 0:fan_size]
    _cx = 0.5 * (fan_size - 1)
    _x_m = (_xx - _cx) / _s
    _y_m = (fan_size - 1 - _yy) / _s
    _rng = np.hypot(_x_m, _y_m)
    _az = np.arctan2(_x_m, _y_m)
    _fov = float(np.radians(fov_deg))
    fan_mask = ((_rng >= min_r) & (_rng <= max_r) & (np.abs(_az) <= 0.5 * _fov)).astype(np.uint8)

    obs = vec.reset()
    for i in range(args_cli.steps):
        act, _ = agent.predict(obs, deterministic=True)
        act = np.asarray(act, dtype=np.float32).reshape(vec.num_envs, -1)
        obs, _, _, _ = vec.step(act)

        # flow_image: [N, 3, 64, 64] uint8 — ch0=curr, ch1=residual(128=no), ch2=|residual|
        img = raw._flow_image[0].detach().cpu().numpy()  # [3, 64, 64]
        ch0, ch1, ch2 = img[0], img[1], img[2]

        # Also get the raw polar for reference
        polar = raw.pipeline.polar_image[0].detach().cpu().numpy()  # [70, 61]
        polar_u8 = np.clip(polar / max(float(np.log1p(1.0)), 1e-6) * 255.0, 0, 255).astype(np.uint8)

        # Convert each channel to PPI fan
        fan_ch0 = polar_to_fan(ch0, min_range=min_r, max_range=max_r, hori_fov_deg=fov_deg, out_size=fan_size)
        fan_ch1 = polar_to_fan(ch1, min_range=min_r, max_range=max_r, hori_fov_deg=fov_deg, out_size=fan_size)
        fan_ch2 = polar_to_fan(ch2, min_range=min_r, max_range=max_r, hori_fov_deg=fov_deg, out_size=fan_size)
        fan_raw = polar_to_fan(polar_u8, min_range=min_r, max_range=max_r, hori_fov_deg=fov_deg, out_size=fan_size)

        # ch1: residual — use signed BGR colormap (gray=no change, red=brighter, blue=darker)
        fan_ch1_bgr = residual_bgr(fan_ch1)

        # ch2: |residual| — apply JET colormap for visibility
        fan_ch2_color = cv2.applyColorMap(fan_ch2, cv2.COLORMAP_JET)

        # ch0: current polar — grayscale, suppress noise floor
        fan_ch0_bgr = cv2.cvtColor(fan_ch0, cv2.COLOR_GRAY2BGR)
        fan_raw_bgr = cv2.cvtColor(fan_raw, cv2.COLOR_GRAY2BGR)
        # Suppress low-intensity noise inside fan for ch0 (keep only bright returns)
        _noise_floor = 25
        _ch0_mask = fan_ch0 < _noise_floor  # [H, W] boolean
        fan_ch0_bgr[_ch0_mask] = 0  # numpy broadcasts [H,W] mask to [H,W,3]

        # Mask outside-fan to pure black
        fan_ch0_bgr[fan_mask == 0] = 0
        fan_ch1_bgr[fan_mask == 0] = 0
        fan_ch2_color[fan_mask == 0] = 0
        fan_raw_bgr[fan_mask == 0] = 0

        # Side-by-side: raw polar | ch0 | ch1(residual) | ch2(|residual|)
        combo = np.concatenate([fan_raw_bgr, fan_ch0_bgr, fan_ch1_bgr, fan_ch2_color], axis=1)

        # Labels
        cv2.putText(fan_raw_bgr, "raw polar", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(fan_ch0_bgr, "ch0 curr", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(fan_ch1_bgr, "ch1 residual", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(fan_ch2_color, "ch2 |residual|", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        combo = np.concatenate([fan_raw_bgr, fan_ch0_bgr, fan_ch1_bgr, fan_ch2_color], axis=1)

        stem = out_dir / f"step{i:02d}"
        cv2.imwrite(str(stem) + "_raw_ppi.png", fan_raw_bgr)
        cv2.imwrite(str(stem) + "_ch0_ppi.png", fan_ch0_bgr)
        cv2.imwrite(str(stem) + "_ch1_ppi.png", fan_ch1_bgr)
        cv2.imwrite(str(stem) + "_ch2_ppi.png", fan_ch2_color)
        cv2.imwrite(str(stem) + "_combo.png", combo)

        # Stats
        r = float(raw._flow_r[0])
        a = float(raw._flow_a[0])
        valid = bool(raw._flow_valid[0])
        print(f"[3ch-ppi] step={i:02d} r={r:.2f} a={a:.3f} valid={int(valid)} "
              f"ch0[{int(ch0.min())}-{int(ch0.max())}] "
              f"ch1[{int(ch1.min())}-{int(ch1.max())}] "
              f"ch2[{int(ch2.min())}-{int(ch2.max())}] "
              f"=> {stem}_*.png", flush=True)

    print(f"[3ch-ppi] all dumps in {out_dir.resolve()}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
