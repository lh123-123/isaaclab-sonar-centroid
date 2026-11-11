"""Calibrate sigma_flow for v7: collect flow_rms (64-grid pixels) under zero + random
actions and report mean/std so sigma_flow can be set to ~1x the observed std.

Usage:
  ./scripts/debug_auv_v7_flow_hover_gsrov.sh --num_envs 1 --steps 60
  (or run this file directly)
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate v7 sigma_flow.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--random", action="store_true", help="use small random actions instead of zero")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import torch

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)

cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
n = env.num_envs
env.reset()

vals = []
for step in range(args_cli.steps):
    if args_cli.random:
        act = (torch.rand(n, 3) - 0.5) * 0.4
    else:
        act = torch.zeros(n, 4)
    env.step(act)
    flow_px = env._flow_rms.detach().cpu().numpy()
    vals.extend(float(x) for x in flow_px)
    print(f"[calib] step {step:2d} flow_rms_px_mean={float(flow_px.mean()):.3f}", flush=True)

import statistics
arr = vals
mean = statistics.fmean(arr)
std = statistics.pstdev(arr)
mn = min(arr)
mx = max(arr)
print("\n[calib] flow_rms (64-grid px):", flush=True)
print(f"  n={len(arr)} mean={mean:.3f} std={std:.3f} min={mn:.3f} max={mx:.3f}", flush=True)
norm_mean = mean / 64.0
norm_std = std / 64.0
print(f"  normalized (÷64): mean={norm_mean:.4f} std={norm_std:.4f}", flush=True)
sug = min(0.5, max(0.05, norm_std))
print(f"  suggested sigma_flow (1x std, clamped [0.05,0.5]) = {sug:.4f}", flush=True)
env.close()
raise SystemExit(0)
