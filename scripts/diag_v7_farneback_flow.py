#!/usr/bin/env python3
"""Farneback optical-flow diagnostic for Direct-v7-flow-hover (GS-ROV).

Runs the canonical v7 env (cuboid proxy, calm water), drives a scripted motion
surge -> sway -> yaw so there is real image motion, then:
  * records the env's own flow_rms / du / dv / texture coverage per step,
  * dumps RGB + texture mask + du + dv + |flow| + quiver montages to PNG,
  * sweeps Farneback params (winsize / levels) on the SAME frame pair to see
    whether the current settings are reasonable.

Usage:
    ./isaaclab.sh -p scripts/diag_v7_farneback_flow.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser(description="Farneback flow diagnostic for v7.")
parser.add_argument("--out", type=str, default="/home/lh/IsaacLab/diag_farneback")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)

os.makedirs(args_cli.out, exist_ok=True)

# --------------------------------------------------------------------------- #
# env
# --------------------------------------------------------------------------- #
cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
cfg.scene.num_envs = 1
cfg.use_real_usd = False
# Docstring says calm water. The training cfg enables a ~0.4 m/s current, which
# would add a drift flow on top of the scripted motion and hide the noise floor.
cfg.ocean_current.enabled = False
env = AUVTrackingV7FlowHoverGsrovEnv(cfg)

fh, fw = int(cfg.flow_render_height), int(cfg.flow_render_width)
print(f"[diag] render {cfg.render_width}x{cfg.render_height} -> flow input {fw}x{fh}", flush=True)
print(f"[diag] flow_method={cfg.flow_method}  min_texture_px={cfg.min_texture_px} "
      f"grad_thr={cfg.texture_grad_threshold}  sigma_flow={cfg.sigma_flow}", flush=True)


def farneback(prev_rgb_u8, curr_rgb_u8, winsize=21, levels=3, iters=2, poly_n=5, poly_sigma=1.2):
    """prev/curr: (rh,rw,3) uint8 numpy. Returns (H,W,2) float32 px flow at flow res."""
    p = cv2.resize(prev_rgb_u8, (fw, fh), interpolation=cv2.INTER_AREA)
    c = cv2.resize(curr_rgb_u8, (fw, fh), interpolation=cv2.INTER_AREA)
    gp = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
    gc = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        gp, gc, None, pyr_scale=0.5, levels=levels, winsize=winsize,
        iterations=iters, poly_n=poly_n, poly_sigma=poly_sigma, flags=0,
    )


def grab_rgb():
    return env._camera_rgb()[0].detach().cpu().numpy()  # (rh,rw,3) uint8


# --------------------------------------------------------------------------- #
# scripted motion: surge -> sway -> yaw   (action in [-1,1]^3 -> cmd_vel*0.5)
# --------------------------------------------------------------------------- #
# hold first: noise floor with a valid previous frame and no commanded motion.
phases = [("hold", 0, 0.0, 6), ("surge", 0, 0.8, 8), ("sway", 1, 0.8, 8), ("yaw", 2, 0.8, 10)]
rec = {
    "step": [], "phase": [], "flow_rms": [], "du": [], "dv": [], "tex": [],
    "v_surge": [], "v_sway": [], "yaw_rate": [], "div": [], "abs_u": [],
}

env.reset()
# warm up so first PrevFrame is valid
zero = torch.zeros(1, 3, device=env.device)
for _ in range(3):
    env.step(zero)

saved_pair = None  # (prev_rgb, curr_rgb) at a clear-motion moment
saved_env_flow = None
step = 0
prev_rgb = grab_rgb()
for name, axis, mag, n in phases:
    for _ in range(n):
        act = zero.clone()
        act[0, axis] = mag
        env.step(act)
        curr_rgb = grab_rgb()
        fi = env._flow_image[0].detach().cpu().numpy()  # (2,64,64) normalized
        # Expansion check: du should grow with image-x under forward surge.
        xs = np.linspace(-1.0, 1.0, fi.shape[-1], dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, fi.shape[-2], dtype=np.float32)
        xx = np.broadcast_to(xs, fi.shape[-2:])
        yy = np.broadcast_to(ys[:, None], fi.shape[-2:])
        div = float((fi[0] * xx + fi[1] * yy).mean())
        st = env._ego_vel_state()[0].detach().cpu().numpy()
        rec["step"].append(step)
        rec["phase"].append(name)
        rec["flow_rms"].append(float(env._flow_rms[0].item()))
        rec["du"].append(float(fi[0].mean()))
        rec["dv"].append(float(fi[1].mean()))
        rec["tex"].append(float(env._texture_count[0].item()))
        rec["v_surge"].append(float(st[4]))
        rec["v_sway"].append(float(st[5]))
        rec["yaw_rate"].append(float(st[7]))
        rec["div"].append(div)
        rec["abs_u"].append(float(np.abs(fi[0]).mean()))
        # capture a frame pair mid-yaw (clearest uniform horizontal flow)
        if name == "yaw" and step == rec["step"][len(rec["step"]) - 1] and saved_pair is None and _ == 4:
            saved_pair = (prev_rgb.copy(), curr_rgb.copy())
            saved_env_flow = fi.copy()
        prev_rgb = curr_rgb
        step += 1

print("\n[diag] per-phase summary (flow_rms in px @64 grid):", flush=True)
import statistics
for name, *_ in phases:
    fr = [r for r, p in zip(rec["flow_rms"], rec["phase"]) if p == name]
    du = [r for r, p in zip(rec["du"], rec["phase"]) if p == name]
    dv = [r for r, p in zip(rec["dv"], rec["phase"]) if p == name]
    tx = [r for r, p in zip(rec["tex"], rec["phase"]) if p == name]
    vs = [r for r, p in zip(rec["v_surge"], rec["phase"]) if p == name]
    vy = [r for r, p in zip(rec["v_sway"], rec["phase"]) if p == name]
    wz = [r for r, p in zip(rec["yaw_rate"], rec["phase"]) if p == name]
    dvg = [r for r, p in zip(rec["div"], rec["phase"]) if p == name]
    abu = [r for r, p in zip(rec["abs_u"], rec["phase"]) if p == name]
    print(f"  {name:5s} flow_rms mean={statistics.mean(fr):6.2f}px  "
          f"du mean={statistics.mean(du):+.4f}  dv mean={statistics.mean(dv):+.4f}  "
          f"|du|={statistics.mean(abu):.4f}  div={statistics.mean(dvg):+.5f}  "
          f"v=({statistics.mean(vs):+.3f},{statistics.mean(vy):+.3f}) m/s  "
          f"yaw_rate={statistics.mean(wz):+.3f} rad/s  "
          f"tex~{statistics.mean(tx):.0f}/{64*64}", flush=True)

# --------------------------------------------------------------------------- #
# Farneback param sweep on the captured yaw pair
# --------------------------------------------------------------------------- #
if saved_pair is None:
    saved_pair = (prev_rgb, grab_rgb())
    saved_env_flow = env._flow_image[0].detach().cpu().numpy()

prev_rgb, curr_rgb = saved_pair
print("\n[diag] Farneback parameter sweep on one yaw frame pair:", flush=True)
print("  levels winsize | mean|f| px  median|f| px  %zero", flush=True)
sweep = []
for levels in (2, 3, 4):
    for ws in (15, 21, 31):
        # iters=2 matches the env. The old sweep used iters=3 and could not be
        # compared with the flow the policy actually sees.
        fl = farneback(prev_rgb, curr_rgb, winsize=ws, levels=levels, iters=2)
        mag = np.linalg.norm(fl, axis=-1)
        pct_zero = float((mag < 0.1).mean() * 100.0)
        sweep.append((levels, ws, float(mag.mean()), float(np.median(mag)), pct_zero))
        print(f"  {levels:6d} {ws:7d} | {mag.mean():11.3f}  {np.median(mag):12.3f}  {pct_zero:5.1f}", flush=True)

# --------------------------------------------------------------------------- #
# montage using CURRENT default params
# --------------------------------------------------------------------------- #
flow = farneback(prev_rgb, curr_rgb, winsize=21, levels=3, iters=2)
mag = np.linalg.norm(flow, axis=-1)

# env texture mask at 64x64
mask = env._texture_mask[0, 0].detach().cpu().numpy()

fig, ax = plt.subplots(3, 2, figsize=(13, 10))
ax[0, 0].imshow(curr_rgb)
ax[0, 0].set_title("camera RGB (render)")
ax[0, 1].imshow(mask, cmap="gray")
ax[0, 1].set_title(f"texture mask @64  ({mask.sum():.0f} px)")
im1 = ax[1, 0].imshow(flow[..., 0], cmap="RdBu_r", vmin=-np.abs(flow).max(), vmax=np.abs(flow).max())
ax[1, 0].set_title("du (horizontal, px)")
fig.colorbar(im1, ax=ax[1, 0], fraction=0.046)
im2 = ax[1, 1].imshow(flow[..., 1], cmap="RdBu_r", vmin=-np.abs(flow).max(), vmax=np.abs(flow).max())
ax[1, 1].set_title("dv (vertical, px)")
fig.colorbar(im2, ax=ax[1, 1], fraction=0.046)
im3 = ax[2, 0].imshow(mag, cmap="viridis")
ax[2, 0].set_title(f"|flow| px  (mean {mag.mean():.2f})")
fig.colorbar(im3, ax=ax[2, 0], fraction=0.046)
# quiver
ys, xs = np.mgrid[0:fh:16, 0:fw:16]
u = flow[::16, ::16, 0]
v = flow[::16, ::16, 1]
ax[2, 1].imshow(curr_rgb)
ax[2, 1].quiver(xs, ys, u, v, color="yellow", scale=50, width=0.004)
ax[2, 1].set_title("flow quiver on RGB (subsampled)")
for a in ax.ravel():
    a.axis("off") if a.get_images() or a.get_title().startswith("flow quiver") else None
fig.suptitle("v7 Farneback flow — yaw motion (default params: ws21/lev3/iter2)", fontsize=13)
fig.tight_layout()
montage_path = os.path.join(args_cli.out, "flow_montage.png")
fig.savefig(montage_path, dpi=110)
print(f"\n[diag] saved montage -> {montage_path}", flush=True)

# --------------------------------------------------------------------------- #
# time-series plot
# --------------------------------------------------------------------------- #
fig2, axx = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axx[0].plot(rec["step"], rec["flow_rms"], "-o", ms=3)
axx[0].set_ylabel("flow_rms (px @64)")
axx[0].set_title("v7 flow response to hold -> surge -> sway -> yaw")
for b in (5.5, 13.5, 21.5):
    axx[0].axvline(b, color="gray", ls="--", alpha=0.6)
axx[1].plot(rec["step"], rec["du"], "-o", ms=3, label="mean du")
axx[1].plot(rec["step"], rec["dv"], "-s", ms=3, label="mean dv")
axx[1].set_ylabel("mean flow dir (normalized)")
axx[1].set_xlabel("step")
axx[1].legend()
for b in (5.5, 13.5, 21.5):
    axx[1].axvline(b, color="gray", ls="--", alpha=0.6)
fig2.tight_layout()
ts_path = os.path.join(args_cli.out, "flow_timeseries.png")
fig2.savefig(ts_path, dpi=110)
print(f"[diag] saved timeseries -> {ts_path}", flush=True)

env.close()
print("[diag] done.", flush=True)
