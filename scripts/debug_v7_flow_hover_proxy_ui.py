#!/usr/bin/env python3
"""UI viewer for Direct-v7-flow-hover-gsrov with the CUBOID proxy (no real USD).
Shows the camera transform and intrinsics so you can verify them against the real GS-ROV camera.
Run: ./scripts/debug_v7_flow_hover_proxy_ui.py
"""
from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direct-v7-flow-hover-gsrov UI (proxy cube).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--texture", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import torch
import numpy as np

import isaaclab_tasks  # noqa: F401
import gymnasium as gym

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)

# -- env setup --
cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
# Use the proxy cuboid, NOT the real CAD USD.
cfg.use_real_usd = False
print(f"[v7-proxy-ui] use_real_usd={cfg.use_real_usd}  -> cuboid proxy active", flush=True)
if args_cli.texture:
    cfg.plane_texture_path = args_cli.texture
env = gym.make("Isaac-AUV-Tracking-Direct-v7-flow-hover-gsrov", cfg=cfg)
raw = env.unwrapped if hasattr(env, "unwrapped") else env

H, W = cfg.cam_height, cfg.cam_width

print(f"\n[v7-proxy-ui] Env created with {raw.num_envs} env(s).", flush=True)

# ---- Camera diagnostics (before stepping) ----
print("\n=== Camera Configuration ===", flush=True)
print(f"  cam_hfov_deg       = {cfg.cam_hfov_deg:.4f}°", flush=True)
print(f"  cam_aperture_mm    = {cfg.cam_aperture_mm:.4f} mm", flush=True)
print(f"  cam_width          = {cfg.cam_width} px", flush=True)
print(f"  cam_height         = {cfg.cam_height} px", flush=True)
print(f"  render_width       = {cfg.render_width} px", flush=True)
print(f"  render_height      = {cfg.render_height} px", flush=True)
print(f"  frame_stack        = {cfg.frame_stack}", flush=True)
focal = float(cfg.cam_aperture_mm * 0.5 / __import__('numpy').tan(__import__('numpy').deg2rad(cfg.cam_hfov_deg) * 0.5))
print(f"  computed focal_len = {focal:.4f} px", flush=True)
print(f"  tiled_camera width = {cfg.tiled_camera.width} px", flush=True)
print(f"  tiled_camera height= {cfg.tiled_camera.height} px", flush=True)

# Print the camera world pose from USD (set during reset)
obs, _ = env.reset()
# Robot center-of-mass world pose
pos = raw._robot.data.root_pos_w[0].cpu().numpy()
quat = raw._robot.data.root_quat_w[0].cpu().numpy()
print(f"\n=== Robot root (COM) in world ===", flush=True)
print(f"  pos = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})  m  (z positive=up, depth={pos[2]:.2f}m)", flush=True)
print(f"  quat (w,x,y,z) = ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})", flush=True)

# Try to read the camera prims from the USD stage directly
from pxr import UsdGeom
stage = raw.sim.stage
for cam_name in ("ForwardCamera0", str(cfg.view_camera_name)):
    cam_prim_path = f"{raw.scene.env_prim_paths[0]}/Robot/{cam_name}"
    cam_prim = stage.GetPrimAtPath(cam_prim_path)
    if not (cam_prim and cam_prim.IsValid()):
        print(f"\n[WARN] camera prim missing: {cam_prim_path}", flush=True)
        continue
    xf = UsdGeom.Xformable(cam_prim)
    t_op = xf.GetTranslateOp()
    t = t_op.Get() if t_op is not None else (0.0, 0.0, 0.0)
    cam = UsdGeom.Camera(cam_prim)
    print(f"\n=== Camera {cam_name} ===", flush=True)
    print(f"  path                = {cam_prim_path}", flush=True)
    print(f"  translate (body)    = ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m", flush=True)
    print(f"  => world pos        = ({pos[0]+t[0]:.4f}, {pos[1]+t[1]:.4f}, {pos[2]+t[2]:.4f}) m", flush=True)
    print(f"  focal_length_mm     = {cam.GetFocalLengthAttr().Get():.4f}", flush=True)
    print(f"  horizontal_aperture = {cam.GetHorizontalApertureAttr().Get():.4f} mm", flush=True)
    print(f"  vertical_aperture   = {cam.GetVerticalApertureAttr().Get():.4f} mm", flush=True)
    rot_op = xf.GetOrientOp()
    if rot_op is not None:
        q = rot_op.Get()
        try:
            qw, qx, qy, qz = float(q.GetReal()), *(float(v) for v in q.GetImaginary())
        except Exception:
            qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        from isaaclab.utils.math import euler_xyz_from_quat
        q_t = torch.tensor([qw, qx, qy, qz], device=raw.device).unsqueeze(0)
        roll_, pitch_, yaw_ = euler_xyz_from_quat(q_t)
        print(f"  rotation (w,x,y,z)  = ({qw:.4f}, {qx:.4f}, {qy:.4f}, {qz:.4f})", flush=True)
        print(f"  euler (deg)         = ({np.degrees(float(roll_[0])):.1f}, "
              f"{np.degrees(float(pitch_[0])):.1f}, {np.degrees(float(yaw_[0])):.1f})", flush=True)

print(f"\n=== Flow & Reward thresholds ===", flush=True)
print(f"  sigma_flow = {cfg.sigma_flow}", flush=True)
print(f"  texture_lost_limit_steps = {cfg.texture_lost_limit_steps}", flush=True)

# ---- main loop ----
print("\n[v7-proxy-ui] Stepping (zero action) ... press Ctrl+C to stop.", flush=True)
step = 0
running = True
try:
    while running and step < args_cli.steps:
        act = torch.zeros(raw.num_envs, 3)
        obs, rew, term, trunc, info = env.step(act)
        if step % 30 == 0:
            st = obs["policy"]["state"][0]
            flow_rms = float(st[8].item()) * W
            depth_err = float(st[0].item())
            att_err = torch.sqrt(st[1]**2 + st[2]**2).item()
            pos = raw._robot.data.root_pos_w[0].cpu().numpy()
            print(f"[v7] step={step:4d} rew={float(rew.mean()):.3f} flow={flow_rms:.1f}px "
                  f"depth_err={depth_err:.3f}m att={att_err:.4f} "
                  f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})", flush=True)
        if term[0] or trunc[0]:
            print(f"[v7] Terminated at step {step}", flush=True)
            break
        time.sleep(0.2)
        step += 1
except KeyboardInterrupt:
    print("\n[v7-proxy-ui] Interrupted by user.", flush=True)

env.close()
print(f"\n[v7-proxy-ui] Done after {step} steps. You can now interact with the Isaac Sim viewport.", flush=True)
print("[v7-proxy-ui] Press Esc or close the Isaac Sim window to exit.", flush=True)
