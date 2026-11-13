#!/usr/bin/env python3
"""GUI of the level-attitude handover: zero action, thrusters holding level.

Viewport follows the ROV camera. No policy — the attitude PID alone holds
roll = pitch = 0 while the ocean current is on, which is the state the
thruster-budget numbers were measured in.

  ./isaaclab.sh -p scripts/play_auv_v7_level_ui.py
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GUI of the level-attitude hold, zero action.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--pace_hz", type=float, default=15.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import torch
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)

cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.use_real_usd = False
env = AUVTrackingV7FlowHoverGsrovEnv(cfg)

if not args_cli.headless:
    try:
        from omni.kit.viewport.utility import get_viewport_from_window_name

        cam_path = f"{env.scene.env_prim_paths[0]}/Robot/{env.cfg.view_camera_name}"
        vp = get_viewport_from_window_name("Viewport")
        if vp is not None:
            vp.set_active_camera(cam_path)
            print(f"[level-ui] viewport camera -> {cam_path}", flush=True)
    except Exception as exc:
        print(f"[level-ui] viewport camera switch failed: {exc}", flush=True)

env.reset()
print(
    f"[level-ui] viewport open. zero action, level attitude, ocean current on. "
    f"pre_charge={getattr(env, '_level_pitch_moment', 0.0):.3f}",
    flush=True,
)

t0 = time.perf_counter()
pace_dt = 1.0 / args_cli.pace_hz if args_cli.pace_hz > 0 else 0.0
for step in range(args_cli.steps):
    env.step(torch.zeros(env.num_envs, 3))
    if pace_dt > 0.0:
        sleep_for = (t0 + (step + 1) * pace_dt) - time.perf_counter()
        if sleep_for > 0.0:
            time.sleep(sleep_for)
    if step % 30 == 0:
        roll, pitch, _ = euler_xyz_from_quat(env._robot.data.root_quat_w[0:1])
        pwm = env._pwm[0].detach().cpu().numpy()
        verts = pwm[[0, 1, 2, 7]]
        print(
            f"[level-ui] step={step:4d} roll={float(torch.rad2deg(roll).cpu()):.2f} deg "
            f"pitch={float(torch.rad2deg(pitch).cpu()):.2f} deg "
            f"vert_pwm=({verts[0]:+.2f},{verts[1]:+.2f},{verts[2]:+.2f},{verts[3]:+.2f})",
            flush=True,
        )

env.close()
print("[level-ui] done", flush=True)
