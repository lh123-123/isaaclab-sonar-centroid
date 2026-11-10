#!/usr/bin/env python3
"""Headless: zero-action drift vs ocean current, and whether optical flow can see it.

Prints per-step ROV motion (pos, world/body vel), the current velocity at the
ROV position, the measured flow (rms px @ flow-render res), and the *predicted*
pixel displacement per frame from the actual body velocity — so we can tell
whether Farneback is losing the slow drift or the ROV isn't actually drifting.

Also prints the first-frame-anchor readout (template-match offset px, metric
displacement, ncc) next to the GT horizontal drift — the anchor signal must
track the GT drift for the v7 anchor objective to be learnable.

Usage:
    ./isaaclab.sh -p scripts/debug_auv_v7_drift_flow.py --headless --num_envs 1 --steps 300
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 zero-action drift vs flow.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--print_every", type=int, default=10)
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


def main():
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    n = env.num_envs

    # Frame spacing for flow estimation: one frame per policy step.
    dt_step = cfg.decimation * cfg.sim.dt
    # Camera geometry: pixel scale at the flow-estimation resolution.
    fw = float(cfg.flow_render_width)
    hfov = math.radians(float(cfg.cam_hfov_deg))
    f_px = fw / (2.0 * math.tan(hfov / 2.0))  # px per radian @ 320-wide
    W = float(cfg.cam_width)

    obs, _ = env.reset()
    pos0 = env._robot.data.root_pos_w.clone()

    print(
        f"\n[drift] envs={n} steps={args_cli.steps} dt={dt_step:.4f}s "
        f"({1.0/dt_step:.1f} Hz) f_px={f_px:.1f} px/rad plane@x={cfg.plane_distance}m "
        f"current={cfg.ocean_current.level} dir={cfg.ocean_current.params.get('direction')}"
    )
    print(
        "[drift] cols: step t | pos(x,y,z) drift(dx,dy) | vel_w(vx,vy,vz) |v_h| | "
        "current(cx,cy) | flow_rms_px320 pred_px320 tex_px | "
        "anchor(u,v)px anchor_d_m ncc | gt_h_m"
    )

    for step in range(args_cli.steps):
        if step == 0:
            print("[drift] entering step 0", flush=True)
        act = torch.zeros(n, 3)
        obs, rew, term, trunc, info = env.step(act)

        if step % args_cli.print_every == 0 or step == args_cli.steps - 1:
            pos = env._robot.data.root_pos_w[0]
            vel = env._robot.data.root_lin_vel_w[0]
            drift = pos - pos0[0]
            st = [float(x) for x in obs["policy"]["state"][0]]
            depth, roll, pitch, yaw_s = st[0], st[1], st[2], st[3]
            v_surge, v_sway, v_heave, yaw_rate = st[4], st[5], st[6], st[7]
            flow_n = st[8]
            flow_rms_px320 = flow_n * W  # rms in flow-render (176×320) pixels
            tex_px = int(obs["policy"]["image"][0, 6].sum().item())

            t = float(env._robot.data.root_pos_w.new_tensor(0)) if False else float(env.episode_length_buf[0]) * dt_step
            # current at ROV position
            cur = torch.zeros(3)
            if getattr(env, "_ocean_current_field", None) is not None:
                cur = env._ocean_current_field(env._robot.data.root_pos_w, t).flatten()[:3]

            # Predicted pixel displacement per frame @ 320-res from actual horizontal velocity
            dist_to_plane = max(cfg.plane_distance - float(pos[0]), 0.3)
            v_h = math.hypot(float(vel[0]), float(vel[1]))
            pred_px320 = v_h * dt_step * f_px / dist_to_plane

            # First-frame anchor readout vs ground-truth horizontal drift.
            au, av = env._anchor_off_px[0].tolist()
            anchor_d = float(env._anchor_disp_m[0])
            anchor_ncc = float(env._anchor_ncc[0])
            gt_h = math.hypot(float(drift[0]), float(drift[1]))

            print(
                f"[drift] {step:4d} t={t:6.2f} | "
                f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
                f"drift=({drift[0]:+.3f},{drift[1]:+.3f}) | "
                f"vel=({vel[0]:+.3f},{vel[1]:+.3f},{vel[2]:+.3f}) |v_h|={v_h:.3f} surge={v_surge:+.3f} | "
                f"cur=({float(cur[0]):+.2f},{float(cur[1]):+.2f}) | "
                f"flow={flow_rms_px320:5.2f}px pred={pred_px320:5.2f}px tex={tex_px} | "
                f"anchor=({au:+6.1f},{av:+6.1f})px d={anchor_d:6.3f}m ncc={anchor_ncc:5.3f} | gt_h={gt_h:.3f}m"
            )

        if term[0] or trunc[0]:
            print(f"[drift] terminated/truncated at step {step}")
            break

    pos = env._robot.data.root_pos_w[0]
    vel = env._robot.data.root_lin_vel_w[0]
    drift = pos - pos0[0]
    print(
        f"\n[drift] FINAL after {step+1} steps ({(step+1)*dt_step:.1f}s): "
        f"drift=({float(drift[0]):+.3f},{float(drift[1]):+.3f},{float(drift[2]):+.3f})m "
        f"vel=({float(vel[0]):+.3f},{float(vel[1]):+.3f},{float(vel[2]):+.3f})m/s "
        f"|v_h|={math.hypot(float(vel[0]), float(vel[1])):.3f}m/s"
    )
    env.close()


if __name__ == "__main__":
    main()
