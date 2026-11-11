#!/usr/bin/env python3
"""Can the radial/lateral flow split replace the ground-truth distance reward?

Does not change the reward or the observation. One 15 s zero-action episode, then
a short surge command that tries to stop the leak. Prints the anchor term the
training reward actually uses, the drift-counter term, and the term we would
get from the diagnostic flow split instead of the spawn distance.

  ./isaaclab.sh -p scripts/debug_auv_v7_expand_vs_anchor.py --headless
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 expansion reward vs GT anchor.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--hold_steps", type=int, default=224)
parser.add_argument("--cancel_steps", type=int, default=80)
parser.add_argument("--surge", type=float, default=-0.15, help="cancel-phase surge command")
parser.add_argument("--sigma_v", type=float, default=0.05, help="m/s, expansion reward width")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)


def _body_horizontal(env):
    vel_w = env._robot.data.root_lin_vel_w[0]
    quat = env._robot.data.root_quat_w[0]
    vel_b = quat_apply(quat_conjugate(quat.unsqueeze(0)), vel_w.unsqueeze(0))[0]
    return float(vel_b[0]), float(vel_b[1])


def _horiz_disp(env, origin):
    pos = env._robot.data.root_pos_w[0, :2] - origin
    return float(torch.linalg.norm(pos)), float(pos[0]), float(pos[1])


def main():
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    n = env.num_envs
    dt = float(cfg.decimation) * float(cfg.sim.dt)
    z_plane = float(cfg.plane_distance)
    hfov = float(cfg.cam_hfov_deg)
    fw = float(cfg.flow_render_width)
    f_px = fw / (2.0 * math.tan(math.radians(hfov) / 2.0))
    sigma_v = float(args_cli.sigma_v)
    w = 0.3

    env.reset()
    origin = env.scene.env_origins[0, :2].detach().clone()
    print(
        f"\n[expand] dt={dt:.4f}s Z={z_plane:.2f}m f_px={f_px:.1f} "
        f"sigma_v={sigma_v:.3f} m/s  hold={args_cli.hold_steps} cancel={args_cli.cancel_steps}",
        flush=True,
    )
    print(
        "[expand] step | surge | d_xy | anchor | drift | expand | v_ax v_lat",
        flush=True,
    )

    rows = []
    total = args_cli.hold_steps + args_cli.cancel_steps
    for step in range(total):
        act = torch.zeros(n, 4)
        if step >= args_cli.hold_steps:
            act[:, 0] = args_cli.surge
        env.step(act)
        surge, _sway = _body_horizontal(env)
        d_xy, dx, dy = _horiz_disp(env, origin)
        anchor = math.exp(-(d_xy ** 2) / (float(cfg.sigma_anchor_m) ** 2))
        drift_mag = float(env._drift_int[0].norm())
        drift = math.exp(-(drift_mag ** 2) / (float(cfg.sigma_drift) ** 2))
        a_px = float(env._flow_uniform[0, 0])
        k = float(env._flow_expand[0])
        v_axial = k * z_plane / dt
        v_lat = a_px / max(f_px, 1.0) * z_plane / dt
        v_h = math.hypot(v_axial, v_lat)
        expand = math.exp(-(v_h ** 2) / (sigma_v ** 2))
        rows.append((surge, d_xy, dx, dy, anchor, drift, expand, v_axial, v_lat, drift_mag))
        if step % 30 == 0 or step in (args_cli.hold_steps - 1, total - 1):
            print(
                f"[expand] {step:4d} | {surge:+.3f} | {d_xy:.3f} | {anchor:.3f} | "
                f"{drift:.3f} | {expand:.3f} | {v_axial:+.3f} {v_lat:+.3f}",
                flush=True,
            )

    hold = rows[: args_cli.hold_steps]
    cancel = rows[args_cli.hold_steps :]

    def mean(idx, src):
        return sum(r[idx] for r in src) / len(src)

    # Points over the hold episode, weight 0.3, same budget as the anchor (67.2).
    anchor_pts = w * sum(r[4] for r in hold)
    drift_pts = w * sum(r[5] for r in hold)
    expand_pts = w * sum(r[6] for r in hold)
    full = w * len(hold)
    end = hold[-1]
    print(
        f"\n[expand] HOLD {len(hold)} steps  end d_xy={end[1]:.3f} m "
        f"({end[2]:+.3f},{end[3]:+.3f})  surge={mean(0, hold):+.3f} m/s",
        flush=True,
    )
    print(
        f"[expand] HOLD points / {full:.1f}: anchor={anchor_pts:.1f}  "
        f"drift_counter={drift_pts:.1f}  expand={expand_pts:.1f}",
        flush=True,
    )
    print(
        f"[expand] HOLD mean terms: anchor={mean(4, hold):.3f}  "
        f"drift={mean(5, hold):.3f}  expand={mean(6, hold):.3f}  "
        f"v_axial={mean(7, hold):+.3f}  v_lat={mean(8, hold):+.3f}",
        flush=True,
    )
    if cancel:
        print(
            f"[expand] CANCEL mean v_axial={mean(7, cancel):+.3f}  "
            f"expand={mean(6, cancel):.3f}  end d_xy={cancel[-1][1]:.3f} m  "
            f"anchor={cancel[-1][4]:.3f}",
            flush=True,
        )

    # Feasible as a replacement if, during the leak, expansion loses points the
    # counter does not, and it moves when the surge command changes the speed.
    sees_leak = expand_pts < full * 0.75 and drift_pts > full * 0.85 and end[1] > 0.4
    tracks = mean(0, hold) * mean(7, hold) > 0.0 and abs(mean(7, hold)) > 0.02
    moves = bool(cancel) and abs(mean(7, cancel)) < abs(mean(7, hold)) * 0.7
    print(
        f"[expand] replaces the blind counter: {'PASS' if sees_leak else 'FAIL'}; "
        f"same sign as surge: {'PASS' if tracks else 'FAIL'}; "
        f"cancel moves v_axial: {'PASS' if moves else 'FAIL'}",
        flush=True,
    )
    env.close()
    print("[expand] done", flush=True)


if __name__ == "__main__":
    main()
