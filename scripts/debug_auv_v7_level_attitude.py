#!/usr/bin/env python3
"""Does the GS-ROV hold LEVEL attitude with thrusters, from frame 0?

level_attitude=True spawns level, aims the attitude PID at roll=pitch=0, and
pre-charges the attitude integrator with the buoyancy couple (~5.1 N·m). This
script runs a zero-action episode (ocean current on) and checks:

  1. no nose-up transient at the start (pre-charge works, frame 0 is the handover
     state),
  2. pitch/roll stay within ~3 deg for the whole episode,
  3. depth holds,
  4. the vertical thrusters carry a sustained, nonzero command ("the thrusters
     are moving") rather than idling at the free-floating trim.

  ./isaaclab.sh -p scripts/debug_auv_v7_level_attitude.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 level-attitude hold check.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=224)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import numpy as np
import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_conjugate

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)


def _deg(rad: float) -> float:
    return float(np.degrees(rad))


def main():
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    n = env.num_envs
    print(
        f"[level] level_attitude={cfg.level_attitude} att_ki={cfg.att_ki} "
        f"heave_wrench_max={cfg.heave_wrench_max} pre_charge={getattr(env, '_level_pitch_moment', 0.0):.3f} wrench-units "
        f"(1 unit = one thruster max force)",
        flush=True,
    )
    print("[level] step | roll pitch (deg) | depth_err (m) | vert pwm (V1,V2,V3,V4) | mean|pwm|", flush=True)

    env.reset()
    rows = []
    for step in range(args_cli.steps):
        env.step(torch.zeros(n, 3))
        roll, pitch, _ = euler_xyz_from_quat(env._robot.data.root_quat_w[0:1])
        depth_err = float(env._robot.data.root_pos_w[0, 2] - float(cfg.starting_depth))
        pwm = env._pwm[0].detach().cpu().numpy()
        # CAD order [V1,V2,V3,H1,H2,H3,H4,V4] -> verticals are 0,1,2,7
        verts = pwm[[0, 1, 2, 7]]
        horiz = pwm[[3, 4, 5, 6]]
        rows.append((_deg(float(roll)), _deg(float(pitch)), depth_err, verts.copy(), float(np.abs(horiz).mean())))
        if step < 24 or step % 15 == 0 or step == args_cli.steps - 1:
            q = env._robot.data.root_quat_w[0].detach().cpu()
            err = float(q[2])  # sin(pitch/2) against level target 0
            att_sum = env.pid.att_error_sum[0].detach().cpu().numpy()
            ang_w = env._robot.data.root_ang_vel_w[0:1]
            quat_c = env._robot.data.root_quat_w[0:1]
            ang_b = quat_apply(quat_conjugate(quat_c), ang_w)[0]
            r = rows[-1]
            print(
                f"[level] {step:4d} | {r[0]:+6.2f} {r[1]:+6.2f} | {r[2]:+.3f} | "
                f"({r[3][0]:+.2f},{r[3][1]:+.2f},{r[3][2]:+.2f},{r[3][3]:+.2f}) | {r[4]:.3f} | "
                f"qy={float(q[2]):+.4f} err={err:+.4f} sum=({att_sum[0]:+.2f},{att_sum[1]:+.2f}) "
                f"wy_b={float(ang_b[1]):+.3f}",
                flush=True,
            )

    # The last sample is the post-timeout reset (pitch and pwm back to the spawn).
    body = rows[:-1]
    tail = body[-50:]
    max_abs_pitch_tail = max(abs(r[1]) for r in tail)
    mean_abs_roll = sum(abs(r[0]) for r in tail) / len(tail)
    mean_abs_pitch = sum(abs(r[1]) for r in tail) / len(tail)
    max_abs_pitch_all = max(abs(r[1]) for r in body)
    depth_end = abs(body[-1][2])
    mean_abs_vert = float(np.mean([np.abs(r[3]) for r in tail]))

    print(
        f"\n[level] whole episode max |pitch| = {max_abs_pitch_all:.2f} deg "
        f"(no nose-up transient expected)",
        flush=True,
    )
    print(
        f"[level] last 50: max |pitch| = {max_abs_pitch_tail:.2f} deg, mean |roll| = {mean_abs_roll:.2f} deg, "
        f"mean |pitch| = {mean_abs_pitch:.2f} deg, depth err = {depth_end:.3f} m, "
        f"mean |vert pwm| = {mean_abs_vert:.3f}",
        flush=True,
    )
    no_transient = max_abs_pitch_all < 1.0
    holds = max_abs_pitch_tail < 3.0 and mean_abs_roll < 3.0 and mean_abs_pitch < 2.0
    depth_ok = depth_end < 0.15
    thrusters_on = mean_abs_vert > 0.05
    print(
        f"[level] frame0 is the handover state: {'PASS' if no_transient else 'FAIL'}; "
        f"level hold (last 50): {'PASS' if holds else 'FAIL'}; "
        f"depth: {'PASS' if depth_ok else 'FAIL'}; "
        f"thrusters working: {'PASS' if thrusters_on else 'FAIL'}",
        flush=True,
    )
    env.close()
    print("[level] done", flush=True)


if __name__ == "__main__":
    main()
