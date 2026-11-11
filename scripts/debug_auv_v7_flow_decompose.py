#!/usr/bin/env python3
"""Local check: does radial flow see the current-aligned forward leak?

The signed-mean flow cancels on axial motion (the picture expands). This script
does not change the reward. It only reads the diagnostic fit added beside the
existing Farneback field.

Phase 1, zero action: expect body surge around +5 cm/s, signed-mean flow near 0,
and v_axial from the radial coefficient the same sign and similar size.

Phase 2, surge command against the leak: both body surge and v_axial should fall.

  ./isaaclab.sh -p scripts/debug_auv_v7_flow_decompose.py --headless --steps 160
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 translational-flow split check.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--phase2_step", type=int, default=90)
parser.add_argument("--surge", type=float, default=-0.25, help="phase-2 surge command")
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


def main():
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    n = env.num_envs
    dt = float(cfg.decimation) * float(cfg.sim.dt)
    z_plane = float(cfg.plane_distance)
    hfov = float(cfg.cam_hfov_deg)
    fw = float(cfg.flow_render_width)
    f_px = fw / (2.0 * torch.tan(torch.tensor(hfov * 3.14159265 / 360.0)).item())

    env.reset()
    print(
        f"\n[decomp] dt={dt:.4f}s Z={z_plane:.2f}m f_px={f_px:.1f} "
        f"phase2 surge={args_cli.surge} from step {args_cli.phase2_step}",
        flush=True,
    )
    print(
        "[decomp] step | surge sway | current | mean_u | v_lat v_axial | k",
        flush=True,
    )
    p1_surge, p1_axial, p1_mean = [], [], []
    p2_axial = []
    for step in range(args_cli.steps):
        act = torch.zeros(n, 4)
        if step >= args_cli.phase2_step:
            act[:, 0] = args_cli.surge
        env.step(act)
        surge, sway = _body_horizontal(env)
        cur = env._ocean_current_vel[0]
        cur_s = float(torch.linalg.norm(cur[:2]))
        # Signed mean is what the counter integrates (normalized by width/height).
        # Recover px/frame roughly via the uniform fit, which is already in pixels.
        a_px = float(env._flow_uniform[0, 0])
        b_px = float(env._flow_uniform[0, 1])
        k = float(env._flow_expand[0])
        v_axial = k * z_plane / dt
        v_lat = a_px / max(f_px, 1.0) * z_plane / dt
        mean_u = float(env._drift_int[0, 0])  # not the instantaneous mean
        # Instantaneous normalized mean is not stored. Use uniform px / fw as the
        # same quantity the counter adds each step.
        mean_inst = a_px / fw
        if step % 10 == 0 or step in (args_cli.phase2_step - 1, args_cli.steps - 1):
            print(
                f"[decomp] {step:4d} | {surge:+.3f} {sway:+.3f} | {cur_s:.3f} | "
                f"{mean_inst:+.5f} | {v_lat:+.3f} {v_axial:+.3f} | {k:+.5f}  "
                f"(dv_px={b_px:+.3f})",
                flush=True,
            )
        if 30 <= step < args_cli.phase2_step:
            p1_surge.append(surge)
            p1_axial.append(v_axial)
            p1_mean.append(abs(mean_inst))
        if step >= args_cli.phase2_step + 40:
            p2_axial.append(v_axial)

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    s1, a1, m1 = avg(p1_surge), avg(p1_axial), avg(p1_mean)
    a2 = avg(p2_axial)
    print(
        f"\n[decomp] phase1 surge={s1:+.3f} m/s  v_axial={a1:+.3f} m/s  |mean_u|={m1:.5f}",
        flush=True,
    )
    print(f"[decomp] phase2 v_axial={a2:+.3f} m/s", flush=True)
    # Axial estimate should track the forward leak; the signed mean should stay tiny.
    see_forward = abs(a1) > 0.02 and abs(a1) > 5.0 * max(m1, 1e-6)
    same_sign = s1 * a1 > 0.0
    responds = abs(a2) < abs(a1) * 0.7 or a2 * a1 < 0.0
    print(
        f"[decomp] axial sees forward leak: {'PASS' if see_forward else 'FAIL'}; "
        f"same sign as surge: {'PASS' if same_sign else 'FAIL'}; "
        f"phase2 moves it: {'PASS' if responds else 'FAIL'}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
