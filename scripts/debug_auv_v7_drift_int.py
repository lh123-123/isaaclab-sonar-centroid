#!/usr/bin/env python3
"""Verify the leaky flow-integrator exposes the slow drift (and calibrate sigma_drift).

Phase 1 (zero action): the ROV leaks ~5 cm/s downstream; drift_int should climb
monotonically toward its steady state ≈ mean_flow x tau/dt (expect |drift| ~0.04-0.06
within ~2x tau). The old flow_rms_n stays flat at ~0.005 — that contrast IS the fix.

Phase 2 (constant surge against the leak, e.g. -0.15): drift_int should fall back
toward 0 as the net leak vanishes, proving the policy has a learnable gradient.

Phase 3 (constant yaw, e.g. +0.8): the "spin-laundering" exploit check. The 287k
model drifted 2.8 m/episode while yawing steadily — the body-frame flow signature
rotated and the OLD body-frame integrator cancelled to ~0. With the de-rotated
accumulator, spinning must NOT collapse |drift|; the anchor displacement keeps
growing and is printed alongside.

  ./isaaclab.sh -p scripts/debug_auv_v7_drift_int.py --headless --steps 220
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 drift integrator verification.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=220)
parser.add_argument("--phase2_step", type=int, default=90, help="switch to constant surge here")
parser.add_argument("--phase3_step", type=int, default=170, help="switch to constant yaw here")
parser.add_argument("--surge", type=float, default=-0.15, help="phase-2 constant surge action")
parser.add_argument("--yaw", type=float, default=0.8, help="phase-3 constant yaw action")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import numpy as np
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
    tau = cfg.drift_tau_s

    obs, _ = env.reset()
    print(
        f"\n[drift-int] tau={tau}s leak/frame={env._drift_leak:.4f} sigma_drift={cfg.sigma_drift} "
        f"phase2: surge={args_cli.surge} from step {args_cli.phase2_step}; "
        f"phase3: yaw={args_cli.yaw} from step {args_cli.phase3_step}"
    )
    print("[drift-int] step | drift_u drift_v |drift| | flow_rms_n | drift_term | anchor_d anchor_t | rew")
    samples_p1 = []
    samples_p3 = []
    p2_end_drift = (0.0, 0.0)
    for step in range(args_cli.steps):
        act = torch.zeros(n, 4)
        if step >= args_cli.phase2_step:
            act[:, 0] = args_cli.surge
        if step >= args_cli.phase3_step:
            act[:, 0] = 0.0
            act[:, 2] = args_cli.yaw
        obs, rew, term, trunc, info = env.step(act)
        st = obs["policy"]["state"][0]
        # state[11:13] were the leaky drift integral until the first-frame anchor landed
        # (they are now the anchor offset) — read the integrator from the env buffer.
        drift_u, drift_v = float(env._drift_int[0, 0]), float(env._drift_int[0, 1])
        drift_mag = float(np.hypot(drift_u, drift_v))
        flow_n = float(st[8])
        present = float(st[9]) > 0.5
        drift_term = float(np.exp(-(drift_mag**2) / cfg.sigma_drift**2)) if present else 0.0
        disp_xy = env._robot.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2]
        anchor_d = float(torch.norm(disp_xy))
        anchor_t = float(np.exp(-(anchor_d**2) / cfg.sigma_anchor_m**2))
        if step % 10 == 0 or step in (args_cli.phase2_step - 1, args_cli.phase3_step - 1, args_cli.steps - 1):
            print(
                f"[drift-int] {step:4d} | {drift_u:+.4f} {drift_v:+.4f} {drift_mag:.4f} | "
                f"{flow_n:.4f} | {drift_term:.3f} | {anchor_d:.3f}m {anchor_t:.3f} | {float(rew.mean()):.3f}",
                flush=True,
            )
        if 10 <= step < args_cli.phase2_step:
            samples_p1.append(drift_mag)
        if step == args_cli.phase3_step - 1:
            p2_end_drift = (drift_u, drift_v)
        if step >= args_cli.phase3_step + 10:
            samples_p3.append(drift_mag)

    steady = max(samples_p1) if samples_p1 else 0.0
    p3_mag = max(samples_p3) if samples_p3 else 0.0
    print(
        f"\n[drift-int] phase-1 max |drift| = {steady:.4f} "
        f"(target: reaches O(sigma_drift={cfg.sigma_drift}) within ~{2*tau:.0f}s)"
    )
    print(f"[drift-int] phase-2 end drift = ({p2_end_drift[0]:+.4f}, {p2_end_drift[1]:+.4f})")
    print(f"[drift-int] phase-3 (spin) max |drift| = {p3_mag:.4f} (must NOT collapse to ~0)")
    verdict1 = steady > 0.5 * cfg.sigma_drift
    verdict2 = abs(p2_end_drift[0]) < 0.5 * steady or abs(p2_end_drift[1]) < 0.5 * steady
    verdict3 = p3_mag > 0.5 * max(steady, 1e-6)
    print(
        f"[drift-int] climb visible: {'PASS' if verdict1 else 'FAIL'}; "
        f"phase-2 recovers: {'PASS' if verdict2 else 'FAIL'}; "
        f"spin-laundering resisted: {'PASS' if verdict3 else 'FAIL'}"
    )
    env.close()


if __name__ == "__main__":
    main()
