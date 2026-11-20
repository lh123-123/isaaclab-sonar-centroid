#!/usr/bin/env python3
"""Trace the yaw chain end to end: action -> cmd_vel -> PID -> PWM -> body rate -> yaw.

The classical-baseline probe measured ~0.0004 rad of heading change for 20 steps of a
0.3 rad/s yaw command, three orders of magnitude below what that command should deliver.
Either the command never reaches the thrusters, or the yaw authority is negligible. This
prints every link so the break is visible instead of inferred.

    ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_yaw.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 yaw-chain diagnostic.")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--amp", type=float, default=0.6)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--every", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import torch

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)


def main() -> None:
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.dr_enable = False  # isolate the control chain from the randomization
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)

    print(
        f"[yaw] cmd_vel_scale={cfg.cmd_vel_scale}  "
        f"cmd_rate_limit={getattr(cfg, 'cmd_rate_limit', 'UNDEFINED')}",
        flush=True,
    )
    pid = env.pid
    for attr in ("yaw_pwm_cap", "kp_ang", "yaw_damp", "_yaw_scale", "thruster_layout"):
        print(f"[yaw] pid.{attr} = {getattr(pid, attr, 'n/a')}", flush=True)
    print(f"[yaw] action_space={env.action_space}", flush=True)

    env.reset()
    a = torch.zeros((env.num_envs, 3), device=env.device)
    a[:, 2] = float(args_cli.amp)

    yaw0 = euler_xyz_from_quat(env._robot.data.root_quat_w)[2]
    print(f"\n{'t':>4} {'cmd_vel_z':>10} {'pwm0':>8} {'pwm1':>8} {'w_z':>9} {'yaw':>9} {'dyaw':>9}", flush=True)
    for t in range(int(args_cli.steps)):
        env.step(a)
        if t % int(args_cli.every) == 0 or t == int(args_cli.steps) - 1:
            yaw = euler_xyz_from_quat(env._robot.data.root_quat_w)[2]
            wz = env._robot.data.root_ang_vel_b[:, 2]
            pwm = env._pwm
            print(
                f"{t:4d} {float(env._cmd_vel[:, 2].mean()):10.4f} "
                f"{float(pwm[:, 0].mean()):8.4f} {float(pwm[:, 1].mean()):8.4f} "
                f"{float(wz.mean()):9.4f} {float(yaw.mean()):9.4f} "
                f"{float((yaw - yaw0).mean()):9.4f}",
                flush=True,
            )

    dyaw = float((euler_xyz_from_quat(env._robot.data.root_quat_w)[2] - yaw0).mean())
    wz = float(env._robot.data.root_ang_vel_b[:, 2].mean())
    print(f"\n[yaw] verdict: {args_cli.steps} steps at amp={args_cli.amp} "
          f"(cmd {float(args_cli.amp) * float(cfg.cmd_vel_scale):.3f} rad/s) "
          f"-> dyaw={dyaw:+.4f} rad, final w_z={wz:+.4f} rad/s", flush=True)
    print(f"[yaw] expected if the loop tracks the command: "
          f"~{float(args_cli.amp) * float(cfg.cmd_vel_scale) * int(args_cli.steps) * float(cfg.decimation) * float(cfg.sim.dt):.3f} rad",
          flush=True)
    if abs(dyaw) < 0.05:
        print("[yaw] BROKEN: the yaw command produces no measurable heading change.", flush=True)
    else:
        print("[yaw] OK: yaw authority exists; the earlier near-zero probe was a "
              "measurement artifact, not a dead channel.", flush=True)


if __name__ == "__main__":
    main()
