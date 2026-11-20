"""Probe the GS-ROV thruster path: depth PID -> wrench -> 8 PWM -> body force.

Usage:
    python probe_auv_v7_gsrov_thrusters.py --steps 6
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GS-ROV thruster path probe.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--use-real-usd", action="store_true", default=True)
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


def main() -> None:
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)

    print(f"\n[probe] layout={env.thruster_layout} num_thrusters={env.num_thrusters}", flush=True)
    print(f"[probe] pid.num_thrusters={env.pid.num_thrusters}", flush=True)
    print(
        f"[probe] surge_scale={env.pid._surge_scale:.4f} yaw_scale={env.pid._yaw_scale:.4f}",
        flush=True,
    )
    print(f"[probe] max_fwd={env.cfg.thruster_max_thrust_fwd_n} N", flush=True)

    env.reset()
    act = torch.zeros(env.num_envs, 3, device=env.device)
    torch.set_printoptions(precision=4, sci_mode=False)

    for step in range(args_cli.steps):
        env.step(act)
        z = env._robot.data.root_pos_w[:, 2]
        vz = env._robot.data.root_lin_vel_w[:, 2]
        depth_error = env.pid.depth_target - z
        fz = env.pid.kp_depth * depth_error - env.pid.kd_depth * vz
        print(f"\n[probe] step {step}", flush=True)
        print(f"  cmd_vel   = {env._cmd_vel[0].tolist()}", flush=True)
        print(f"  z={float(z[0]):.4f}  vz={float(vz[0]):.4f}  depth_target={float(env.pid.depth_target[0]):.4f}", flush=True)
        print(f"  depth_err(target-z)={float(depth_error[0]):.4f}  fz_wrench={float(fz[0]):.4f}", flush=True)
        print(f"  pwm       = {env._pwm[0].tolist()}", flush=True)
        print(f"  body_force= {env._thrust[0, 0].tolist()}", flush=True)
        print(f"  body_torq = {env._moment[0, 0].tolist()}", flush=True)

    env.close()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
