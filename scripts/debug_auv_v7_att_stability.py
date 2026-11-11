#!/usr/bin/env python3
"""Which action channel kills attitude? 4 envs, constant actions, watch roll/pitch.

env0: zero action      (baseline, should hold trim)
env1: surge = +0.5
env2: sway  = +0.5
env3: yaw   = +0.5

Prints per-env attitude error vs trim (rad) every few steps and reports which
env hit att_limit=0.3 rad first. Run locally, headless:

  ./isaaclab.sh -p scripts/debug_auv_v7_att_stability.py --headless
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 attitude stability per action channel.")
parser.add_argument("--steps", type=int, default=225)
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument("--mode", type=str, default="random", choices=["random", "constant"],
                    help="random: SAC-like actions ~U(-1,1) every step (training condition)")
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

CHANNELS = ["zero", "surge=+0.5", "sway=+0.5", "yaw=+0.5"]


def main():
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = 4
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    n = env.num_envs
    dt_step = cfg.decimation * cfg.sim.dt

    trim_roll = float(cfg.trim_roll_rad)
    trim_pitch = float(cfg.trim_pitch_rad)

    obs, _ = env.reset()
    print(f"\n[att] 4 envs: {CHANNELS}")
    print(f"[att] trim: roll={trim_roll:+.4f} pitch={trim_pitch:+.4f} rad "
          f"att_limit={cfg.att_limit} rad  dt={dt_step:.4f}s  steps={args_cli.steps}")
    print("[att] cols per env: roll_err pitch_err att_err | dead@step\n")

    # constant actions per env
    acts = torch.zeros(n, 3)
    acts[1, 0] = 0.5  # surge
    acts[2, 1] = 0.5  # sway
    acts[3, 2] = 0.5  # yaw

    dead_at = [None] * n
    dead_cause = [None] * n
    alive = torch.ones(n, dtype=torch.bool)

    for step in range(args_cli.steps):
        if args_cli.mode == "random":
            acts = torch.rand(n, 3, device=env.device) * 2.0 - 1.0
        obs, rew, term, trunc, info = env.step(acts)

        pos = env._robot.data.root_pos_w
        quat = env._robot.data.root_quat_w
        # euler_xyz_from_quat returns (cos,sin)-half-angle components; use state obs instead
        st = obs["policy"]["state"]  # (N, 11)
        roll = st[:, 1]
        pitch = st[:, 2]
        roll_err = (roll - trim_roll).cpu().numpy()
        pitch_err = (pitch - trim_pitch).cpu().numpy()
        att_err = np.sqrt(roll_err**2 + pitch_err**2)

        for i in range(n):
            if alive[i] and (term[i] or trunc[i]):
                alive[i] = False
                dead_at[i] = step
                # read the env's own recorded cause bitmask (tex/depth/att/timeout)
                cause = "?"
                if hasattr(env, "_ep_cause") and env._ep_cause is not None:
                    c = env._ep_cause[i].cpu().tolist()
                    names = ["tex_lost", "depth", "att", "timeout"]
                    hits = [names[k] for k in range(4) if c[k]]
                    cause = "+".join(hits) if hits else "none"
                dead_cause[i] = cause
                print(f"[att] env{i} ({CHANNELS[i]:>12}) DIED at step {step:3d} "
                      f"(t={step*dt_step:5.2f}s) cause={cause} "
                      f"roll_err={roll_err[i]:+.3f} pitch_err={pitch_err[i]:+.3f}", flush=True)

        if step % args_cli.print_every == 0 or step == args_cli.steps - 1:
            cells = []
            for i in range(n):
                mark = " " if alive[i] else "x"
                cells.append(f"{CHANNELS[i][:6]}:{mark} r{roll_err[i]:+.2f} p{pitch_err[i]:+.2f} e{att_err[i]:.2f}")
            print(f"[att] step={step:3d} t={step*dt_step:5.2f} | " + " | ".join(cells), flush=True)

        if not alive.any():
            print(f"[att] all dead at step {step}")
            break

    print("\n[att] ---- summary ----")
    for i in range(n):
        d = dead_at[i]
        print(f"[att] env{i} {CHANNELS[i]:>12}: "
              + (f"survived full episode" if d is None else f"died at step {d} (t={d*dt_step:.2f}s) cause={dead_cause[i]}"))
    env.close()


if __name__ == "__main__":
    main()
