#!/usr/bin/env python3
"""Thruster budget of the level-attitude handover state.

Zero action, level_attitude=True, ocean current on. At three moments — the
frame-1 anchor capture, mid-episode steady state, and the late-episode current
peak — prints every thruster's command, thrust in N (T = T_max·u·|u|, forward
19.613 N / reverse 17.652 N, deadband 0.05), and its share of the datasheet
range, plus the grouped sums (pitch couple, heave, surge/sway).

  ./isaaclab.sh -p scripts/debug_auv_v7_thrust_budget.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 level-attitude thruster budget.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=224)
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
from isaaclab_tasks.direct.isaac_auv_tracking.thruster_dynamics import GSROV_THRUSTERS

# CAD order [V1, V2, V3, H1, H2, H3, H4, V4]; verticals are indexes 0, 1, 2, 7.
_NAMES = [t[0] for t in GSROV_THRUSTERS]
_OFFSETS = np.array([t[1] for t in GSROV_THRUSTERS])
_AXES = np.array([t[2] for t in GSROV_THRUSTERS])


def _thruster_n(pwm: np.ndarray, fwd: float, rev: float, deadband: float) -> np.ndarray:
    out = np.where(pwm >= 0.0, fwd, rev) * pwm * np.abs(pwm)
    return np.where(np.abs(pwm) < deadband, 0.0, out)


def _report(env, cfg, step: int) -> None:
    pwm = env._pwm[0].detach().cpu().numpy()
    deadband = float(getattr(cfg, "thruster_deadband", 0.05))
    thrust = _thruster_n(pwm, env._thrust_fwd_n, env._thrust_rev_n, deadband)
    # force vectors in N (body frame)
    forces = thrust[:, None] * _AXES
    torques = np.cross(_OFFSETS, forces)
    pitch_couple = float(torques[:, 1].sum())
    heave = float(forces[:, 2].sum())
    surge = float(forces[:, 0].sum())
    sway = float(forces[:, 1].sum())
    cur = env._ocean_current_vel[0].detach().cpu().numpy()
    cur_s = float(np.linalg.norm(cur[:2]))
    print(f"\n[budget] step={step}  current={cur_s:.2f} m/s  (fwd max {env._thrust_fwd_n:.1f} N, rev max {env._thrust_rev_n:.1f} N)", flush=True)
    print("[budget]  thr  pwm    thrust    % of range", flush=True)
    for name, u, f in zip(_NAMES, pwm, thrust):
        cap = env._thrust_fwd_n if u >= 0 else env._thrust_rev_n
        pct = abs(f) / cap * 100.0 if cap > 0 else 0.0
        print(f"[budget]  {name:>3}  {u:+.3f}  {f:+7.2f} N   {pct:5.1f}%", flush=True)
    print(
        f"[budget]  sums: pitch couple {pitch_couple:+.2f} N·m (level needs +5.07 against buoyancy), "
        f"heave {heave:+.2f} N, surge {surge:+.2f} N, sway {sway:+.2f} N",
        flush=True,
    )


def main():
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    n = env.num_envs
    print(
        f"[budget] level_attitude={cfg.level_attitude} deadband={getattr(cfg, 'thruster_deadband', 0.05)}",
        flush=True,
    )
    env.reset()
    marks = {1: "frame-1 anchor capture", 60: "mid-episode steady state", 200: "late-episode current peak"}
    for step in range(args_cli.steps):
        env.step(torch.zeros(n, 3))
        if step in marks:
            print(f"\n[budget] === {marks[step]} ===", flush=True)
            _report(env, cfg, step)
    print(f"\n[budget] === final step ({args_cli.steps - 1}) ===", flush=True)
    _report(env, cfg, args_cli.steps - 1)
    env.close()
    print("[budget] done", flush=True)


if __name__ == "__main__":
    main()
