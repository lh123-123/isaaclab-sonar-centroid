#!/usr/bin/env python3
"""Thruster budget of the level-attitude handover state.

Zero action, level_attitude=True, ocean current on. At three moments — the
frame-1 anchor capture, mid-episode steady state, and the late-episode current
peak — prints every thruster's command, thrust in N (T = T_max·u·|u|, forward
19.613 N / reverse 17.652 N, deadband 0.05), and its share of the datasheet
range, plus the grouped sums (pitch couple, heave, surge/sway).

  ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_thrust_budget.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 level-attitude thruster budget.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=224)
parser.add_argument(
    "--sweep",
    action="store_true",
    help="Current sweep: one env per speed, full opposing surge command, report steady-state "
    "ground velocity and horizontal-thruster usage. Answers 'can this hull hold station at "
    "speed X' instead of guessing it from episode statistics.",
)
parser.add_argument(
    "--speeds",
    type=str,
    default="0.20,0.30,0.38,0.45",
    help="Comma-separated current speeds for --sweep (m/s).",
)
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


def _sweep():
    """Current sweep: hold station against a fixed current with full opposing surge.

    The question this answers is physical, not policy: at what current does this hull run
    out of authority? Each env gets one current speed, is driven with full opposing surge
    (-0.75 m/s command, the actuator limit), and the last quarter of the run is averaged.
    A speed is "holdable" when the ground-frame velocity has been nulled to within 50 mm/s
    and the horizontal thrusters are not pinned at their limit.
    """
    speeds = [float(x) for x in args_cli.speeds.split(",") if x.strip()]
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = len(speeds)
    # No episode may end inside the sweep: a reset redraws the per-episode current and silently
    # replaces the speed this test set (observed: every env read back ~0.36 m/s regardless of
    # the requested 0.20-1.00, because the default episode length is shorter than the run).
    _ctrl_dt = float(cfg.sim.dt) * float(cfg.decimation)
    cfg.episode_length_s = (int(args_cli.steps) + 20) * _ctrl_dt
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    env.reset()
    n = env.num_envs
    dev = env.device
    # Override the per-episode DR draw: one fixed current per env, pointing +x.
    env._dr_cur_speed[:] = torch.tensor(speeds, device=dev, dtype=torch.float32)
    env._dr_cur_dir[:, 0] = 1.0
    env._dr_cur_dir[:, 1] = 0.0
    env._dr_cur_vz[:] = 0.0
    # Neutralise the per-run gains: without this each env gets its own thrust/drag draw, and
    # a 0.625 vs 0.387 m/s spread would be read as "current matters" when part of it is just
    # which unit the env happened to draw. The sweep is about authority, not unit luck.
    for _name in ("_dr_thrust_gain", "_dr_drag_gain"):
        if hasattr(env, _name):
            getattr(env, _name)[:] = 1.0
    # Command zero, not full stick: with act=0 the PID's target is zero *ground* velocity,
    # so the thrusters fight the current for us and the residual ground drift IS the answer
    # to "can this hull hold station here". A full or matched stick command only says how
    # fast it can fly, which is a different question.
    act = torch.zeros(n, 3, device=dev)
    k_meas = max(int(args_cli.steps) // 4, 30)
    v_sum = torch.zeros(n, 2)
    hpwm_sum = torch.zeros(n)
    hpwm_max = torch.zeros(n)
    for step in range(int(args_cli.steps)):
        env.step(act)
        if step >= int(args_cli.steps) - k_meas:
            v_sum += env._robot.data.root_lin_vel_w[:, :2].detach().cpu()
            # CAD order [V1, V2, V3, H1, H2, H3, H4, V4]: horizontals are indexes 3..6.
            hpwm = env._pwm[:, 3:7].abs().detach().cpu()
            hpwm_sum += hpwm.mean(dim=1)
            hpwm_max = torch.maximum(hpwm_max, hpwm.max(dim=1).values)
    v_mean = v_sum / float(k_meas)
    # Read the draw the physics actually uses (_dr_cur_*), not _ocean_current_vel: under DR
    # the latter is never written, so reporting it prints 0.000 and looks like no current.
    applied = env._dr_cur_speed.detach().cpu()
    print(
        "\n[budget] === current sweep (full opposing surge = -%.2f m/s command) ==="
        % (float(cfg.cmd_vel_scale)),
        flush=True,
    )
    print(
        "[budget] %6s %10s %10s %10s %10s  %s"
        % ("set", "applied|vx|", "ground vx", "H mean|pwm|", "H max|pwm|", "verdict"),
        flush=True,
    )
    for i, c in enumerate(speeds):
        app = float(applied[i].norm())
        gvx = float(v_mean[i, 0])
        hmean = float(hpwm_sum[i] / float(k_meas))
        hmax = float(hpwm_max[i])
        holdable = abs(gvx) <= 0.05 and hmean < 0.90
        print(
            "[budget] %6.2f %10.3f %10.3f %10.2f %10.2f  %s"
            % (c, app, gvx, hmean, hmax, "HOLDABLE" if holdable else "BEYOND ENVELOPE"),
            flush=True,
        )
    print(
        "[budget] note: ground vx > 0 means still being pushed downstream (authority short);"
        " vx < 0 means over-correcting upstream.",
        flush=True,
    )
    env.close()
    print("[budget] done", flush=True)


def main():
    if args_cli.sweep:
        _sweep()
        return
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
