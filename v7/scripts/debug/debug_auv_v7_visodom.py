#!/usr/bin/env python3
"""Position from INTEGRATED consecutive-frame flow (visual odometry).

The previous probe showed frame-1-vs-current Farneback dies once the ROV has
moved (locks onto wrong peaks, a ≈ 0 at 0.4 m). The signal that keeps working
at ANY distance is the consecutive-frame flow: it measures ground-relative
velocity. Integrating it — de-rotated by yaw so heading changes don't cancel
it, no leak so it doesn't forget — gives a body-frame displacement estimate.
That is the position cue the privileged GT anchor term used to provide.

Accumulators (all from buffers the env already updates each step):
  _flow_uniform  (a_px, b_px)  uniform component of consecutive flow, flow-res px
  _flow_expand    k             radial expansion per frame (ΔZ / Z)

  sway estimate:   acc_u = Σ R(dyaw) · a_px      → px, /f_px · Z → m
  surge estimate:  disp_x += k · Z,  Z ← Z/(1+k) → m (scale tracked, not fixed)

Episode 1 (cal): sway drive, stop, surge drive, stop — fit acc vs GT.
Episode 2 (visual): drive away 0.5 m, then P-servo home on the estimate only.
Episode 3 (zero): same drive, then nothing — the comparison.

    ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_visodom.py --headless
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 consecutive-flow odometry probe.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import numpy as np
import torch

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)

_KP = 0.8
_DEAD_M = 0.05
_V_MAX = 0.5
_ABORT_M = 1.5
_DRIVE = 16
_HOLD_STEPS = 170
f_px = 1.0  # set in main()


class Odo:
    """Non-leaky de-rotated integral of the consecutive-flow uniform component."""

    def __init__(self, z0: float):
        self.z = z0
        self.u = 0.0  # px, flow-res, de-rotated
        self.v = 0.0
        self.x = 0.0  # m, along optical axis
        self.yaw_prev = None

    def step(self, yaw: float, a_px: float, b_px: float, k: float) -> None:
        if self.yaw_prev is not None:
            dyaw = yaw - self.yaw_prev
            dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
            c, s = math.cos(dyaw), math.sin(dyaw)
            u0, v0 = self.u, self.v
            self.u = u0 * c - v0 * s
            self.v = u0 * s + v0 * c
        self.yaw_prev = yaw
        self.u += a_px
        self.v += b_px
        self.x += k * self.z
        if k > -0.9:
            self.z /= 1.0 + k


def _through_origin(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    den = float(np.dot(x, x))
    if den < 1e-9:
        return 0.0, 0.0
    slope = float(np.dot(x, y) / den)
    if x.size < 3 or float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
        return slope, 0.0
    return slope, float(np.corrcoef(x, y)[0, 1])


def _body(pos, pos0, yaw):
    dx = float(pos[0] - pos0[0])
    dy = float(pos[1] - pos0[1])
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _read(env):
    origin = env.scene.env_origins[0]
    pos = (env._robot.data.root_pos_w[0] - origin).detach()
    _roll, pitch, yaw = euler_xyz_from_quat(env._robot.data.root_quat_w)
    cur = float(torch.linalg.norm(env._ocean_current_vel[0, :2]))
    return pos, float(yaw[0]), float(pitch[0]), cur


def _command(env, surge, sway):
    scale = float(env.cfg.cmd_vel_scale)
    raw = [surge / scale, sway / scale, 0.0]
    return torch.tensor([raw], dtype=torch.float32, device=env.device).clamp(-1.0, 1.0)


def _print_row(tag, step, gs, gy, odo, env, cur, pitch):
    rms = float(env._flow_rms[0])
    y_m = odo.u * odo.z / f_px
    print(
        f"[visodom] {tag:7} step={step:3d}  gt=({gs:+.3f},{gy:+.3f}) m  "
        f"odo=({odo.x:+.3f},{y_m:+.3f}) m  Z={odo.z:.2f}  "
        f"flow_rms={rms:.2f}px  cur={cur:.2f}  pitch={math.degrees(pitch):+.2f}°",
        flush=True,
    )


def run_episode(env, kind, metres, steps, z0):
    env.reset()
    odo = Odo(z0)
    pos0 = None
    prev_ep = -1
    sway_gt, sway_acc = [], []
    surge_gt, surge_acc = [], []
    switched = None
    last = None
    aborted = ""
    cmd = (0.0, 0.0)
    for step in range(steps):
        if kind == "cal":
            if step < 8:
                vs, vy = 0.0, 0.0
            elif step < 23:
                vs, vy = 0.0, 0.5
            elif step < 31:
                vs, vy = 0.0, 0.0
            elif step < 46:
                vs, vy = 0.5, 0.0
            else:
                vs, vy = 0.0, 0.0
        elif step < _DRIVE:
            vs, vy = 0.5, 0.5
        elif kind == "visual":
            vs, vy = cmd
        else:
            vs, vy = 0.0, 0.0
        env.step(_command(env, vs, vy))
        ep = int(env.episode_length_buf[0])
        if prev_ep >= 0 and ep < prev_ep:
            aborted = "episode reset"
            break
        prev_ep = ep
        pos, yaw, pitch, cur = _read(env)
        odo.step(
            yaw,
            float(env._flow_uniform[0, 0]),
            float(env._flow_uniform[0, 1]),
            float(env._flow_expand[0]),
        )
        if pos0 is None:
            pos0 = pos.clone()
            continue
        gs, gy = _body(pos, pos0, yaw)
        if kind == "visual" and step >= _DRIVE - 1:
            x_m, y_m = metres(odo)
            if abs(x_m) < _DEAD_M:
                x_m = 0.0
            if abs(y_m) < _DEAD_M:
                y_m = 0.0
            cmd = (
                float(np.clip(-_KP * x_m, -_V_MAX, _V_MAX)),
                float(np.clip(-_KP * y_m, -_V_MAX, _V_MAX)),
            )
        if kind == "cal" and 8 <= step < 31:
            sway_gt.append(gy)
            sway_acc.append(odo.u)
        if kind == "cal" and 31 <= step < 54:
            surge_gt.append(gs)
            surge_acc.append(odo.x)
        if kind != "cal" and step == _DRIVE:
            switched = (gs, gy)
        last = (gs, gy, pitch, cur, step)
        if step % 15 == 0 or step in (22, 30, 45, 53, _DRIVE, steps - 1):
            _print_row(kind, step, gs, gy, odo, env, cur, pitch)
        if math.hypot(gs, gy) > _ABORT_M:
            aborted = f"|gt| {math.hypot(gs, gy):.2f} m"
            _print_row(kind, step, gs, gy, odo, env, cur, pitch)
            break
        if abs(pitch) > 0.25:
            aborted = f"pitch {math.degrees(pitch):.1f} deg"
            break
    return {
        "sway_gt": np.asarray(sway_gt),
        "sway_acc": np.asarray(sway_acc),
        "surge_gt": np.asarray(surge_gt),
        "surge_acc": np.asarray(surge_acc),
        "switched": switched,
        "last": last,
        "aborted": aborted,
    }


def main():
    global f_px
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    z0 = float(cfg.plane_distance)
    f_px = float(env._anchor_f_px)

    def metres_geom(odo: Odo):
        return odo.x, odo.u * odo.z / f_px

    cal = run_episode(env, "cal", metres_geom, 60, z0)
    slope_y, corr_y = _through_origin(cal["sway_gt"], cal["sway_acc"])
    slope_x, corr_x = _through_origin(cal["surge_gt"], cal["surge_acc"])
    print(
        f"[visodom] sway: acc_u = {slope_y:+.1f} px/m * gt  corr={corr_y:+.3f}  "
        f"(geometry {f_px / z0:+.1f})",
        flush=True,
    )
    print(f"[visodom] surge: disp_x = {slope_x:+.3f} * gt  corr={corr_x:+.3f}  (ideal +1.0)", flush=True)

    use_y = abs(corr_y) >= 0.85 and abs(slope_y) > 10.0
    use_x = abs(corr_x) >= 0.85 and abs(slope_x) > 0.3
    slope_y_eff = slope_y if use_y else f_px / z0
    slope_x_eff = slope_x if use_x else 1.0

    def metres(odo: Odo):
        return odo.x / slope_x_eff, odo.u / slope_y_eff

    visual = run_episode(env, "visual", metres, _HOLD_STEPS, z0)
    zero = run_episode(env, "zero", metres, _HOLD_STEPS, z0)

    def _end(res):
        last = res["last"]
        sw = res["switched"]
        if last is None:
            return "no sample"
        gs, gy = last[0], last[1]
        extra = f"  aborted: {res['aborted']}" if res["aborted"] else ""
        return (
            f"switch {math.hypot(sw[0], sw[1]):.3f} m -> end {math.hypot(gs, gy):.3f} m "
            f"at step {last[4]}{extra}"
        )

    print(f"[visodom] visual return: {_end(visual)}", flush=True)
    print(f"[visodom] zero hold:     {_end(zero)}", flush=True)
    env.close()
    print("[visodom] done", flush=True)


if __name__ == "__main__":
    main()
