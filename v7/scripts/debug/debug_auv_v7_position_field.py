#!/usr/bin/env python3
"""Can a first-frame Farneback field hold station?

The field is Farneback(frame_1, frame_now), fit as du = a + k x, dv = b + k y.
Template matching stays in the env and is only logged.

Three episodes, one env, current ramp on, level-attitude precharge untouched:

  1. calibration — sway, then surge, so we can see whether a and k track
     ground-truth body displacement.
  2. visual return — drive off the capture pose, then a P servo on that field
     tries to come home while the current keeps rising.
  3. zero hold — the same drive, then action 0. The velocity PID holds the
     offset; it does not know the way back. This is the comparison.

    ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_position_field.py --headless
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 first-frame position-field hover probe.")
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
from isaaclab_tasks.direct.isaac_auv_tracking.first_frame_position_field import FirstFramePositionField

_KP = 0.8
_DEAD_M = 0.04
_V_MAX = 0.5
_ABORT_M = 1.2
_DRIVE = 16
_HOLD_STEPS = 160


def _through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    den = float(np.dot(x, x))
    if den < 1e-8:
        return 0.0, 0.0
    slope = float(np.dot(x, y) / den)
    if x.size < 3 or float(x.std()) < 1e-4 or float(y.std()) < 1e-4:
        return slope, 0.0
    return slope, float(np.corrcoef(x, y)[0, 1])


def _body(pos: torch.Tensor, pos0: torch.Tensor, yaw: float) -> tuple[float, float]:
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


def _command(env, surge: float, sway: float) -> torch.Tensor:
    scale = float(env.cfg.cmd_vel_scale)
    raw = [surge / scale, sway / scale, 0.0]
    return torch.tensor([raw], dtype=torch.float32, device=env.device).clamp(-1.0, 1.0)


def _print_row(tag, step, gs, gy, a_px, k, env, cur, pitch) -> None:
    au = float(env._anchor_off_px[0, 0])
    rms = float(env._flow_rms[0])
    print(
        f"[posfield] {tag:7} step={step:3d}  gt=({gs:+.3f},{gy:+.3f}) m  "
        f"|gt|={math.hypot(gs, gy):.3f}  a={a_px:+.1f}px  k={k:+.4f}  "
        f"tmpl_u={au:+.1f}px  flow_rms={rms:.2f}px  cur={cur:.2f}  pitch={math.degrees(pitch):+.2f}°",
        flush=True,
    )


def _metres_from(slope_x: float, slope_y: float, corr_x: float, corr_y: float, f_px: float, z_m: float):
    """Map (k, a_px) to (surge_m, sway_m). Prefer the calibrated slope."""
    use_x = abs(corr_x) >= 0.7 and abs(slope_x) > 0.05
    use_y = abs(corr_y) >= 0.7 and abs(slope_y) > 5.0
    # Geometric fallback: +k is expansion (moved toward the plane), +a is scene-right
    # (moved to the body's left, +Y). Calibration replaces the sign and the scale.
    geom_x = 1.0 / z_m
    geom_y = f_px / z_m

    def metres(a_px: float, k: float) -> tuple[float, float]:
        x_m = k / slope_x if use_x else k / geom_x
        y_m = a_px / slope_y if use_y else a_px / geom_y
        return x_m, y_m

    return metres, use_x, use_y


def run_episode(env, field, kind: str, metres, steps: int):
    env.reset()
    field.reset()
    pos0 = None
    prev_ep = -1
    sway_gt, sway_a = [], []
    surge_gt, surge_k = [], []
    switched = None
    last = None
    aborted = ""
    # Servo uses the field measured on the previous step.
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
        tex_ok = env._texture_count >= int(env.cfg.min_texture_px)
        field.observe(env._camera_rgb(), tex_ok, env._texture_mask)
        pos, yaw, pitch, cur = _read(env)
        if pos0 is None:
            if bool(field.valid[0]):
                pos0 = pos.clone()
            continue
        gs, gy = _body(pos, pos0, yaw)
        a_px = float(field.a_px[0])
        k = float(field.k[0])
        if kind == "visual" and step >= _DRIVE - 1 and not bool(field.just_captured[0]):
            x_m, y_m = metres(a_px, k)
            if abs(x_m) < _DEAD_M:
                x_m = 0.0
            if abs(y_m) < _DEAD_M:
                y_m = 0.0
            cmd = (
                float(np.clip(-_KP * x_m, -_V_MAX, _V_MAX)),
                float(np.clip(-_KP * y_m, -_V_MAX, _V_MAX)),
            )
        if kind == "cal" and 8 <= step < 31 and not bool(field.just_captured[0]):
            sway_gt.append(gy)
            sway_a.append(a_px)
        if kind == "cal" and 31 <= step < 54 and not bool(field.just_captured[0]):
            surge_gt.append(gs)
            surge_k.append(k)
        if kind != "cal" and step == _DRIVE - 1:
            switched = (gs, gy)
        last = (gs, gy, a_px, k, pitch, cur, step)
        if step % 15 == 0 or step in (22, 30, 45, 53, _DRIVE, steps - 1):
            _print_row(kind, step, gs, gy, a_px, k, env, cur, pitch)
        if math.hypot(gs, gy) > _ABORT_M:
            aborted = f"|gt| {math.hypot(gs, gy):.2f} m"
            _print_row(kind, step, gs, gy, a_px, k, env, cur, pitch)
            break
        if abs(pitch) > 0.25:
            aborted = f"pitch {math.degrees(pitch):.1f} deg"
            break
    return {
        "sway_gt": np.asarray(sway_gt, dtype=np.float64),
        "sway_a": np.asarray(sway_a, dtype=np.float64),
        "surge_gt": np.asarray(surge_gt, dtype=np.float64),
        "surge_k": np.asarray(surge_k, dtype=np.float64),
        "switched": switched,
        "last": last,
        "aborted": aborted,
    }


def _norm(pair) -> float:
    if pair is None:
        return float("nan")
    return math.hypot(pair[0], pair[1])


def main() -> None:
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    field = FirstFramePositionField(
        env.num_envs,
        int(cfg.flow_render_height),
        int(cfg.flow_render_width),
        int(cfg.cam_height),
        int(cfg.cam_width),
        env.device,
    )
    z_m = float(cfg.plane_distance)
    f_px = float(env._anchor_f_px)
    print(
        f"[posfield] f_px={f_px:.1f}  Z={z_m:.2f} m  "
        f"geom sway {f_px / z_m:.1f} px/m  geom surge {1.0 / z_m:.2f} /m",
        flush=True,
    )

    def ident_geom(a_px: float, k: float) -> tuple[float, float]:
        return k * z_m, a_px * z_m / f_px

    cal = run_episode(env, field, "cal", ident_geom, 54)
    slope_y, corr_y = _through_origin(cal["sway_gt"], cal["sway_a"])
    slope_x, corr_x = _through_origin(cal["surge_gt"], cal["surge_k"])
    print(
        f"[posfield] sway: a_px = {slope_y:+.1f} px/m * gt_sway   corr={corr_y:+.3f}  "
        f"(geometry {f_px / z_m:+.1f})",
        flush=True,
    )
    print(
        f"[posfield] surge: k = {slope_x:+.3f} /m * gt_surge   corr={corr_x:+.3f}  "
        f"(geometry {1.0 / z_m:+.3f})",
        flush=True,
    )
    metres, use_x, use_y = _metres_from(slope_x, slope_y, corr_x, corr_y, f_px, z_m)
    print(f"[posfield] servo scale: surge {'calibrated' if use_x else 'geometric'}, "
          f"sway {'calibrated' if use_y else 'geometric'}", flush=True)
    visual = run_episode(env, field, "visual", metres, _HOLD_STEPS)
    zero = run_episode(env, field, "zero", metres, _HOLD_STEPS)

    def _end(result) -> str:
        last = result["last"]
        if last is None:
            return "no sample"
        gs, gy = last[0], last[1]
        extra = f"  aborted: {result['aborted']}" if result["aborted"] else ""
        return (
            f"switch {_norm(result['switched']):.3f} m -> end {math.hypot(gs, gy):.3f} m "
            f"at step {last[6]}{extra}"
        )

    print(f"[posfield] visual return: {_end(visual)}", flush=True)
    print(f"[posfield] zero hold:     {_end(zero)}", flush=True)
    env.close()
    print("[posfield] done", flush=True)


if __name__ == "__main__":
    main()
