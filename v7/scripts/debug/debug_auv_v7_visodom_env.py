#!/usr/bin/env python3
"""Check the env's own visual odometry, not the standalone probe.

Drives sway then surge, compares env._visodom_m() with ground-truth body
displacement, checks the image/state channels carry that estimate, and checks
a reset clears it. Reward must follow the visodom distance.

    ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_visodom_env.py --headless
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 env visodom wiring check.")
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

_FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    if not cond:
        _FAILS.append(name)
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def _corr_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    den = float(np.dot(x, x))
    slope = float(np.dot(x, y) / den) if den > 1e-9 else 0.0
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-6 and y.std() > 1e-6 else 0.0
    return slope, corr


def _read(env):
    origin = env.scene.env_origins[0]
    pos = (env._robot.data.root_pos_w[0] - origin).detach()
    _roll, pitch, yaw = euler_xyz_from_quat(env._robot.data.root_quat_w)
    return pos, float(yaw[0]), float(pitch[0])


def _body(pos, pos0, yaw):
    dx = float(pos[0] - pos0[0])
    dy = float(pos[1] - pos0[1])
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _act(env, surge: float, sway: float) -> torch.Tensor:
    scale = float(env.cfg.cmd_vel_scale)
    raw = [surge / scale, sway / scale, 0.0]
    return torch.tensor([raw], dtype=torch.float32, device=env.device).clamp(-1.0, 1.0)


def main() -> None:
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    check("vis_odometry on", bool(cfg.vis_odometry))
    env.reset()

    pos0 = None
    sway_gt, sway_vo, surge_gt, surge_vo = [], [], [], []
    pitch_max = 0.0
    rew_at_rest = None
    rew_displaced = None
    for step in range(70):
        if step < 8:
            vs, vy = 0.0, 0.0
        elif step < 28:
            vs, vy = 0.0, 0.5
        elif step < 36:
            vs, vy = 0.0, 0.0
        else:
            vs, vy = 0.5, 0.0
        _obs, rew, _term, _trunc, _info = env.step(_act(env, vs, vy))
        pos, yaw, pitch = _read(env)
        pitch_max = max(pitch_max, abs(pitch))
        vo = env._visodom_m()[0].detach()
        if pos0 is None:
            pos0 = pos.clone()
            rew_at_rest = float(rew[0])
            continue
        gs, gy = _body(pos, pos0, yaw)
        if 8 <= step < 36:
            sway_gt.append(gy)
            sway_vo.append(float(vo[1]))
        if step >= 36:
            surge_gt.append(gs)
            surge_vo.append(float(vo[0]))
        if step in (27, 35, 55, 69):
            print(
                f"[env] step={step:3d} gt=({gs:+.3f},{gy:+.3f}) "
                f"visodom=({float(vo[0]):+.3f},{float(vo[1]):+.3f}) "
                f"rew={float(rew[0]):.3f} pitch={math.degrees(pitch):+.2f}°",
                flush=True,
            )
        if step == 69:
            rew_displaced = float(rew[0])
            img = _obs["policy"]["image"][0]
            st = _obs["policy"]["state"][0]
            sig = float(cfg.sigma_visodom_m)
            ch_x = float(img[9].mean())
            ch_y = float(img[10].mean())
            expect_x = float(np.clip(float(vo[0]) / sig, -1.0, 1.0))
            expect_y = float(np.clip(float(vo[1]) / sig, -1.0, 1.0))
            check("image ch9 = visodom x", abs(ch_x - expect_x) < 1e-4, f"{ch_x:.4f} vs {expect_x:.4f}")
            check("image ch10 = visodom y", abs(ch_y - expect_y) < 1e-4, f"{ch_y:.4f} vs {expect_y:.4f}")
            check("state[15] = visodom x", abs(float(st[15]) - float(vo[0])) < 1e-4, f"{float(st[15]):.4f}")
            check("state[16] = visodom y", abs(float(st[16]) - float(vo[1])) < 1e-4, f"{float(st[16]):.4f}")

    sy_s, sy_c = _corr_slope(np.asarray(sway_gt), np.asarray(sway_vo))
    sx_s, sx_c = _corr_slope(np.asarray(surge_gt), np.asarray(surge_vo))
    print(f"[env] sway visodom/gt slope={sy_s:+.3f} corr={sy_c:+.3f}", flush=True)
    print(f"[env] surge visodom/gt slope={sx_s:+.3f} corr={sx_c:+.3f}", flush=True)
    check("sway tracks", sy_c > 0.95 and 0.7 < sy_s < 1.6, f"slope={sy_s:.3f} corr={sy_c:.3f}")
    check("surge tracks", sx_c > 0.95 and 0.7 < sx_s < 1.6, f"slope={sx_s:.3f} corr={sx_c:.3f}")
    check("pitch stayed level", pitch_max < math.radians(2.0), f"max {math.degrees(pitch_max):.2f}°")
    check(
        "reward fell while displaced",
        rew_displaced is not None and rew_at_rest is not None and rew_displaced < rew_at_rest - 0.05,
        f"{rew_at_rest:.3f} -> {rew_displaced:.3f}",
    )

    env.reset()
    vo0 = env._visodom_m()[0].detach()
    check("reset clears visodom", float(vo0.norm()) < 0.02, f"|vo|={float(vo0.norm()):.4f}")
    _obs, rew, *_ = env.step(_act(env, 0.0, 0.0))
    check("reward back after reset", float(rew[0]) > 2.2, f"rew={float(rew[0]):.3f}")

    env.close()
    print(f"[env] {'FAIL ' + str(len(_FAILS)) if _FAILS else 'ALL PASS'}", flush=True)
    if _FAILS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
