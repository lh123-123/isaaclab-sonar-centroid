#!/usr/bin/env python3
"""UI viewer for Direct-v7-flow-hover with the OCEAN CURRENT enabled and VISUALISED.

Opens the Isaac Sim viewport (1 env) with the water current switched on and draws a live
blue flow-field (grid of direction/magnitude arrows) so you can actually see the current
while the ROV tries to hold station against it.

Run:
    ./isaaclab.sh -p scripts/debug_v7_water_current_ui.py
    # or with the wrapper:
    bash scripts/debug_v7_water_current_ui.sh

Useful flags:
    --num_envs 1
    --steps 600            # how many policy steps to run (~0.2 s each -> 600 ~= 2 min)
    --strength 0.6         # base current speed in m/s (overrides the "medium" level)
    --direction 1.0,0.3,0.0  # unit-ish direction of the current (world frame)
    --no-modulation        # turn off the slow +/- sinusoid so the current is constant
"""
from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direct-v7-flow-hover UI with visible ocean current.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--strength", type=float, default=None, help="base current speed (m/s)")
parser.add_argument("--direction", type=str, default=None, help="comma-separated world-frame direction, e.g. 1.0,0.3,0.0")
parser.add_argument("--no-modulation", action="store_true", help="disable the slow time modulation")
parser.add_argument("--no-arrows", action="store_true", help="disable the flow-field arrows (physics current stays on)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import torch

import isaaclab_tasks  # noqa: F401
import gymnasium as gym

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)

# -- env setup ---------------------------------------------------------------
cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.use_real_usd = False

# Make sure the water current is ON (the v7 default already enables it, but be explicit).
cfg.ocean_current.enabled = True
cfg.ocean_current.type = "uniform"
if args_cli.strength is not None:
    cfg.ocean_current.level = None
    cfg.ocean_current.strength = float(args_cli.strength)
if args_cli.direction is not None:
    direction = [float(v) for v in args_cli.direction.split(",")]
    if len(direction) != 3:
        raise SystemExit("--direction must be three comma-separated numbers, e.g. 1.0,0.3,0.0")
    cfg.ocean_current.params = dict(cfg.ocean_current.params)
    cfg.ocean_current.params["direction"] = direction
if args_cli.no_modulation:
    cfg.ocean_current.params = dict(cfg.ocean_current.params)
    cfg.ocean_current.params["time_modulation"] = False

env = gym.make("Isaac-AUV-Tracking-Direct-v7-flow-hover", cfg=cfg)
raw = env.unwrapped if hasattr(env, "unwrapped") else env
device = raw.device

print(f"\n[v7-current-ui] Env created with {raw.num_envs} env(s).", flush=True)
print(
    f"[v7-current-ui] ocean_current: enabled={cfg.ocean_current.enabled} "
    f"type={cfg.ocean_current.type} level={cfg.ocean_current.level} "
    f"strength={cfg.ocean_current.strength} params={cfg.ocean_current.params}",
    flush=True,
)
print("[v7-current-ui] Opening Isaac Sim viewport... check your Isaac Sim window.", flush=True)

# -- flow-field arrows via debug draw ---------------------------------------
draw = None
if not args_cli.no_arrows:
    try:
        from isaacsim.util.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
        print("[v7-current-ui] debug-draw flow field: ENABLED", flush=True)
    except Exception as exc:
        draw = None
        print(f"[v7-current-ui] debug-draw unavailable, arrows disabled: {exc}", flush=True)

# Build a static grid of sample points in the world volume around the ROV (env 0 is at origin).
grid_pts = []
if draw is not None:
    xs = [-1.0, 0.0, 1.0, 2.0]
    ys = [-1.5, -0.75, 0.0, 0.75, 1.5]
    zs = [-1.0, 0.0, 1.0]
    for x in xs:
        for y in ys:
            for z in zs:
                grid_pts.append((x, y, z))
    grid_tensor = torch.tensor(grid_pts, dtype=torch.float32, device=device)

ARROW_SCALE = 2.0     # metres of arrow per (m/s) of current -> visible length
LINE_WIDTH = 3.0
TIP_SIZE = 8.0
BASE_SIZE = 3.0
LINE_COLOR = (0.20, 0.55, 1.0, 1.0)   # blue water
TIP_COLOR = (0.55, 0.85, 1.0, 1.0)    # lighter blue tips
BASE_COLOR = (0.10, 0.30, 0.70, 1.0)  # darker base dots


def draw_flow_field(t_now: float) -> float:
    """Redraw the current arrows for time ``t_now``; returns mean speed (m/s)."""
    if draw is None or raw._ocean_current_field is None:
        return 0.0
    with torch.no_grad():
        vel = raw._ocean_current_field(grid_tensor, float(t_now))  # (M, 3) world m/s
    vel = vel.detach().cpu()
    base_cpu = grid_tensor.detach().cpu()
    speed = float(vel.norm(dim=-1).mean().item())
    ends = base_cpu + vel * ARROW_SCALE
    base_list = [tuple(p) for p in base_cpu.tolist()]
    end_list = [tuple(p) for p in ends.tolist()]
    try:
        draw.clear_lines()
        draw.clear_points()
        draw.draw_lines(base_list, end_list, [LINE_COLOR] * len(base_list), [LINE_WIDTH] * len(base_list))
        draw.draw_points(end_list, [TIP_COLOR] * len(end_list), [TIP_SIZE] * len(end_list))
        draw.draw_points(base_list, [BASE_COLOR] * len(base_list), [BASE_SIZE] * len(base_list))
    except Exception as exc:  # don't let viz break the run
        print(f"[v7-current-ui] draw error (ignored): {exc}", flush=True)
    return speed


# -- main loop --------------------------------------------------------------
obs, _ = env.reset()
step = 0
running = True
t0 = time.time()

while running and step < args_cli.steps:
    # Zero joystick action: the PID holds depth/attitude, the water current pushes the hull,
    # and the policy objective would be to null the ground-locked optical flow.
    act = torch.zeros(raw.num_envs, 4, device=device)
    obs, rew, term, trunc, info = env.step(act)

    t_now = time.time() - t0
    speed = 0.0
    if step % 2 == 0:
        speed = draw_flow_field(t_now)

    if step % 30 == 0:
        st = obs["policy"]["state"][0]
        flow_rms = float(st[8].item()) * 64.0
        depth_err = float(st[0].item())
        att_err = torch.sqrt(st[1] ** 2 + st[2] ** 2).item()
        pos = raw._robot.data.root_pos_w[0].cpu().numpy()
        cur = raw.get_ocean_currents()[0].cpu().numpy() if hasattr(raw, "get_ocean_currents") else None
        cur_str = (
            f" cur=({cur[0]:.2f},{cur[1]:.2f},{cur[2]:.2f})|{float((cur**2).sum()**0.5):.2f}m/s"
            if cur is not None else ""
        )
        print(
            f"[v7-current] step={step:4d} rew={float(rew.mean()):.3f} flow={flow_rms:.1f}px "
            f"depth={depth_err:.3f}m att={att_err:.4f} pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})"
            f"{cur_str}",
            flush=True,
        )

    if term[0] or trunc[0]:
        print(f"[v7-current] Terminated at step {step} (env reset continues).", flush=True)
        obs, _ = env.reset()

    time.sleep(0.2)
    step += 1

env.close()
print(f"\n[v7-current-ui] Done after {step} steps.", flush=True)
print("[v7-current-ui] Press Esc or close the Isaac Sim window to exit.", flush=True)
