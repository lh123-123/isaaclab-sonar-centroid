#!/usr/bin/env python3
"""UI viewer for Direct-v7-flow-hover-gsrov: opens Isaac Sim viewport with AUV + 4 orange spheres.
Run: ./scripts/debug_auv_v7_flow_hover_gsrov_ui.sh
"""
from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direct-v7-flow-hover-gsrov UI viewer.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--texture", type=str, default=None)
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

# -- env setup --
cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
# Use cuboid proxy (matches training config).
cfg.use_real_usd = False
if args_cli.texture:
    cfg.plane_texture_path = args_cli.texture
    # Match plane aspect to the image so it isn't stretched.
    try:
        from PIL import Image
        _w, _h = Image.open(args_cli.texture).size
        cfg.plane_aspect = float(_w) / float(_h)
        cfg.plane_texture_tile = 1.0
        # Half the previous size (was 2.2*dist); board length/width reduced by half.
        cfg.plane_size = 1.1 * cfg.plane_distance
    except Exception as _e:
        print(f"[v7-ui] aspect detect failed: {_e}", flush=True)
env = gym.make("Isaac-AUV-Tracking-Direct-v7-flow-hover-gsrov", cfg=cfg)
raw = env.unwrapped if hasattr(env, "unwrapped") else env

H, W = 64, 64

print(f"\n[v7-ui] Env created with {raw.num_envs} env(s).", flush=True)
print("[v7-ui] Opening Isaac Sim viewport... check your Isaac Sim window.", flush=True)

# -- main loop --
obs, _ = env.reset()
step = 0
running = True

while running and step < args_cli.steps:
    # Step env with zero actions (3-D [surge, sway, yaw])
    act = torch.zeros(raw.num_envs, 3)
    obs, rew, term, trunc, info = env.step(act)

    # Print metrics every 30 steps
    if step % 30 == 0:
        img = obs["policy"]["image"][0]
        st = obs["policy"]["state"][0]
        tex_px = int(img[6].sum().item())
        flow_rms = float(st[8].item()) * W
        depth_err = float(st[0].item())
        att_err = torch.sqrt(st[1]**2 + st[2]**2).item()
        pos = raw._robot.data.root_pos_w[0].cpu().numpy()
        print(f"[v7] step={step:4d} rew={float(rew.mean()):.3f} flow={flow_rms:.1f}px "
              f"tex={tex_px}/{H*W} depth={depth_err:.3f}m att={att_err:.4f} "
              f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})", flush=True)

    # Check termination
    if term[0] or trunc[0]:
        print(f"[v7] Terminated at step {step}", flush=True)
        break

    # Let UI update (approx 5 fps)
    time.sleep(0.2)
    step += 1

env.close()
print(f"\n[v7-ui] Done after {step} steps. You can now interact with the Isaac Sim viewport.", flush=True)
print("[v7-ui] Press Esc or close the Isaac Sim window to exit.", flush=True)
