# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Save the 4 RGB frames the v4 actor actually receives.

The policy image is ``(12, 64, 64)`` uint8: four RGB frames, oldest first,
newest in the last three channels. Each frame is written at native 64 and
nearest-neighbour scaled so the pixels stay visible.

    ./scripts/debug_auv_v4_actor_frames.sh --steps 8
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump the v4 actor frame stack.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--action_v", type=float, default=0.3)
parser.add_argument("--action_w", type=float, default=0.15)
parser.add_argument("--out_dir", type=str, default="logs/auv_v4_actor_frames")
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v4-cam")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v4_cam_env import AUVTrackingV4CamEnvCfg


def _save_stack(buf: torch.Tensor, out_dir: Path, step: int, scale: int = 8) -> None:
    """``buf`` is ``(12, 64, 64)`` uint8, oldest frame first."""
    frames = buf.detach().byte().cpu()
    native = []
    for i in range(4):
        rgb = frames[i * 3 : (i + 1) * 3].permute(1, 2, 0).numpy()
        native.append(rgb)
        path = out_dir / f"step{step:02d}_frame{i}.png"
        Image.fromarray(rgb, mode="RGB").resize((64 * scale, 64 * scale), Image.Resampling.NEAREST).save(path)
    strip = np.concatenate(native, axis=1)
    Image.fromarray(strip, mode="RGB").resize((strip.shape[1] * scale, strip.shape[0] * scale), Image.Resampling.NEAREST).save(
        out_dir / f"step{step:02d}_stack.png"
    )
    print(f"[frames] step {step} oldest->newest {out_dir}", flush=True)


def main() -> None:
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = AUVTrackingV4CamEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg)
    raw = env.unwrapped
    env.reset()
    actions = torch.zeros((raw.num_envs, raw.action_space.shape[-1]), device=raw.device)
    actions[:, 0] = float(args_cli.action_v)
    actions[:, 1] = float(args_cli.action_w)
    # Reset fills all four slots with the same first frame. Step until the stack has moved.
    for step in range(int(args_cli.steps)):
        env.step(actions)
        if step in (0, 3, 7, int(args_cli.steps) - 1) or (step + 1) % 8 == 0:
            _save_stack(raw._frame_buf[0], out_dir, step + 1)
            yaw = float(raw.target_motion.yaw[0])
            print(f"[frames] target yaw {math.degrees(yaw):+.1f} deg", flush=True)
    env.close()
    print(f"[frames] wrote {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
