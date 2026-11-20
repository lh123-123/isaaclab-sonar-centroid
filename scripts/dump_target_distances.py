# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dump v5-flow camera frames with the target parked at fixed distances dead ahead.

    bash isaaclab.sh -p scripts/dump_target_distances.py --headless

Writes logs/auv_tracking/target_distances/d_{0.5,1,1.5,2}m.png (render-res RGB).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="logs/auv_tracking/target_distances")
    args, unknown = parser.parse_known_args()

    import sys

    import torch

    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]] + unknown
    live = argparse.ArgumentParser()
    live.add_argument("--out_dir", type=str, default=args.out_dir)
    AppLauncher.add_app_launcher_args(live)
    live_args = live.parse_args()
    live_args.enable_cameras = True

    app_launcher = AppLauncher(live_args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from PIL import Image

    env_cfg = parse_env_cfg("Isaac-AUV-Tracking-Direct-v5-flow", device="cuda:0", num_envs=1)
    env = gym.make("Isaac-AUV-Tracking-Direct-v5-flow", cfg=env_cfg)
    raw = env.unwrapped
    out = Path(live_args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset()
    # (dist, depth_offset): 0.5m dead-ahead overlaps the agent body (0.61m contact),
    # so PhysX depenetration ejects the target; lift it 0.2m for that shot only.
    shots = ((0.5, 0.2), (0.7, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0))
    for dist, dz in shots:
        env.reset()
        # Park the target ahead of the agent at `dist`, same depth (+dz).
        with torch.no_grad():
            agent_xy, agent_yaw = raw._agent_pose_2d()
            agent_xy = agent_xy[0:1]
            fwd = torch.stack(
                [torch.cos(agent_yaw[0:1]), torch.sin(agent_yaw[0:1])], dim=-1
            )
            want = agent_xy + fwd * dist
            raw._target_state[0:1, 0:2] = want
            raw._target_state[0:1, 2:4] = 0.0
            raw.target_motion.yaw[0:1] = torch.atan2(fwd[0, 1], fwd[0, 0])
            raw._sync_target_entity(torch.tensor([0], device=raw.device))
            if dz:
                p = raw._target.data.root_pos_w.clone()
                q = raw._target.data.root_quat_w.clone()
                p[0:1, 2] -= dz
                raw._target.write_root_pose_to_sim(
                    torch.cat([p[0:1], q[0:1]], dim=-1), torch.tensor([0], device=raw.device)
                )
            zero = torch.zeros((1, 2), device=raw.device)
            for _ in range(30):
                raw._target_state[0:1, 0:2] = want
                raw._target_state[0:1, 2:4] = 0.0
                raw._sync_target_entity(torch.tensor([0], device=raw.device))
                if dz:
                    p = raw._target.data.root_pos_w.clone()
                    q = raw._target.data.root_quat_w.clone()
                    p[0:1, 2] -= dz
                    raw._target.write_root_pose_to_sim(
                        torch.cat([p[0:1], q[0:1]], dim=-1), torch.tensor([0], device=raw.device)
                    )
                env.step(zero)
            frame = raw._camera_rgb()[0].detach().cpu().numpy()
        tag = str(dist).replace(".", "p")
        Image.fromarray(frame).save(out / f"d_{tag}m.png")
        print(f"[dump] dist={dist}m frame={frame.shape} saved", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
