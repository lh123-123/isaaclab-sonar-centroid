#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase A0: WarpAUV hover + OceanSim sonar + fixed-point / centroid check."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sonar centroid track — WarpAUV sonar smoke test.")
parser.add_argument("--task", type=str, default="Isaac-SonarCentroidTrack-AUV-Debug-Direct-v0")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/centroid_track/auv_sonar_smoke",
    help="Save annotated polar frames here (empty to skip).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import cv2
import numpy as np

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _polar_to_u8(polar: np.ndarray) -> np.ndarray:
    x = polar.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi > lo:
        x = (x - lo) / (hi - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def _save_overlay(raw, frame, out_path: Path, *, step: int, depth: float) -> None:
    gray = _polar_to_u8(raw.pipeline.polar_image[0].detach().cpu().numpy())
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    p_fixed = raw.fixed.p_fixed_abs_env(0)

    def _dot(pt, color: tuple[int, int, int]) -> None:
        if pt is None:
            return
        c, r = int(round(pt.col)), int(round(pt.row))
        cv2.circle(img, (c, r), 1, color, -1, lineType=cv2.LINE_AA)

    _dot(p_fixed, (0, 255, 255))
    _dot(frame.p_sonar_abs, (0, 255, 0))
    _dot(frame.p_perc_abs, (255, 0, 0))
    cv2.putText(
        img,
        f"step={step} depth_z={depth:.3f} valid={frame.valid_ratio:.2f}",
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[auv_sonar] saved {out_path.resolve()}", flush=True)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    out_dir = Path(args_cli.output_dir) if str(args_cli.output_dir).strip() else None
    obs, _ = env.reset()
    print(f"[auv_sonar] obs shape={obs['policy'].shape}", flush=True)

    zero = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    saved = False
    for step in range(int(args_cli.steps)):
        obs, _, terminated, truncated, _ = env.step(zero)
        if step % 10 == 0:
            raw = env.unwrapped
            frame = raw.get_centroid_frame(0)
            depth = float(raw.robot.data.root_pos_w[0, 2].item())
            print(f"[auv_sonar] step={step:4d} depth_z={depth:.3f} {frame.summary()}", flush=True)
            if out_dir is not None and not saved:
                _save_overlay(raw, frame, out_dir / "polar_auv_step0.png", step=step, depth=depth)
                saved = True
        if bool(terminated.any()) or bool(truncated.any()):
            print(f"[auv_sonar] done at step={step}", flush=True)
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
