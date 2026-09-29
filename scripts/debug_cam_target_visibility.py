# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure how visually distinguishable the target AUV is in the Direct-v3-cam image.

Why this exists
---------------
``debug_auv_cam_frames.py`` now proves the target is *rendered* (the depth image returns
~2.5 m at the target's projected pixel). But being present is not the same as being
*learnable*: the target is a second WarpAUV with the same materials as the agent, seen through
water fog, so its pixel contrast may be negligible. A CNN cannot track what it cannot see.

Method
------
Same camera pose for both cases (robot parked at the world origin so USD and sim agree, target
dead ahead in the camera's field of view), differing only in whether the target is present:

  ON  : target 2.65 m straight ahead
  OFF : target parked far behind/below the camera, outside the frustum

The absolute difference of the two RGB frames isolates the target's visual footprint exactly, so
``max|diff|`` is an upper bound on its contrast and the diff support is its silhouette. A depth
image is also dumped so the geometric silhouette can be compared against the RGB one.

Usage::

    ./scripts/debug_cam_target_visibility.sh --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Target AUV visibility / contrast check.")
    parser.add_argument("--out_dir", type=str, default="logs/auv_tracking/v3_cam_visibility")
    args, unknown = parser.parse_known_args()

    import torch

    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]] + unknown
    live_parser = argparse.ArgumentParser()
    live_parser.add_argument("--out_dir", type=str, default=args.out_dir)
    live_parser.add_argument("--hang_timeout", type=float, default=180.0)
    AppLauncher.add_app_launcher_args(live_parser)
    live_args = live_parser.parse_args()
    live_args.enable_cameras = True

    import faulthandler

    faulthandler.enable()
    if float(live_args.hang_timeout) > 0:
        faulthandler.dump_traceback_later(float(live_args.hang_timeout), exit=True)

    app_launcher = AppLauncher(live_args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from PIL import Image

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_cam_env import AUVTrackingCamEnvCfg

    cfg = AUVTrackingCamEnvCfg()
    cfg.scene.num_envs = 1
    cfg.tiled_camera.data_types = ["rgb", "distance_to_image_plane"]
    env = gym.make("Isaac-AUV-Tracking-Direct-v3-cam", cfg=cfg)
    raw = env.unwrapped
    dev = raw.device
    e0 = torch.tensor([0], device=dev, dtype=torch.long)

    print("[vis] resetting...", flush=True)
    env.reset()
    print("[vis] reset done", flush=True)

    # Freeze episode control so the parked poses stick.
    raw._get_dones = lambda: (
        torch.zeros(raw.num_envs, dtype=torch.bool, device=dev),
        torch.zeros(raw.num_envs, dtype=torch.bool, device=dev),
    )
    raw._sync_target_entity = lambda *a, **k: None

    out_dir = Path(live_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ROBOT = (0.0, 0.0, 8.0)  # matches the robot's authored spawn, so USD and sim coincide
    TGT_ON = (2.65 + 0.35, 0.0, 8.0)  # 2.65 m in front of the camera at x=0.35
    TGT_OFF = (0.0, 0.0, 1.0)  # below and behind the camera, outside the frustum

    def place(target_pos) -> tuple[np.ndarray, np.ndarray]:
        zero = torch.zeros((raw.num_envs, raw.action_space.shape[-1]), device=dev)
        pose = torch.tensor([[*ROBOT, 1.0, 0.0, 0.0, 0.0]], device=dev)
        tgt = torch.tensor([[*target_pos, 1.0, 0.0, 0.0, 0.0]], device=dev)
        vel = torch.zeros(1, 6, device=dev)
        for _ in range(4):
            raw._robot.write_root_pose_to_sim(pose, e0)
            raw._robot.write_root_velocity_to_sim(vel, e0)
            raw._target.write_root_pose_to_sim(tgt, e0)
            raw._target.write_root_velocity_to_sim(vel, e0)
            env.step(zero)
        rgb = raw._tiled_camera.data.output["rgb"]
        rgb = (rgb if rgb.dtype == torch.uint8 else rgb.to(torch.uint8))[0, ..., :3].detach().cpu().numpy()
        d = raw._tiled_camera.data.output["distance_to_image_plane"][0]
        depth = (d[..., 0] if d.ndim == 3 else d).detach().cpu().numpy()
        return rgb, depth

    rgb_on, depth_on = place(TGT_ON)
    Image.fromarray(rgb_on).save(str(out_dir / "target_on.png"))
    Image.fromarray(rgb_off := place(TGT_OFF)[0]).save(str(out_dir / "target_off.png"))

    # --- RGB difference isolates the target's visual footprint exactly ---
    diff = np.abs(rgb_on.astype(np.int16) - rgb_off.astype(np.int16)).max(axis=2)
    Image.fromarray(diff.astype(np.uint8)).resize((512, 512), Image.NEAREST).save(
        str(out_dir / "diff_x8.png")
    )

    # --- geometric silhouette from depth ---
    tgt_mask = depth_on < 10.0  # the seafloor is ~19.7 m away, the target ~2.65 m

    print(f"\n[vis] RGB: on mean={rgb_on.mean():.1f} std={rgb_on.std():.1f}"
          f"  | off mean={rgb_off.mean():.1f} std={rgb_off.std():.1f}", flush=True)
    print(
        f"[vis] |diff| max={diff.max()}  mean={diff.mean():.2f}"
        f"  pixels>10: {(diff > 10).sum()}  pixels>20: {(diff > 20).sum()}"
        f"  (out of {diff.size})",
        flush=True,
    )
    print(
        f"[vis] depth<10m (target silhouette): {int(tgt_mask.sum())} px"
        f"  range={depth_on[tgt_mask].min() if tgt_mask.any() else float('nan'):.2f}"
        f"..{depth_on[tgt_mask].max() if tgt_mask.any() else float('nan'):.2f} m",
        flush=True,
    )
    if tgt_mask.any():
        yy, xx = np.mgrid[0 : depth_on.shape[0], 0 : depth_on.shape[1]]
        print(
            f"[vis] silhouette bbox: x={xx[tgt_mask].min()}..{xx[tgt_mask].max()}"
            f"  y={yy[tgt_mask].min()}..{yy[tgt_mask].max()}",
            flush=True,
        )
        rgb_in = rgb_on[tgt_mask].astype(np.float32)
        rgb_out = rgb_on[~tgt_mask].astype(np.float32)
        print(
            f"[vis] target pixels RGB mean={rgb_in.mean(axis=0).round(1).tolist()}"
            f"  background RGB mean={rgb_out.mean(axis=0).round(1).tolist()}",
            flush=True,
        )
        print(
            f"[vis] => luminance contrast |target-bg| ="
            f" {abs(rgb_in.mean() - rgb_out.mean()):.1f} / 255",
            flush=True,
        )
    # How much of the target silhouette actually changes the image?
    overlap = int(((diff > 10) & tgt_mask).sum())
    denom = max(int(tgt_mask.sum()), 1)
    print(
        f"[vis] silhouette pixels also changed by >10: {overlap}/{denom}"
        f" = {100.0 * overlap / denom:.1f}%",
        flush=True,
    )
    print(f"[vis] PNGs in {out_dir}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    import os
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    os._exit(0)
