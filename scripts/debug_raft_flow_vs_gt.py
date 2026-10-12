#!/usr/bin/env python3
"""RAFT-Small vs GT flow on the real v6 scene render.

Runs Isaac-AUV-Tracking-Direct-v6-flowvec headless, steps with a fixed mild
action so the target moves, captures render-res RGB frames, runs torchvision
RAFT-Small on consecutive frames, and compares against the env's GT flow disk.

Usage:
    ./scripts/debug_raft_flow_vs_gt.sh   # or python directly with Isaac args
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RAFT-Small vs GT flow on v6 scene.")
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v6-flowvec")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--out", type=str, default="raft_vs_gt_scene.png")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def flow_to_rgb(f: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
    hsv = np.zeros((f.shape[0], f.shape[1], 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / (mag.max() + 1e-6) * 255, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        raw = env.unwrapped
        device = str(raw.device)

        weights = Raft_Small_Weights.DEFAULT
        raft = raft_small(weights=weights).to(device).eval()
        pre = weights.transforms()

        env.reset()
        # let soft-start pass so there is real motion
        act = torch.zeros(raw.action_space.shape, device=raw.device)
        act[:, 0] = 0.3
        prev_rgb = None
        rows = []
        for step in range(args_cli.steps):
            out = env.step(act)
            rgb = raw._camera_rgb()[0].detach()  # (Rh, Rw, 3) uint8 RGB
            gt = raw._flow_image[0].detach().cpu().numpy().transpose(1, 2, 0)  # (64,64,2) du,dv normed
            gt_px = gt * np.array([64.0, 64.0], np.float32)  # back to px at 64
            if prev_rgb is not None:
                t1 = prev_rgb.permute(2, 0, 1).unsqueeze(0).to(device)
                t2 = rgb.permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    b1, b2 = pre(t1, t2)
                    est = raft(b1, b2)[-1][0].permute(1, 2, 0).cpu().numpy()  # px at render res
                Rh = rgb.shape[0]
                est64 = cv2.resize(est, (64, 64)) * (64.0 / Rh)
                gt_mask = (np.abs(gt).sum(-1) > 0)
                if gt_mask.sum() > 0:
                    epe = np.linalg.norm(est64 - gt_px, axis=-1)[gt_mask].mean()
                    emag = np.linalg.norm(est64, axis=-1)[gt_mask].mean()
                    gmag = np.linalg.norm(gt_px, axis=-1)[gt_mask].mean()
                else:
                    epe, emag, gmag = float("nan"), 0.0, 0.0
                print(f"step {step}: gt_lit={int(gt_mask.sum())}px gt_mag={gmag:.3f}px "
                      f"raft_mag={emag:.3f}px EPE_on_disk={epe:.3f}px", flush=True)
                if step in (args_cli.steps - 3, args_cli.steps - 1):
                    f1 = (prev_rgb.cpu().numpy()).astype(np.uint8)[..., ::-1]
                    f1 = cv2.resize(f1, (256, 256))
                    rows.append((f1, est64, gt_px, epe))
            prev_rgb = rgb.clone()

        panels = []
        for f1, est64, gt_px, epe in rows:
            panels.append(np.hstack([
                f1,
                cv2.resize(flow_to_rgb(est64), (256, 256), interpolation=cv2.INTER_NEAREST),
                cv2.resize(flow_to_rgb(gt_px), (256, 256), interpolation=cv2.INTER_NEAREST),
            ]))
        viz = np.vstack(panels)
        cv2.imwrite(args_cli.out, viz)
        print(f"saved {args_cli.out}", flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    import traceback

    try:
        rc = main()
    except BaseException:
        traceback.print_exc()
        rc = 1
    finally:
        simulation_app.close()
    sys.exit(rc)
