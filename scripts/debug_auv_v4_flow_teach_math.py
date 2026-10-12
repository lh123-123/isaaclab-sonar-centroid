"""Synthetic checks for sparse flow, teach scale, and the MidCNN trunk.

No Isaac Sim. Run from the repo root:

    python scripts/debug_auv_v4_flow_teach_math.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks"
# Stub parents so importing the two modules does not execute isaaclab_tasks/__init__.py,
# which pulls in Isaac Sim.
for name, path in (
    ("isaaclab_tasks", PKG),
    ("isaaclab_tasks.direct", PKG / "direct"),
    ("isaaclab_tasks.direct.isaac_auv_tracking", PKG / "direct" / "isaac_auv_tracking"),
    (
        "isaaclab_tasks.direct.isaac_auv_tracking.agents",
        PKG / "direct" / "isaac_auv_tracking" / "agents",
    ),
):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules[name] = mod

from isaaclab_tasks.direct.isaac_auv_tracking.sparse_flow import sparse_flow_vector, step_locked_flow
from isaaclab_tasks.direct.isaac_auv_tracking.teach_match import detect_points, match_rgb, teach_from_numpy


def _rect(h: int, w: int, x0: int, y0: int, bw: int, bh: int) -> torch.Tensor:
    img = torch.full((h, w, 3), 40, dtype=torch.uint8)
    img[y0 : y0 + bh, x0 : x0 + bw, 0] = 220
    img[y0 : y0 + bh, x0 : x0 + bw, 1] = 90
    img[y0 : y0 + bh, x0 : x0 + bw, 2] = 30
    return img


def _stack_from_pair(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
    """prev/curr are (H, W) luma uint8. Build (1, 12, H, W) RGB with the pair at the end."""
    def rgb(gray: torch.Tensor) -> torch.Tensor:
        g = gray.float().view(1, 1, *gray.shape)
        return g.repeat(1, 3, 1, 1)

    prev_r = rgb(prev)
    curr_r = rgb(curr)
    # 4 RGB frames: two copies of prev, then prev, then curr. Newest is last.
    return torch.cat([prev_r, prev_r, prev_r, curr_r], dim=1)


def main() -> None:
    h = w = 64
    base = torch.zeros(h, w, dtype=torch.uint8)
    base[20:36, 16:32] = 255
    shifted = torch.zeros(h, w, dtype=torch.uint8)
    shifted[20:36, 20:36] = 255  # +4 px in x
    moved = sparse_flow_vector(_stack_from_pair(base, shifted).float() / 255.0)
    still = sparse_flow_vector(_stack_from_pair(base, base).float() / 255.0)
    print(
        f"[math] flow moved dx/W={float(moved[0, 0]):+.4f} dy/H={float(moved[0, 1]):+.4f} "
        f"valid={float(moved[0, 2]):.2f}",
        flush=True,
    )
    print(
        f"[math] flow still dx/W={float(still[0, 0]):+.4f} dy/H={float(still[0, 1]):+.4f} "
        f"valid={float(still[0, 2]):.2f}",
        flush=True,
    )
    orange = torch.zeros(1, 3, h, w)
    orange[:, 0, 20:36, 16:32] = 220.0 / 255.0
    orange[:, 1, 20:36, 16:32] = 90.0 / 255.0
    orange[:, 2, 20:36, 16:32] = 30.0 / 255.0
    orange_shift = torch.zeros_like(orange)
    orange_shift[:, 0, 20:36, 20:36] = 220.0 / 255.0
    orange_shift[:, 1, 20:36, 20:36] = 90.0 / 255.0
    orange_shift[:, 2, 20:36, 20:36] = 30.0 / 255.0
    xy0 = torch.zeros(1, 32, 2)
    alive0 = torch.zeros(1, 32, dtype=torch.bool)
    locked, _, _, _, _, _, xy1, alive1 = step_locked_flow(orange, orange, xy0, alive0)
    tracked, _, _, _, _, _, xy2, alive2 = step_locked_flow(orange, orange_shift, xy1, alive1)
    same_pts = int((alive1 & alive2).sum())
    print(
        f"[math] lock seed valid={float(locked[0, 2]):.2f} alive={int(alive1.sum())} "
        f"then dx/W={float(tracked[0, 0]):+.4f} kept={same_pts}",
        flush=True,
    )

    teach_img = _rect(128, 128, 44, 54, 40, 20)
    pts = detect_points(teach_img.unsqueeze(0))
    keep = pts[0, :, 0] >= 0
    print(f"[math] teach points={int(keep.sum())}", flush=True)
    ref = teach_from_numpy(teach_img.numpy(), pts[0, keep].numpy(), 2.0)
    same = match_rgb(teach_img.unsqueeze(0), ref, hori_fov_deg=90.0)
    small = _rect(128, 128, 54, 59, 20, 10)
    far = match_rgb(small.unsqueeze(0), ref, hori_fov_deg=90.0)
    right = _rect(128, 128, 70, 54, 40, 20)
    right_m = match_rgb(right.unsqueeze(0), ref, hori_fov_deg=90.0)
    print(
        f"[math] same r={float(same[0, 0]):.2f} a={float(same[0, 1]):+.3f} "
        f"s={float(same[0, 2]):.2f} inl={float(same[0, 3]):.2f}",
        flush=True,
    )
    print(
        f"[math] far  r={float(far[0, 0]):.2f} a={float(far[0, 1]):+.3f} "
        f"s={float(far[0, 2]):.2f} inl={float(far[0, 3]):.2f}",
        flush=True,
    )
    print(
        f"[math] right r={float(right_m[0, 0]):.2f} a={float(right_m[0, 1]):+.3f} "
        f"s={float(right_m[0, 2]):.2f} inl={float(right_m[0, 3]):.2f}",
        flush=True,
    )

    import gymnasium as gym
    import numpy as np
    from isaaclab_tasks.direct.isaac_auv_tracking.agents.cam_encoder import MidCNNExtractor

    space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(0, 255, (12, 64, 64), dtype=np.uint8),
            "state": gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
            "teach": gym.spaces.Box(0.0, 1.0, (4,), dtype=np.float32),
        }
    )
    net = MidCNNExtractor(space, features_dim=32, use_sparse_flow=True).eval()
    images = torch.randint(0, 255, (2, 12, 64, 64), dtype=torch.uint8)
    teach = torch.zeros(2, 4)
    with torch.no_grad():
        y = net({"image": images, "state": torch.zeros(2, 4), "teach": teach})
    v3 = MidCNNExtractor(gym.spaces.Box(0, 255, (4, 64, 64), dtype=np.uint8), features_dim=32).eval()
    with torch.no_grad():
        y3 = v3(torch.randint(0, 255, (2, 4, 64, 64), dtype=torch.uint8))
    print(f"[math] v4 out={tuple(y.shape)} finite={bool(torch.isfinite(y).all())}", flush=True)
    print(f"[math] v3 out={tuple(y3.shape)} trunk_in={v3.trunk[0].in_features}", flush=True)


if __name__ == "__main__":
    main()
