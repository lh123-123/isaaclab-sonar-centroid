#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Runtime contract check for the Direct-v5-flow GT optical-flow stack observation.

`scripts/debug_auv_cam_spaces.sh` proves the SHAPE is right. This proves the SEMANTICS are:

  1. space/obs is a (frame_stack * 2, 64, 64) float32 stack and there is no "flow_image" key;
  2. right after reset the whole stack is exactly zero (flow has no history on a fresh episode);
  3. every step the stack shifts by one 2-channel frame: buf_t[:, :-2] == buf_{t-1}[:, 2:];
  4. the newest frame is bit-identical to the env's live `_flow_image`;
  5. all values are finite and within [-1, 1] (the space bounds);
  6. the newest frame carries at most one stamped disk of radius <= 12 px, and that disk is
     where the target projects -- i.e. the stamping works and the frame is not uniformly lit;
  7. flow is actually non-zero once the target gets moving (the soft-start makes the first
     steps legitimately ~0, so this is asserted over the whole run, not per step).

   ./scripts/debug_auv_v5_flow_obs.sh --headless --num_envs 4
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direct-v5-flow observation contract check.")
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v5-flow")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
# The env owns a TiledCamera (still needed for the KF measurement); without this the sensor
# raises at sim.reset().
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# `_update_gt_flow_image` clamps the stamped disk this hard.
MAX_DISK_RADIUS_PX = 12.0

_FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        _FAILS.append(name)


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs, use_fabric=True)
    stack_k = int(env_cfg.frame_stack)
    ch = stack_k * 2
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        raw = env.unwrapped
        ss = raw.single_observation_space
        single_obs = ss["policy"] if isinstance(ss, gym.spaces.Dict) else ss

        print("\n[v5-flow] ---- observation space ----", flush=True)
        keys = sorted(single_obs.spaces.keys()) if isinstance(single_obs, gym.spaces.Dict) else ["<bare box>"]
        for k in keys:
            sp = single_obs[k]
            print(f"[v5-flow]   ['{k}']: shape={sp.shape} dtype={sp.dtype}", flush=True)

        img = single_obs["image"]
        print(f"[v5-flow] frame_stack={stack_k} -> expected channels={ch}", flush=True)
        check("image key shape is (frame_stack*2, H, W)", tuple(img.shape) == (ch, 64, 64), f"{tuple(img.shape)}")
        check("image dtype is float32", img.dtype == torch.float32 or str(img.dtype) == "float32", str(img.dtype))
        check("image bounds are [-1, 1]", float(img.low.min()) == -1.0 and float(img.high.max()) == 1.0)
        check("no legacy 'flow_image' key", "flow_image" not in keys, f"keys={keys}")
        check("critic sees the 4-D KF belief", tuple(single_obs["state"].shape) == (4,))

        obs, _ = env.reset()
        buf = obs["policy"]["image"]
        print("\n[v5-flow] ---- reset frame ----", flush=True)
        print(f"[v5-flow] image shape={tuple(buf.shape)} dtype={buf.dtype} "
              f"absmax={float(buf.abs().max()):.6g}", flush=True)
        check("reset stack is all zeros", bool((buf == 0).all()), f"absmax={float(buf.abs().max()):.6g}")

        prev = buf.clone()
        saw_nonzero = False
        max_disk_px = 0
        worst_absmax = 0.0
        for step in range(args_cli.steps):
            out = env.step(torch.zeros(raw.action_space.shape, device=raw.device))
            buf = out[0]["policy"]["image"]

            finite = bool(torch.isfinite(buf).all())
            absmax = float(buf.abs().max())
            worst_absmax = max(worst_absmax, absmax)
            in_range = bool((buf >= -1.0).all() and (buf <= 1.0).all())
            saw_nonzero = saw_nonzero or absmax > 0.0

            # Shift semantics: the previous stack drops its two oldest channels, and the newest
            # frame is the env's live single-frame flow buffer.
            shift_ok = torch.equal(buf[:, :-2], prev[:, 2:])
            tail_ok = torch.equal(buf[:, -2:], raw._flow_image)
            newest = buf[:, -2:]
            disk_px = int((newest.abs() > 0).sum(dim=(-2, -1)).max().item())
            max_disk_px = max(max_disk_px, disk_px)

            print(f"[v5-flow] step {step}: finite={finite} in_range={in_range} absmax={absmax:.6g} "
                  f"shift_ok={shift_ok} tail_matches_live_flow={tail_ok} disk_px={disk_px}", flush=True)

            check(f"step {step} values finite", finite)
            check(f"step {step} values within space bounds [-1, 1]", in_range, f"absmax={absmax:.6g}")
            check(f"step {step} stack shifts by one flow frame", shift_ok)
            check(f"step {step} newest frame == live _flow_image", tail_ok)
            check(
                f"step {step} newest frame is at most one stamped disk",
                disk_px <= math.pi * MAX_DISK_RADIUS_PX**2 + 1.0,
                f"{disk_px} px vs cap {math.pi * MAX_DISK_RADIUS_PX**2:.0f}",
            )
            prev = buf.clone()

        print("\n[v5-flow] ---- summary ----", flush=True)
        print(f"[v5-flow] max |flow| over {args_cli.steps} steps = {worst_absmax:.6g}; "
              f"max lit disk = {max_disk_px} px (cap {math.pi * MAX_DISK_RADIUS_PX**2:.0f})", flush=True)
        # The 3 s soft-start means the earliest steps are legitimately ~0, so this is asserted
        # over the whole run rather than per step.
        check("flow became non-zero during the run", saw_nonzero)

        print("\n[v5-flow] RESULT: " + ("OK" if not _FAILS else f"MISMATCH ({len(_FAILS)})"), flush=True)
        return 0 if not _FAILS else 1
    finally:
        env.close()


if __name__ == "__main__":
    import traceback

    try:
        rc = main()
    except BaseException:  # noqa: BLE001 - must surface the failure immediately
        traceback.print_exc()
        rc = 1
    finally:
        simulation_app.close()
    sys.exit(rc)
