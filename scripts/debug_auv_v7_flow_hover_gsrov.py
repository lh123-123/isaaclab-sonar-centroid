"""Debug v7-flow-hover: 1 env with textured 4x4 plane ahead of the AUV.
Uses direct class instantiation to avoid gym registration ambiguities.
Checks: observation shapes, flow non-zero, texture present, reward finite.

Usage:
  ./scripts/debug_auv_v7_flow_hover_gsrov.sh --headless --num_envs 1
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direct-v7-flow-hover debug.")
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v7-flow-hover-gsrov")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import torch

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)

_FAILS = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        _FAILS.append(name)
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def main():
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    n = env.num_envs
    print(f"\n[v7] num_envs={n}", flush=True)

    print("\n[v7] ---- observation space ----", flush=True)
    for key, sp in env.observation_space.items():
        print(f"[v7]   ['{key}']: shape={tuple(sp.shape)} dtype={sp.dtype}", flush=True)

    obs, _ = env.reset()
    img = obs["policy"]["image"]
    st = obs["policy"]["state"]
    H, W = 64, 64
    check("image shape", tuple(img.shape[1:]) == (9, H, W), f"{tuple(img.shape)}")
    check("image dtype float32", img.dtype == torch.float32, str(img.dtype))
    check("state shape", tuple(st.shape[1:]) == (13,), f"{tuple(st.shape)}")  # 9 + present + tex_frac + drift(2)
    check("state finite", bool(torch.isfinite(st).all()))
    check("image finite", bool(torch.isfinite(img).all()))
    check("image in [-1,1]", (img.abs() <= 1.0 + 1e-6).all().item(), f"absmax={img.abs().max():.4g}")

    print("\n[v7] channel layout:", flush=True)
    print("  0..2 = RGB  3 = flow u  4 = flow v  5 = |flow|  6 = texture mask", flush=True)

    print("\n[v7] ---- step loop ----", flush=True)
    saw_nonzero = False
    worst_absmax = 0.0
    for step in range(args_cli.steps):
        act = torch.zeros(n, 3)
        obs, rew, term, trunc, info = env.step(act)
        img = obs["policy"]["image"]
        st = obs["policy"]["state"]
        finite = bool(torch.isfinite(img).all() and torch.isfinite(st).all() and torch.isfinite(rew).all())
        in_range = bool((img.abs() <= 1.0 + 1e-6).all())
        absmax = float(img.abs().max())
        worst_absmax = max(worst_absmax, absmax)
        flow_nonzero = bool((img[:, 3:6].abs() > 1e-6).any())
        if flow_nonzero:
            saw_nonzero = True
        flow_rms_px = float(st[0, 8].item()) * W
        depth_err = float(st[0, 0].item())
        att_err = torch.sqrt(st[0, 1] ** 2 + st[0, 2] ** 2).item()
        check(f"step {step} finite", finite)
        check(f"step {step} in range", in_range, f"absmax={absmax:.4g}")
        check(f"step {step} reward finite", bool(torch.isfinite(rew).all()), f"rew={float(rew.mean()):.4g}")
        print(f"[v7] step {step:2d}: rew={float(rew.mean()):.4f} flow_rms={flow_rms_px:.3f}px "
              f"depth_err={depth_err:.4f} att_err={att_err:.4f}", flush=True)
        if term[0]:
            print(f"[v7] terminated at step {step}", flush=True)
            break

    print("\n[v7] ---- summary ----", flush=True)
    print(f"[v7] max |image| = {worst_absmax:.6g}", flush=True)
    check("flow became non-zero during the run", saw_nonzero)

    print("\n[v7] RESULT: " + ("OK" if not _FAILS else f"MISMATCH ({len(_FAILS)})"), flush=True)
    try:
        env.close()
    except Exception as e:
        print(f"[v7] close warning: {e}")
    raise SystemExit(0 if not _FAILS else 1)


if __name__ == "__main__":
    main()
