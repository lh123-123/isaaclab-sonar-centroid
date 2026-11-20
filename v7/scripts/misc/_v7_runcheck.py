"""Minimal run-check for the v7 flow-hover GS-ROV env.

Launches Isaac Sim, builds the env directly, resets, then steps a few times with zero
actions. Reports observation/reward sanity and per-env motion (so we can see buoyancy /
PID / thrusters actually move the body). Works headless or with the UI, single or multi env.

Usage:
  ./isaaclab.sh -p v7/scripts/misc/_v7_runcheck.py --headless --num_envs 1
  ./isaaclab.sh -p v7/scripts/misc/_v7_runcheck.py --num_envs 1            # UI (needs DISPLAY)
  ./isaaclab.sh -p v7/scripts/misc/_v7_runcheck.py --headless --num_envs 4
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 run-check")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # v7 needs the TiledCamera

app_launcher = AppLauncher(args_cli)

import torch

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)

FAILS = []


def check(name: str, cond: bool, detail: str = ""):
    ok = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"  [{ok}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def main():
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    n = env.num_envs
    mode = "headless" if args_cli.headless else "UI"
    print(f"\n[v7-run] mode={mode}  num_envs={n}", flush=True)

    obs, _ = env.reset()
    img = obs["policy"]["image"]
    st = obs["policy"]["state"]
    print(f"[v7-run] image={tuple(img.shape)} dtype={img.dtype}  state={tuple(st.shape)}", flush=True)
    check("image shape (N,7,64,64)", tuple(img.shape[1:]) == (7, 64, 64), f"{tuple(img.shape)}")
    check("state shape (N,11)", tuple(st.shape[1:]) == (11,), f"{tuple(st.shape)}")  # 9 dims + present + tex_frac
    check("obs finite", bool(torch.isfinite(img).all() and torch.isfinite(st).all()))

    pos0 = env._robot.data.root_pos_w.clone()
    quat0 = env._robot.data.root_quat_w.clone()
    print(f"[v7-run] spawn pos0[0]={pos0[0].tolist()} quat0[0]={quat0[0].tolist()}", flush=True)

    saw_flow = False
    for step in range(args_cli.steps):
        act = torch.zeros(n, 3, device=env.device)
        obs, rew, term, trunc, info = env.step(act)
        img = obs["policy"]["image"]
        st = obs["policy"]["state"]
        finite = bool(torch.isfinite(img).all() and torch.isfinite(st).all() and torch.isfinite(rew).all())
        in_range = bool((img.abs() <= 1.0 + 1e-6).all())
        if bool((img[:, 3:6].abs() > 1e-6).any()):
            saw_flow = True
        dp = (env._robot.data.root_pos_w - pos0).norm(dim=-1)
        check(f"step {step} finite", finite, f"rew={float(rew.mean()):.4f}")
        check(f"step {step} img in [-1,1]", in_range, f"absmax={float(img.abs().max()):.3f}")
        print(
            f"[v7-run] step {step:2d}: rew={float(rew.mean()):+.4f} "
            f"flow_rms_px={float(st[0,8])*64:.3f} depth_err={float(st[0,0]):+.4f} "
            f"pitch={float(st[0,2]):+.4f} att_err_rel~ step  term={int(term.sum())} "
            f"move[0]={float(dp[0]):.4f}m",
            flush=True,
        )
        if bool(term.any()):
            print(f"[v7-run] terminated envs at step {step}: {term.nonzero().flatten().tolist()}", flush=True)

    check("flow became non-zero", saw_flow)
    check("body moved under physics", bool((env._robot.data.root_pos_w - pos0).norm(dim=-1).max() > 1e-4))

    print("\n[v7-run] checks done. RESULT: " + ("OK" if not FAILS else f"MISMATCH ({len(FAILS)}): {FAILS}"), flush=True)

    # UI: keep the app alive and pumping so the viewport is interactive (orbit / click),
    # stepping zero actions so the ROV just floats. Close the window to exit.
    if not args_cli.headless:
        print("[v7-run] UI interactive mode: close the Isaac Sim window to exit.", flush=True)
        app = app_launcher.app
        act = torch.zeros(n, 3, device=env.device)
        try:
            while app.is_running():
                env.step(act)
                app.update()
        except KeyboardInterrupt:
            pass
        try:
            env.close()
        except Exception as e:  # noqa: BLE001
            print(f"[v7-run] close warning: {e}", flush=True)
        return

    try:
        env.close()
    except Exception as e:  # noqa: BLE001
        print(f"[v7-run] close warning: {e}", flush=True)
    raise SystemExit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
