#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify sonar → feature → policy observation chain (headless-safe).

Checks:
  1. OceanSim sonar initialized and producing non-zero images
  2. Feature tracker has valid reference points
  3. Policy obs feat_err / valid_flags are non-trivial
  4. Obs reacts when target moves (proves live sonar, not stale zeros)

Usage:
  python scripts/debug_sonar_obs_headless.py --headless --enable_cameras
  python scripts/debug_sonar_obs_headless.py --sonar_ui          # GUI + viewport
  python scripts/debug_sonar_obs_headless.py --headless --save_png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug sonar → obs pipeline (headless vs GUI).")
parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Lateral1D-Debug-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=60, help="Warmup steps before snapshot.")
parser.add_argument("--target_dy", type=float, default=0.25, help="Body-Y offset (m) for reactivity test.")
parser.add_argument(
    "--sonar_ui",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable OceanSim viewport (GUI only; no effect on tensor pipeline in headless).",
)
parser.add_argument(
    "--save_png",
    action="store_true",
    default=False,
    help="Save raw sonar + cart PNG under logs/sonar_obs_debug/.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.sonar_flow_hover.modules.constants import MAX_TRACK_POINTS, MIN_VALID_FEATURES
from isaaclab_tasks.utils import parse_env_cfg


def _tensor_stats(t: torch.Tensor) -> dict[str, float]:
    flat = t.detach().float().reshape(-1)
    nz = (flat.abs() > 1.0e-8).float().mean().item()
    return {
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "nz_frac": nz,
    }


def _decode_obs(obs_row: torch.Tensor) -> dict[str, torch.Tensor]:
    n = MAX_TRACK_POINTS
    return {
        "feat_err": obs_row[: 2 * n],
        "valid": obs_row[2 * n : 3 * n],
        "body": obs_row[3 * n : 3 * n + 8],
        "prev_action": obs_row[3 * n + 8 :],
    }


def _print_stats(label: str, env, obs: torch.Tensor, step: int) -> dict:
    u = env.unwrapped
    eid = 0
    sonar = u.sonar_image[eid]
    cart = u.cart_image[eid]
    ft = u.feature_tracker
    valid_n = int(ft.valid_count()[eid].item())
    parts = _decode_obs(obs[eid])

    sonar_s = _tensor_stats(sonar)
    cart_s = _tensor_stats(cart)
    feat_s = _tensor_stats(parts["feat_err"])
    valid_sum = float(parts["valid"].sum().item())

    initialized = bool(getattr(u.oceansim_sonar, "_initialized", False))
    has_gui = bool(u.sim.has_gui())
    viewport_on = bool(getattr(u.cfg.oceansim_sonar_cfg, "sonar_initialize_viewport", False))

    print(f"\n--- [{label}] step={step} ---")
    print(f"  sim.has_gui={has_gui}  sonar_viewport_cfg={viewport_on}  oceansim_initialized={initialized}")
    print(f"  ref_mode={u.ref_mgr.mode}  valid_features={valid_n}/{MAX_TRACK_POINTS} (min={MIN_VALID_FEATURES})")
    print(
        f"  sonar_image  min={sonar_s['min']:.4f} max={sonar_s['max']:.4f} "
        f"mean={sonar_s['mean']:.4f} nz={sonar_s['nz_frac']:.2%}"
    )
    print(
        f"  cart_image   min={cart_s['min']:.4f} max={cart_s['max']:.4f} "
        f"mean={cart_s['mean']:.4f} nz={cart_s['nz_frac']:.2%}"
    )
    print(
        f"  obs.feat_err min={feat_s['min']:.4f} max={feat_s['max']:.4f} "
        f"mean={feat_s['mean']:.4f} nz={feat_s['nz_frac']:.2%}"
    )
    print(f"  obs.valid_sum={valid_sum:.0f}  obs.dim={obs.shape[-1]}  pixel_err={u.get_ibvs_metrics(0)['pixel_err']:.4f}")

    return {
        "initialized": initialized,
        "sonar_nz": sonar_s["nz_frac"],
        "cart_nz": cart_s["nz_frac"],
        "valid_n": valid_n,
        "feat_nz": feat_s["nz_frac"],
        "obs": obs[eid].clone(),
    }


def _save_png(env, tag: str) -> None:
    u = env.unwrapped
    out = Path("logs/sonar_obs_debug")
    out.mkdir(parents=True, exist_ok=True)
    sonar = u.sonar_image[0].detach().cpu().numpy()
    cart = u.cart_image[0].detach().cpu().numpy()
    sonar_u8 = np.clip(sonar / max(sonar.max(), 1.0e-6) * 255.0, 0, 255).astype(np.uint8)
    cart_u8 = np.clip(cart / max(cart.max(), 1.0e-6) * 255.0, 0, 255).astype(np.uint8)
    import cv2

    cv2.imwrite(str(out / f"{tag}_sonar.png"), sonar_u8)
    cv2.imwrite(str(out / f"{tag}_cart.png"), cart_u8)
    print(f"  saved PNG → {out}/{tag}_*.png")


def _run_checks(baseline: dict, after_move: dict, headless: bool) -> None:
    print("\n======== DIAGNOSTIC SUMMARY ========")
    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            "OceanSim initialized",
            baseline["initialized"],
            "oceansim_sonar._initialized must be True after scene load",
        )
    )
    checks.append(
        (
            "Sonar image non-empty",
            baseline["sonar_nz"] > 0.01,
            f"sonar nz_frac={baseline['sonar_nz']:.2%} (expect >1% non-zero pixels)",
        )
    )
    checks.append(
        (
            "Cart image non-empty",
            baseline["cart_nz"] > 0.01,
            f"cart nz_frac={baseline['cart_nz']:.2%}",
        )
    )
    checks.append(
        (
            "Enough valid features",
            baseline["valid_n"] >= MIN_VALID_FEATURES,
            f"valid={baseline['valid_n']} need>={MIN_VALID_FEATURES}",
        )
    )
    checks.append(
        (
            "feat_err in policy obs",
            baseline["feat_nz"] > 0.0 and baseline["valid_n"] > 0,
            "obs feat_err should be non-zero when features track",
        )
    )
    delta = torch.linalg.norm(after_move["obs"] - baseline["obs"]).item()
    checks.append(
        (
            "Obs reacts to target motion",
            delta > 1.0e-4,
            f"L2(obs_after - obs_before)={delta:.6f} (proves live sonar, not frozen zeros)",
        )
    )

    mode = "headless" if headless else "GUI"
    print(f"  mode: {mode}")
    all_ok = True
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}: {detail}")
        all_ok = all_ok and ok

    if all_ok:
        print("\n✓ Sonar data reaches the policy observation vector in this mode.")
    else:
        print("\n✗ Pipeline issue detected — see FAIL lines above.")
        if headless and not baseline["initialized"]:
            print("  Hint: ensure --enable_cameras and ISAACSIM_EXTENSIONS_FOLDER are set.")
        if baseline["sonar_nz"] < 0.01:
            print("  Hint: sonar image all zeros → check OceanSim ext + sim.render() path.")
        if baseline["valid_n"] < MIN_VALID_FEATURES:
            print("  Hint: reference file may have untrackable points; check station_ref.npz.")
    print("====================================\n")


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.sonar_ui and hasattr(env_cfg, "oceansim_sonar_cfg"):
        env_cfg.oceansim_sonar_cfg.sonar_initialize_viewport = True
    env_cfg.lateral_target_moving = False

    headless = bool(getattr(args_cli, "headless", False))
    print("\n======== Sonar → Obs Pipeline Debug ========")
    print(f"  task      : {args_cli.task}")
    print(f"  headless  : {headless}")
    print(f"  sonar_ui  : {args_cli.sonar_ui}")
    print(f"  num_envs  : {args_cli.num_envs}")
    print("============================================\n")

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    zero_action = torch.zeros(env.action_space.shape, device=u.device)

    obs, _ = env.reset()
    u.set_smooth_target_offset(0.0, 0.0, 0.0)

    last_obs = obs["policy"]
    for step in range(1, args_cli.max_steps + 1):
        obs, _, _, _, _ = env.step(zero_action)
        last_obs = obs["policy"]

    baseline = _print_stats("baseline", env, last_obs, args_cli.max_steps)
    if args_cli.save_png:
        _save_png(env, "baseline")

    u.set_smooth_target_offset(0.0, float(args_cli.target_dy), 0.0)
    for step in range(5):
        obs, _, _, _, _ = env.step(zero_action)
        last_obs = obs["policy"]

    after = _print_stats(f"after_dy={args_cli.target_dy:+.2f}m", env, last_obs, args_cli.max_steps + 5)
    if args_cli.save_png:
        _save_png(env, "after_move")

    _run_checks(baseline, after, headless)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
