#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Simulate Lateral1D reward: offline edge cases + in-env rollout statistics.

Usage:
  python scripts/debug_reward_sim.py --offline          # no Isaac Sim
  python scripts/debug_reward_sim.py --headless --enable_cameras
  python scripts/debug_reward_sim.py --headless --enable_cameras --action_mode random
"""

from __future__ import annotations

import argparse
import sys


def _offline_sim() -> bool:
    """Synthetic reward checks without Isaac Sim."""
    import torch

    from isaaclab_tasks.direct.sonar_flow_hover.modules.rewards import compute_reward_lateral_1d_terms

    class _FakeFT:
        def __init__(self, ex, valid_n, diagonal=181.0):
            self.num_envs = 1
            self.max_track = 20
            self.diagonal = diagonal
            self._ex = ex
            self._valid_n = valid_n

        def feature_errors(self):
            err = torch.zeros(1, 20, 2)
            err[0, 0, 0] = self._ex
            return err

        def valid_count(self):
            return torch.tensor([self._valid_n])

        @property
        def valid(self):
            v = torch.zeros(1, 20, dtype=torch.bool)
            v[0, : self._valid_n] = True
            return v

    w1, w2, w3 = 2.0, 0.05, 1.0
    depth = torch.tensor([10.0])
    checks: list[tuple[str, bool, str]] = []

    def _run(label: str, ex: float, valid_n: int, sway: float, d: float = 10.0) -> dict:
        terms = compute_reward_lateral_1d_terms(
            feature_tracker=_FakeFT(ex, valid_n),
            actions=torch.tensor([[sway]]),
            depth=torch.tensor([d]),
            w1_pixel=w1,
            w2_action=w2,
            w3_safety=w3,
            min_depth=1.0,
            max_depth=20.0,
            min_valid_features=2,
        )
        t = {k: float(v[0].item()) for k, v in terms.items()}
        print(
            f"  [{label}] track={t['term_pixel']:+.3f} action={t['term_action']:+.3f} "
            f"safety={t['term_safety']:+.3f} total={t['total']:+.3f} "
            f"(e_lat={t['pixel_error']:.3f} valid={int(t['valid_n'])})",
            flush=True,
        )
        return t

    print("\n======== Offline Reward Simulation ========", flush=True)
    t_ok = _run("good track", ex=0.02, valid_n=2, sway=0.2)
    t_bad = _run("poor track", ex=0.10, valid_n=2, sway=0.8)
    t_lost = _run("lost features", ex=0.05, valid_n=1, sway=0.0)
    t_zero = _run("perfect", ex=0.0, valid_n=2, sway=0.0)

    checks.append(
        (
            "tracking dominates when safe",
            abs(t_ok["term_pixel"]) > abs(t_ok["term_action"]) and t_ok["term_safety"] == 0.0,
            f"|track|={abs(t_ok['term_pixel']):.3f} vs |action|={abs(t_ok['term_action']):.3f}",
        )
    )
    checks.append(
        (
            "safety not catastrophic (w3=1)",
            t_lost["term_safety"] == -1.0 and t_lost["total"] > -2.0,
            f"lost-feature total={t_lost['total']:+.3f} (old w3=5 would be ~-5)",
        )
    )
    checks.append(
        (
            "better track => higher reward",
            t_zero["total"] > t_bad["total"],
            f"perfect={t_zero['total']:+.3f} poor={t_bad['total']:+.3f}",
        )
    )
    checks.append(
        (
            "reward always <= 0",
            all(x["total"] <= 0.0 for x in (t_ok, t_bad, t_lost, t_zero)),
            "penalty-only design",
        )
    )

    print("\n--- Offline checks ---", flush=True)
    all_ok = True
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}: {detail}", flush=True)
        all_ok = all_ok and ok
    print("=========================================\n", flush=True)
    return all_ok


def _env_rollout(args_cli) -> bool:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.direct.sonar_flow_hover.modules.rewards import compute_reward_lateral_1d_terms
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    print("\n======== In-Env Reward Rollout ========", flush=True)
    print(f"  task         : {args_cli.task}", flush=True)
    print(f"  action_mode  : {args_cli.action_mode}", flush=True)
    print(f"  steps        : {args_cli.max_steps}", flush=True)
    print(f"  w1/w2/w3     : {env_cfg.rew_w1_pixel}/{env_cfg.rew_w2_action}/{env_cfg.rew_w3_safety}", flush=True)
    print("======================================\n", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    totals: list[float] = []
    tracks: list[float] = []
    actions: list[float] = []
    safeties: list[float] = []
    safety_hits = 0
    terminates = 0

    obs, _ = env.reset()
    for step in range(args_cli.max_steps):
        if args_cli.action_mode == "zero":
            act = torch.zeros(env.action_space.shape, device=unwrapped.device)
        else:
            act = torch.empty(env.action_space.shape, device=unwrapped.device).uniform_(-1.0, 1.0)

        obs, reward, terminated, truncated, _ = env.step(act)
        terms = compute_reward_lateral_1d_terms(
            feature_tracker=unwrapped.feature_tracker,
            actions=unwrapped.actions,
            depth=unwrapped.robot.data.root_pos_w[:, 2],
            w1_pixel=float(env_cfg.rew_w1_pixel),
            w2_action=float(env_cfg.rew_w2_action),
            w3_safety=float(env_cfg.rew_w3_safety),
            min_depth=float(env_cfg.min_depth),
            max_depth=float(env_cfg.max_depth),
            min_valid_features=int(env_cfg.min_valid_features),
        )
        t = float(terms["total"][0].item())
        totals.append(t)
        tracks.append(float(terms["term_pixel"][0].item()))
        actions.append(float(terms["term_action"][0].item()))
        s = float(terms["term_safety"][0].item())
        safeties.append(s)
        if s < 0:
            safety_hits += 1
        if bool((terminated | truncated)[0].item()):
            terminates += 1
            obs, _ = env.reset()

        if step % max(args_cli.interval, 1) == 0:
            print(
                f"  step={step:4d} total={t:+.4f} track={tracks[-1]:+.4f} "
                f"action={actions[-1]:+.4f} safety={s:+.4f} "
                f"e_lat={float(terms['pixel_error'][0].item()):.4f} "
                f"valid={int(terms['valid_n'][0].item())}",
                flush=True,
            )

    def _stats(xs: list[float]) -> str:
        if not xs:
            return "n=0"
        return f"min={min(xs):+.3f} mean={sum(xs)/len(xs):+.3f} max={max(xs):+.3f}"

    safety_frac = safety_hits / max(len(totals), 1)
    track_dom = sum(1 for i in range(len(totals)) if abs(tracks[i]) >= abs(actions[i]) and safeties[i] == 0) / max(
        len(totals), 1
    )

    checks: list[tuple[str, bool, str]] = [
        ("samples collected", len(totals) > 10, f"n={len(totals)}"),
        ("total reward stats", max(totals) <= 0.01, _stats(totals)),
        ("tracking term active", min(tracks) < -0.001, _stats(tracks)),
        (
            "safety rare when tracking ok",
            safety_frac < 0.15,
            f"safety hits={safety_hits}/{len(totals)} ({100*safety_frac:.1f}%)",
        ),
        (
            "tracking dominates most safe steps",
            track_dom > 0.5,
            f"{100*track_dom:.1f}% steps |track|>=|action| and safety=0",
        ),
        (
            "no extreme spikes (<-3)",
            min(totals) > -3.0,
            f"min total={min(totals):+.3f}",
        ),
        (
            "episode terminations reasonable",
            terminates < args_cli.max_steps * 0.5,
            f"resets={terminates}",
        ),
    ]

    print("\n--- Rollout summary ---", flush=True)
    print(f"  total   : {_stats(totals)}", flush=True)
    print(f"  track   : {_stats(tracks)}", flush=True)
    print(f"  action  : {_stats(actions)}", flush=True)
    print(f"  safety  : {_stats(safeties)}  hits={safety_hits}", flush=True)

    print("\n======== REWARD VALIDITY ========", flush=True)
    all_ok = True
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}: {detail}", flush=True)
        all_ok = all_ok and ok
    if all_ok:
        print("\n✓ Reward simulation looks healthy for 1D training.", flush=True)
    else:
        print("\n✗ Reward issues detected — review FAIL lines.", flush=True)
    print("================================\n", flush=True)

    env.close()
    simulation_app.close()
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Lateral1D reward.")
    parser.add_argument("--offline", action="store_true", help="Run synthetic tests only.")
    parser.add_argument("--task", type=str, default="Isaac-SonarIbvsDrl-Lateral1D-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--action_mode", choices=("zero", "random"), default="zero")
    parser.add_argument("--disable_fabric", action="store_true", default=False)

    if "--offline" in sys.argv:
        parser.parse_args()
        ok = _offline_sim()
        sys.exit(0 if ok else 1)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.enable_cameras = True
    ok = _env_rollout(args_cli)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
