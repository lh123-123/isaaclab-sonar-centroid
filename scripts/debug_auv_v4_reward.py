"""Smoke-test the Direct-v4-cam reward after the 2026-09-28 rebalance.

`scripts/debug_auv_cam_spaces.py` validates the spaces and the policy wiring but never inspects a
reward value. This script does the opposite: it drives the env with scripted actions and checks the
*numbers* the reward produces, because the change was a pure rescale of that signal.

What it verifies
  1. the new ``rew_penalty_clip`` field exists and is read (``<= 0`` means "do not clip");
  2. the reward is finite at every step, and never falls to the ``neginf`` guard;
  3. the per-term decomposition recomputed here matches the total ``step()`` returned -- i.e. the
     env's own arithmetic is right, not just finite;
  4. toggling ``rew_penalty_clip`` between 0.0 and 1.0 on the *same* states reproduces the old
     saturated (flat-gradient) shaping, which is the whole reason for the change;
  5. how often the unbounded costs saturate the historical clip band, and whether the harsher
     reward now trips the Holo done-by-reward rolling-mean test early.

Usage:
    ./scripts/debug_auv_v4_reward.sh --num_envs 4 --headless [--steps 90]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reward smoke test for Direct-v4-cam.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=90, help="Total scripted steps (split over 3 phases).")
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v4-cam")
parser.add_argument("--reward_tol", type=float, default=1e-3, help="Max |total - recomputed| allowed.")
parser.add_argument("--seed", type=int, default=1234,
                    help="Fix the env seed: the default (unset) makes each run a different trajectory, "
                         "which turns any intermittent mismatch into a heisenbug.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
# The env owns a TiledCamera; without this the camera sensor raises at sim.reset().
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Must match AUVTrackingEnvCfg.rew_c_penalty / rew_done_threshold; read at runtime where possible.
FAILURES: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"[reward] {'ok  ' if ok else 'FAIL'} {msg}", flush=True)
    if not ok:
        FAILURES.append(msg)


def as_float(x) -> float:
    return float(x.item()) if isinstance(x, torch.Tensor) else float(x)


def recompute(raw, clip: float, logdet: torch.Tensor, observed: torch.Tensor,
              is_col: torch.Tensor, actions: torch.Tensor, prev_actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Independent re-implementation of AUVTrackingV4CamEnv._compute_step_reward.

    Deliberately duplicated rather than called, so a disagreement means the env and this file
    cannot both be right -- which is the point of the check. `clip > 0` mimics the legacy
    clamp(max=clip) on the quadratic costs.

    Every stateful input is passed in and MUST have been captured while `_compute_step_reward`
    ran, because DirectRLEnv mutates this state after the reward:
      * ``_get_observations`` runs ``kf.predict()``, so ``_log_det_cov()`` has advanced;
      * ``_reset_idx`` sets ``_is_col[env_ids]=False`` and ``_observed``/``_target_pos`` for the
        reset envs, so a post-step read describes the *next* episode.

    Term order mirrors the env exactly, which matters: the collision penalty is applied to the
    base reward inside `super()` (step 3) *before* this override subtracts the geometry costs.
    Applying it last instead is wrong whenever `base > 0` (it clamps the already-reduced value).
    """
    cfg = raw.cfg

    # ---- step 1-2: base AUVTrackingEnv reward terms -----------------------------------------
    r_uncertainty = -logdet
    action_diff = torch.norm(actions - prev_actions, dim=-1)
    r_smooth = torch.exp(-action_diff)
    auv_xy = raw._robot.data.root_lin_vel_w[:, :2]
    tgt_xy = raw._target_state[:, 2:4]
    speed_err = torch.sum((auv_xy - tgt_xy) ** 2, dim=-1)
    sigma_s = max(float(cfg.rew_speed_sigma), 1.0e-6)
    r_speed = torch.exp(-speed_err / (sigma_s * sigma_s))
    base = (
        float(cfg.rew_scale_uncertainty_mean) * r_uncertainty
        + float(cfg.rew_scale_smooth) * r_smooth
        + float(cfg.rew_scale_speed) * r_speed
    )
    base = torch.nan_to_num(base, nan=0.0, posinf=0.0, neginf=-50.0)

    # ---- step 3: collision penalty, inside super() -------------------------------------------
    if is_col.any():
        penalized = torch.minimum(base, torch.zeros_like(base)) - float(cfg.rew_c_penalty)
        base = torch.where(is_col, penalized, base)
    rew = base

    # ---- step 4: AUVTrackingV4CamEnv geometry costs ------------------------------------------
    agent_pos, agent_yaw = raw._agent_pose_2d()
    _, gt_r, gt_a = raw.kf.get_observation(raw._target_pos, agent_pos, agent_yaw)

    range_sigma = max(float(cfg.rew_range_sigma), 1.0e-6)
    range_cost = ((gt_r - float(cfg.rew_range_star)) / range_sigma) ** 2

    bearing_sigma = max(float(cfg.rew_bearing_sigma_frac) * float(raw._fov_half), 1.0e-6)
    bearing_cost = (gt_a / bearing_sigma) ** 2

    action_delta_cost = torch.sum((actions - prev_actions) ** 2, dim=-1)
    lost_cost = (~observed).to(dtype=rew.dtype)

    if clip > 0.0:
        range_cost = torch.clamp(range_cost, max=clip)
        bearing_cost = torch.clamp(bearing_cost, max=clip)
        action_delta_cost = torch.clamp(action_delta_cost, max=clip)

    rew = (
        rew
        - float(cfg.rew_penalty_range) * range_cost
        - float(cfg.rew_penalty_bearing) * bearing_cost
        - float(cfg.rew_penalty_action_delta) * action_delta_cost
        - float(cfg.rew_penalty_lost) * lost_cost
    )

    # ---- step 5: final guard -----------------------------------------------------------------
    rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=-50.0)

    return {
        "base": base,
        "r_unc": r_uncertainty,
        "range_cost": range_cost,
        "bear_cost": bearing_cost,
        "dact_cost": action_delta_cost,
        "lost_cost": lost_cost,
        "gt_r": gt_r,
        "gt_a": gt_a,
        "total": rew,
    }


def install_reward_hook(raw) -> dict:
    """Capture the reward-time state, since DirectRLEnv mutates it after the reward is computed.

    Returns a dict that `_compute_step_reward` re-populates on every call:
      ``logdet``   -- `_log_det_cov()` as the reward saw it (`_get_observations` calls `kf.predict()` later)
      ``observed`` -- `_observed` as the reward saw it
      ``is_col``   -- `_is_col` as the reward saw it (`_reset_idx` clears it for reset envs)
      ``actions`` / ``prev`` -- the action delta the reward saw
      ``rew``      -- what `_compute_step_reward` returned, for cross-checking `step()`'s value
    """
    captured: dict = {}
    original = raw._compute_step_reward

    def hooked() -> torch.Tensor:
        # All of these are pure reads (no state mutation), so sampling them here is safe.
        captured["logdet"] = raw._log_det_cov().detach().clone()
        captured["observed"] = raw._observed.clone()
        captured["is_col"] = raw._is_col.clone()
        captured["actions"] = raw._actions.detach().clone()
        captured["prev"] = raw._prev_actions.detach().clone()
        rew = original()
        captured["rew"] = rew.detach().clone()
        return rew

    raw._compute_step_reward = hooked
    return captured


def phase_actions(raw, phase: str) -> torch.Tensor:
    n = raw.num_envs
    a = torch.zeros(n, 2, device=raw.device)
    if phase == "hold":  # do nothing; the target's leash then cruise drags it away
        return a
    if phase == "chase":  # full ahead, turn toward the target using the true bearing sign
        _, _, gt_a = raw.kf.get_observation(raw._target_pos, *raw._agent_pose_2d())
        a[:, 0] = 1.0
        a[:, 1] = torch.clamp(gt_a * 3.0, -1.0, 1.0)
        return a
    if phase == "random":
        return torch.empty(n, 2, device=raw.device).uniform_(-1.0, 1.0)
    raise ValueError(phase)


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.seed = args_cli.seed

    print("\n[reward] ---- cfg ----", flush=True)
    has_clip = hasattr(env_cfg, "rew_penalty_clip")
    check(has_clip, "AUVTrackingV4CamEnvCfg.rew_penalty_clip exists")
    for f in (
        "rew_scale_uncertainty_mean",
        "rew_penalty_clip",
        "rew_penalty_range",
        "rew_penalty_bearing",
        "rew_penalty_lost",
        "rew_penalty_action_delta",
        "rew_range_star",
        "rew_range_sigma",
        "rew_bearing_sigma_frac",
        "rew_done_threshold",
        "rew_c_penalty",
    ):
        print(f"[reward]   {f:30s} = {getattr(env_cfg, f, '<missing>')}", flush=True)
    print(f"[reward]   refocus scales (expect 0): smooth={env_cfg.rew_scale_smooth} "
          f"speed={env_cfg.rew_scale_speed} lag={getattr(env_cfg, 'rew_scale_lag', 0.0)}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        raw = env.unwrapped
        print(f"[reward] fov_half={np.degrees(float(raw._fov_half)):.2f} deg  "
              f"sensor_range={float(raw.cfg.sensor_range):.2f} m  "
              f"bearing_sigma={np.degrees(float(raw.cfg.rew_bearing_sigma_frac) * float(raw._fov_half)):.2f} deg",
              flush=True)

        obs, _ = env.reset()
        # reset() returns the batched policy entry nested as {'policy': {...}} for the Dict case.
        inner = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
        keys = sorted(inner.keys()) if isinstance(inner, dict) else type(inner).__name__
        print(f"[reward] after reset: obs entry={keys}", flush=True)

        # Must be installed *after* reset (which also computes a reward) and before the rollout.
        captured = install_reward_hook(raw)

        n_phases = 3
        per_phase = max(int(args_cli.steps) // n_phases, 1)
        phases = ["hold"] * per_phase + ["chase"] * per_phase + ["random"] * per_phase

        rows = []
        for t, ph in enumerate(phases):
            actions = phase_actions(raw, ph)
            out = env.step(actions)
            rew_returned = out[1]
            d = recompute(raw, clip=float(raw.cfg.rew_penalty_clip),
                          logdet=captured["logdet"], observed=captured["observed"],
                          is_col=captured["is_col"], actions=captured["actions"],
                          prev_actions=captured["prev"])
            diff = (rew_returned - d["total"]).abs().max()
            # step() must return what _get_dones computed -- EXCEPT on reset steps, where
            # `_reset_idx` zeroes `_cached_rew[env_ids]` in place and `_get_rewards` returns that
            # very tensor (no clone), so the learner sees 0 for the reset envs.
            cached_diff = (rew_returned - captured["rew"]).abs().max()
            # kf.predict() in _get_observations advances the covariance; quantify the gap so the
            # captured-logdet requirement above is measured rather than assumed.
            logdet_gap = float((captured["logdet"] - raw._log_det_cov()).abs().mean())

            finite = bool(torch.isfinite(rew_returned).all())
            # A reset re-randomises target + robot geometry *after* the reward was computed, so the
            # reward is legitimately not reproducible from the post-step state on that step.
            terminated = torch.as_tensor(out[2]).bool() if out[2] is not None else torch.zeros(1, dtype=torch.bool)
            truncated = torch.as_tensor(out[3]).bool() if out[3] is not None else torch.zeros(1, dtype=torch.bool)
            reset_mask = terminated | truncated
            reset_any = bool(reset_mask.any())
            row = {
                "phase": ph,
                "finite": finite,
                "diff": float(diff),
                "env0": as_float(rew_returned[0]),
                "base0": as_float(d["base"][0]),
                "rc0": as_float(d["range_cost"][0]),
                "bc0": as_float(d["bear_cost"][0]),
                "dac0": as_float(d["dact_cost"][0]),
                "lost0": as_float(d["lost_cost"][0]),
                "r0": as_float(d["gt_r"][0]),
                "a0": float(np.degrees(as_float(d["gt_a"][0]))),
                "cached_diff": float(cached_diff),
                "logdet_gap": logdet_gap,
                "reset_any": reset_any,
                # Documented behaviour, not a failure: the terminal reward is zeroed for the envs
                # that reset (Holo does not pay out on the reset step).
                "reset_zero_ok": bool(torch.all(rew_returned[reset_mask] == 0.0)) if reset_any else True,
                "done_any": bool(terminated.any()),
                "t": t,
            }
            rows.append(row)
            # Localise any mismatch immediately: the state moves on, so it cannot be re-derived later.
            if (not reset_any) and float(diff) > args_cli.reward_tol and \
                    sum(r["diff"] > args_cli.reward_tol and not r["reset_any"] for r in rows) <= 5:
                print(f"[reward] DIFF t={t} {ph} max={float(diff):.5f}", flush=True)
                print(f"[reward]   env  total={[round(v, 5) for v in rew_returned.tolist()]}", flush=True)
                print(f"[reward]   mine total={[round(v, 5) for v in d['total'].tolist()]}", flush=True)
                print(f"[reward]   base={[round(v, 5) for v in d['base'].tolist()]} "
                      f"range={[round(v, 5) for v in d['range_cost'].tolist()]} "
                      f"bear={[round(v, 5) for v in d['bear_cost'].tolist()]} "
                      f"dact={[round(v, 5) for v in d['dact_cost'].tolist()]} "
                      f"lost={[round(v, 5) for v in d['lost_cost'].tolist()]}", flush=True)
                print(f"[reward]   is_col={captured['is_col'].tolist()} "
                      f"logdet={[round(v, 4) for v in captured['logdet'].tolist()]} "
                      f"actions={[[round(v, 3) for v in a] for a in captured['actions'].tolist()]} "
                      f"prev={[[round(v, 3) for v in a] for a in captured['prev'].tolist()]}", flush=True)
            if t % 5 == 0 or t == 0 or (len(rows) > 1 and rows[-2]["phase"] != ph):
                print(f"[reward] t={t:03d} {ph:6s} r={row['r0']:5.2f}m a={row['a0']:7.1f}deg "
                      f"base={row['base0']:+7.3f} range={row['rc0']:6.3f} bear={row['bc0']:6.3f} "
                      f"dact={row['dac0']:5.3f} lost={row['lost0']:.0f} "
                      f"-> {row['env0']:+8.3f}", flush=True)

        # Only non-reset steps can be verified: a reset re-randomises the geometry after the reward
        # and zeroes `_cached_rew` for the reset envs.
        checkable = [r for r in rows if not r["reset_any"]]
        max_diff = max((r["diff"] for r in checkable), default=0.0)
        max_cached = max((r["cached_diff"] for r in checkable), default=0.0)
        all_finite = all(r["finite"] for r in rows)
        n_reset = sum(r["reset_any"] for r in rows)
        print()
        print(f"[reward] resets during rollout: {n_reset} step(s) "
              f"(excluded from the decomposition check -- geometry is re-randomised "
              f"after the reward)", flush=True)
        check(all_finite, "every step reward is finite")
        check(max_cached <= args_cli.reward_tol,
              f"step() returns exactly what _compute_step_reward cached, on all {len(checkable)} "
              f"non-reset steps (max |diff| = {max_cached:.2e})")
        check(max_diff <= args_cli.reward_tol,
              f"recomputed decomposition matches step() reward on all {len(checkable)} non-reset steps "
              f"(max |diff| = {max_diff:.2e} <= {args_cli.reward_tol})")
        # Explicitly confirm the aliasing behaviour rather than silently skipping reset steps.
        check(all(r["reset_zero_ok"] for r in rows),
              "on reset steps the returned reward is 0 for the reset envs "
              "(_reset_idx zeroes _cached_rew in place; _get_rewards returns that same tensor)")
        check(not any(r["env0"] <= -50.0 for r in rows),
              "no step hit the nan_to_num neginf (-50) guard")
        # Not a failure, but it is the reason `logdet` has to be captured rather than read here:
        # _get_observations runs kf.predict() after the reward, so a second `_log_det_cov()` sees a
        # different covariance. Reporting it keeps the trap visible.
        mean_gap = float(np.mean([r["logdet_gap"] for r in rows]))
        print(f"[reward] note: mean |log det after reward - after predict| = {mean_gap:.3f} "
              f"(x{float(raw.cfg.rew_scale_uncertainty_mean):g} = "
              f"{mean_gap * float(raw.cfg.rew_scale_uncertainty_mean):.4f}/step) -- "
              f"why the probe captures the reward-time covariance", flush=True)

        # ---- clip behaviour: same states, legacy vs current ---------------------------------
        print("\n[reward] ---- clip=1.0 (legacy / Holo-bounded) vs clip=0.0 (current) ----", flush=True)
        clip_now = float(raw.cfg.rew_penalty_clip)
        # Same captured reward-time state as the rollout, so the two clips are compared on
        # identical geometry (the last step's).
        legacy = recompute(raw, clip=1.0, logdet=captured["logdet"], observed=captured["observed"],
                           is_col=captured["is_col"], actions=captured["actions"],
                           prev_actions=captured["prev"])
        cur = recompute(raw, clip=max(clip_now, 0.0), logdet=captured["logdet"], observed=captured["observed"],
                        is_col=captured["is_col"], actions=captured["actions"],
                        prev_actions=captured["prev"])
        sat_r = float((legacy["range_cost"] >= 0.999).float().mean())
        sat_b = float((legacy["bear_cost"] >= 0.999).float().mean())
        gap = (legacy["total"] - cur["total"])
        print(f"[reward]   saturation of the old band on the final state: range={sat_r:.1%} bearing={sat_b:.1%}", flush=True)
        print(f"[reward]   mean(legacy - current) = {as_float(gap.mean()):+.3f} per step "
              f"(positive => legacy was flatter/rewarding off-target)", flush=True)
        # Clipping only ever *raises* the reward (costs are subtracted), so this must hold.
        check(as_float(gap.min()) >= -1e-6, "clip=1.0 is never harsher than clip=0.0 on the same state")

        # Off-target magnitude that the legacy clip removed the gradient for:
        rs, bs = float(raw.cfg.rew_range_sigma), float(raw.cfg.rew_bearing_sigma_frac) * float(raw._fov_half)
        for err_r in (0.6, 1.2, 4.0):
            cost = (err_r / rs) ** 2
            print(f"[reward]   range err +{err_r:.1f} m -> cost {cost:6.2f} -> "
                  f"current {float(raw.cfg.rew_penalty_range) * cost:7.3f}/step, "
                  f"legacy {float(raw.cfg.rew_penalty_range) * min(cost, 1.0):7.3f}/step", flush=True)
        for err_a in (20.0, 45.0, 90.0):
            cost = (np.radians(err_a) / bs) ** 2
            print(f"[reward]   bearing err {err_a:5.1f} deg -> cost {cost:6.2f} -> "
                  f"current {float(raw.cfg.rew_penalty_bearing) * cost:7.3f}/step, "
                  f"legacy {float(raw.cfg.rew_penalty_bearing) * min(cost, 1.0):7.3f}/step", flush=True)

        # ---- done-by-reward pressure --------------------------------------------------------
        print("\n[reward] ---- done-by-reward (Holo: rolling-100 mean < "
              f"{float(raw.cfg.rew_done_threshold)}) ----", flush=True)
        by_phase: dict[str, list[float]] = {}
        for r in rows:
            by_phase.setdefault(r["phase"], []).append(r["env0"])
        for ph, vals in by_phase.items():
            v = np.asarray(vals)
            n_done = sum(1 for r in rows if r["phase"] == ph and r["reset_any"])
            print(f"[reward]   {ph:6s} mean={v.mean():+7.3f}/step  min={v.min():+8.3f}  "
                  f"episode-mean would be {v.mean() * 199:+8.1f}  done_flags={n_done}", flush=True)
        if n_reset:
            # Not a hard failure -- Holo's mechanism is meant to fire on sustained bad tracking --
            # but on a 90-step smoke test it means the reward is harsher than expected.
            print("[reward] WARN done-by-reward fired during the probe; watch ep_len in the first 20k steps",
                  flush=True)
        else:
            check(True, "done-by-reward did not fire during the probe")

        print()
        if FAILURES:
            print(f"[reward] RESULT: FAIL ({len(FAILURES)}): " + "; ".join(FAILURES), flush=True)
            return 1
        print("[reward] RESULT: OK", flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    import traceback

    try:
        rc = main()
    except BaseException:  # noqa: BLE001 - must surface the failure immediately
        traceback.print_exc()
        sys.stderr.flush()
        # Same convention as debug_auv_cam_spaces.py: SimulationApp.shutdown() can block for
        # minutes after a mid-run exception, which wastes a GPU slot and looks like a hang.
        os._exit(1)
    simulation_app.close()
    os._exit(rc)
