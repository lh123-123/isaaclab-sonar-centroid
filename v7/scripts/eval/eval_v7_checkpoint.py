"""Evaluate a v7 flow-hover checkpoint: station-keeping error per episode.

Reports the two numbers that matter and that the training reward cannot tell apart:
  * disp_spawn_m : horizontal displacement from THIS episode's spawn at episode end
  * peak_m       : worst horizontal displacement reached at any point in the episode

Why hooking ``_reset_idx``: the env resets an env inside ``step()``, teleporting the root
back to the spawn. Reading the position after ``env.step()`` returns would therefore read
~0 for every episode. The hook runs BEFORE the parent reset, so it sees the true terminal
pose, and it also lets us pair the value with that episode's DR draw and termination cause.

Usage
  ./isaaclab.sh -p v7/scripts/eval/eval_v7_checkpoint.py \
      --task Isaac-AUV-Tracking-Direct-v7-flow-hover \
      --checkpoint logs/sb3/.../model_960000_steps.zip \
      --episodes 20 --num_envs 4 --seed 1234 --headless
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a v7 flow-hover checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v7-flow-hover")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=20, help="total episodes across all envs")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument(
    "--redraw_per_run_dr",
    type=int,
    default=1,
    help="1 = redraw the per-run hardware DR knobs every episode (independent samples, "
    "honest CIs). 0 = one hardware draw per env for the whole run (the old protocol, kept "
    "only for reproducing older numbers).",
)
parser.add_argument("--algorithm", type=str, default="SAC", choices=["PPO", "SAC"])
parser.add_argument(
    "--action_delay",
    type=int,
    default=0,
    help="apply the action N control steps late (= N x 66.7 ms of actuator-side dead time). "
    "Models the sensor->compute->link->thruster delay of the real vehicle without touching the "
    "env: the policy sees a fresh image but its command reaches the motors N steps later.",
)
parser.add_argument("--out", type=str, default="/tmp/v7_eval.json", help="where to dump the per-episode JSON")
parser.add_argument(
    "--plane_distance",
    type=float,
    default=None,
    help="override cfg.plane_distance (m). The textured wall is generated from this value and the "
    "estimator's Z(0) is initialized from it, so this is a genuine standoff change, not a "
    "scale hack. K150k was trained at 2.0 m; use this to probe other standoffs.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True  # the policy observes RGB + flow
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    # A delay-randomized policy carries its own per-episode lag. Zero it here so the only
    # dead time in an evaluation is --action_delay, which is what the curve is indexed on;
    # leaving both on would stack two independent delays and make the result unreadable.
    if hasattr(env_cfg, "train_action_delay_max_steps"):
        env_cfg.train_action_delay_max_steps = 0
    if hasattr(env_cfg, "train_action_delay_fixed_steps"):
        env_cfg.train_action_delay_fixed_steps = -1
    # Evaluation protocol: redraw the per-run hardware knobs every episode so each episode
    # is an independent sample of the unit distribution. Without this, N episodes on 4 envs
    # share 4 hardware draws and the confidence intervals are optimistic (measured spread on
    # identical weights: est_err median 9 mm on env1 vs 38 mm on env3).
    if hasattr(env_cfg, "eval_redraw_per_run_dr"):
        env_cfg.eval_redraw_per_run_dr = bool(int(args_cli.redraw_per_run_dr))
    if args_cli.plane_distance is not None:
        env_cfg.plane_distance = float(args_cli.plane_distance)
        print("[eval][GEOM] plane_distance override -> %.3f m (wall + estimator Z0 move together)"
              % env_cfg.plane_distance)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    core = gym_env.unwrapped
    while hasattr(core, "unwrapped") and core.unwrapped is not core:
        core = core.unwrapped

    # ---- reward banner: never evaluate a checkpoint without knowing its reward config ----
    c = core.cfg
    print(
        "[eval][REW] sigma_flow=%.3f  sigma_anchor_vis=%.3f m  sigma_visodom=%.3f m"
        "  sigma_visodom_obs=%.3f m | rew flow/anchor_vis/sim/yaw/pos = %.2f/%.2f/%.2f/%.2f/%.2f"
        % (
            float(c.sigma_flow), float(c.sigma_anchor_vis_m), float(c.sigma_visodom_m),
            float(getattr(c, "sigma_visodom_obs_m", 1.0)),
            float(c.rew_flow), float(c.rew_anchor_vis), float(c.rew_sim),
            float(c.rew_yaw_anchor), float(c.rew_anchor),
        ),
        flush=True,
    )
    print(
        "[eval][REW] fine twins: anchor_vis_fine=%.2f@%.3fm  position_fine=%.2f@%.3fm"
        % (
            float(getattr(c, "rew_anchor_vis_fine", 0.0)),
            float(getattr(c, "sigma_anchor_vis_fine_m", 0.0)),
            float(getattr(c, "rew_position_fine", 0.0)),
            float(getattr(c, "sigma_visodom_fine_m", 0.0)),
        ),
        flush=True,
    )

    n = int(core.num_envs)
    peak = torch.zeros(n, device=core.device)
    # Per-step trajectory buffers. The endpoint metric alone is gameable: a policy that
    # circles inside a 10-20 cm cloud and happens to stop near the origin reports the same
    # 44 mm as one that never moved (measured 2026-11-19: path 2.7 m in 15 s, late drift
    # 180 mm/s). These buffers feed rms / path / late-drift / estimate-error.
    t_max = int(core.max_episode_length) + 2
    dt_step = float(core.cfg.sim.dt) * float(core.cfg.decimation)
    buf_xy = torch.zeros((n, t_max, 2), device=core.device)
    buf_est = torch.zeros((n, t_max), device=core.device)
    # The env resets inside env.step(), so the final control step is never sampled: its row
    # stays (0,0) and then reads back as "distance 0, estimate 0". That fabricated a zero
    # est_err on every episode and injected a spurious jump-to-origin into path/rms/drift
    # (caught 2026-11-19). Track the actual recorded length per env instead of trusting
    # episode_length_buf.
    rec_len = torch.zeros(n, dtype=torch.long, device=core.device)
    late_window = max(int(round(5.0 / dt_step)), 2)  # ~5 s, the "did it settle" horizon
    records: list[dict] = []
    orig_reset_idx = core._reset_idx

    def hooked_reset_idx(env_ids):
        if env_ids is None:
            ids = torch.arange(n, device=core.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=core.device, dtype=torch.long).view(-1)
        if ids.numel() > 0:
            # Read BEFORE the parent reset teleports the roots.
            d = torch.norm(
                core._robot.data.root_pos_w[ids, :2] - core._spawn_xy[ids], dim=-1
            )
            cause = core._ep_cause[ids] if hasattr(core, "_ep_cause") else None
            for k, j in enumerate(ids.tolist()):
                # The central reset() at startup also routes through _reset_idx with
                # episode_length_buf == 0 for every env. Those are not episodes; recording
                # them injected four all-zero rows and dragged the mean down (caught 11-16).
                if int(core.episode_length_buf[j]) == 0:
                    continue
                # Trajectory-derived metrics for THIS episode, from the per-step buffers.
                L = int(rec_len[j])
                if L >= 2:
                    xy = buf_xy[j, :L]
                    dist = torch.linalg.norm(xy, dim=-1)
                    rms_m = float(torch.sqrt(torch.mean(dist ** 2)))
                    path_m = float(torch.linalg.norm(torch.diff(xy, dim=0), dim=-1).sum())
                    k_win = min(L, late_window)
                    late_drift = (
                        float(torch.linalg.norm(torch.diff(xy[-k_win:], dim=0), dim=-1).mean()) / dt_step
                    )
                    est_err = abs(float(buf_est[j, L - 1]) - float(dist[-1]))
                else:
                    rms_m = float(d[k])
                    path_m = 0.0
                    late_drift = 0.0
                    est_err = float("nan")
                rec = {
                    "env": j,
                    "disp_spawn_m": float(d[k]),
                    "peak_m": float(torch.maximum(peak[j], d[k])),
                    "rms_m": rms_m,
                    "path_m": path_m,
                    "late_drift_mps": late_drift,
                    "est_err_m": est_err,
                    "t": int(core.episode_length_buf[j]),
                    "dr_cur_speed": float(core._dr_cur_speed[j]) if hasattr(core, "_dr_cur_speed") else float("nan"),
                    "dr_cur_deg": float(np.degrees(np.arctan2(
                        float(core._dr_cur_dir[j, 1]), float(core._dr_cur_dir[j, 0])
                    ))) if hasattr(core, "_dr_cur_dir") else float("nan"),
                    "anchor_valid_frac": float(core._ep_anchor_valid_sum[j] / max(int(core.episode_length_buf[j]), 1)),
                    # Hardware ("per-run") knobs. With --redraw_per_run_dr 1 these are
                    # redrawn every episode, so they are per-episode covariates and can be
                    # regressed against the outcome; with the old protocol they would be
                    # identical for 25 consecutive episodes and would look like a policy
                    # difference. Needed to test "drift fails at low thrust gain x high
                    # current" rather than assume it.
                    "dr_thrust_gain": float(core._dr_thrust_gain[j])
                    if hasattr(core, "_dr_thrust_gain")
                    else float("nan"),
                    "dr_drag_gain": float(core._dr_drag_gain[j])
                    if hasattr(core, "_dr_drag_gain")
                    else float("nan"),
                    "dr_cam_noise": float(core._dr_cam_noise[j])
                    if hasattr(core, "_dr_cam_noise")
                    else float("nan"),
                    "dr_imu_yaw_bias_deg": float(np.degrees(float(core._imu_yaw_bias[j])))
                    if hasattr(core, "_imu_yaw_bias")
                    else float("nan"),
                }
                if cause is not None:
                    rec["term_tex"] = bool(cause[k, 0])
                    rec["term_depth"] = bool(cause[k, 1])
                    rec["term_att"] = bool(cause[k, 2])
                    rec["term_timeout"] = bool(cause[k, 3])
                records.append(rec)
            peak[ids] = 0.0
            rec_len[ids] = 0
        return orig_reset_idx(env_ids)

    core._reset_idx = hooked_reset_idx

    agent_cfg = process_sb3_cfg(agent_cfg, n)
    for _k in ("normalize_input", "normalize_value", "clip_obs"):
        agent_cfg.pop(_k, None)
    env = Sb3VecEnvWrapper(gym_env, fast_variant=True)

    print(f"[eval] loading {args_cli.checkpoint}", flush=True)
    # buffer_size is overridden to a token value: SAC.load() rebuilds the replay buffer at the
    # size recorded in the training cfg (120k × a 180 KB image observation ≈ 43 GB), which OOMs
    # on anything smaller than the training host. Evaluation only ever calls predict(), so the
    # buffer is dead weight; capacity is not part of the saved weights and does not affect the
    # policy.
    agent = {"PPO": PPO, "SAC": SAC}[args_cli.algorithm.upper()].load(
        args_cli.checkpoint, env, print_system_info=False, buffer_size=256
    )

    obs = env.reset()
    # One warm-up step so the first textured frame is captured before the first real step.
    max_steps = int(args_cli.episodes / n + 2) * int(core.max_episode_length) + 10
    delay = max(0, int(args_cli.action_delay))
    pending: deque = deque(maxlen=delay + 1)
    step = 0
    while len(records) < args_cli.episodes and step < max_steps and simulation_app.is_running():
        with torch.inference_mode():
            action, _ = agent.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            if len(pending) == 0:
                # Prime the pipeline with standstill commands so the first `delay` steps do not
                # apply a phantom action; after this the deque always holds delay+1 entries and
                # pending[0] is the command issued `delay` steps ago.
                pending.extend([np.zeros_like(action)] * delay)
            pending.append(action)
            obs, _rew, _dones, _infos = env.step(pending[0])
        d = torch.norm(core._robot.data.root_pos_w[:, :2] - core._spawn_xy, dim=-1)
        peak = torch.maximum(peak, d.detach())
        # Per-step trajectory. Index by episode_length_buf so an env that reset inside
        # env.step() reports idx -1 and is simply skipped for that step.
        xy_now = (core._robot.data.root_pos_w[:, :2] - core._spawn_xy).detach()
        est_now = core._visodom_m().norm(dim=-1).detach()
        ep_now = core.episode_length_buf.detach().cpu().tolist()
        for j in range(n):
            idx = int(ep_now[j]) - 1
            if 0 <= idx < t_max:
                buf_xy[j, idx] = xy_now[j]
                buf_est[j, idx] = est_now[j]
                if idx + 1 > int(rec_len[j]):
                    rec_len[j] = idx + 1
        step += 1

    env.close()

    records = records[: args_cli.episodes]
    if not records:
        print("[eval] no episodes completed", flush=True)
        return

    disp = np.array([r["disp_spawn_m"] for r in records])
    pk = np.array([r["peak_m"] for r in records])
    spd = np.array([r["dr_cur_speed"] for r in records])

    print("\n[eval] ==== per-episode ====", flush=True)
    print(
        "[eval]  #  end       rms      peak     path     drift    est_err  term      dr_spd  len",
        flush=True,
    )
    for i, r in enumerate(records):
        if r.get("term_timeout"):
            term = "timeout"
        elif r.get("term_tex"):
            term = "tex_lost"
        elif r.get("term_depth"):
            term = "depth"
        elif r.get("term_att"):
            term = "att"
        else:
            term = "?"
        print(
            "[eval] %3d %7.1f  %7.1f  %7.1f  %7.1f  %7.1f  %7.1f  %-9s %6.3f  %3d"
            % (
                i, r["disp_spawn_m"] * 1e3, r.get("rms_m", float("nan")) * 1e3,
                r["peak_m"] * 1e3, r.get("path_m", float("nan")) * 1e3,
                r.get("late_drift_mps", float("nan")) * 1e3,
                r.get("est_err_m", float("nan")) * 1e3, term, r["dr_cur_speed"], r["t"],
            ),
            flush=True,
        )

    def _st(x):
        return (float(np.mean(x)), float(np.median(x)), float(np.percentile(x, 90)), float(np.max(x)))

    m, md, p90, mx = _st(disp)
    print("\n[eval] ==== summary (%d episodes) ====" % len(records), flush=True)
    print("[eval] disp_spawn_m  mean=%.1f  median=%.1f  p90=%.1f  max=%.1f  (mm)" % (
        m * 1e3, md * 1e3, p90 * 1e3, mx * 1e3), flush=True)
    m2, md2, p902, mx2 = _st(pk)
    print("[eval] peak_m       mean=%.1f  median=%.1f  p90=%.1f  max=%.1f  (mm)" % (
        m2 * 1e3, md2 * 1e3, p902 * 1e3, mx2 * 1e3), flush=True)

    # ---- hover-quality block: the endpoint number alone is gameable (see module docstring
    # of plot_v7_hover_trajectory.py). A policy that circles inside a 10-20 cm cloud and
    # stops near the origin reports the same endpoint as one that never moved.
    rms = np.array([r.get("rms_m", np.nan) for r in records])
    pth = np.array([r.get("path_m", np.nan) for r in records])
    dfr = np.array([r.get("late_drift_mps", np.nan) for r in records])
    err = np.array([r.get("est_err_m", np.nan) for r in records])
    r3, r5, r9 = _st(rms)[:3]
    t3, t5, t9 = _st(pth)[:3]
    f3, f5, f9 = _st(dfr)[:3]
    e3, e5, e9 = _st(err)[:3]
    print("[eval] rms_m        mean=%.1f  median=%.1f  p90=%.1f  (mm)" % (r3 * 1e3, r5 * 1e3, r9 * 1e3), flush=True)
    print("[eval] path_m       mean=%.1f  median=%.1f  p90=%.1f  (mm over the episode)" % (t3 * 1e3, t5 * 1e3, t9 * 1e3), flush=True)
    print("[eval] late_drift   mean=%.1f  median=%.1f  p90=%.1f  (mm/s over the last 5 s)" % (
        f3 * 1e3, f5 * 1e3, f9 * 1e3), flush=True)
    print("[eval] est_err      mean=%.1f  median=%.1f  p90=%.1f  (mm, |visodom - truth| at end)" % (
        e3 * 1e3, e5 * 1e3, e9 * 1e3), flush=True)
    print(
        "[eval] hoverscore   end<=50mm %d/%d | rms<=50 %d/%d | peak<=100 %d/%d | drift<=20mm/s %d/%d"
        % (
            int(np.sum(disp * 1e3 <= 50.0)), len(records),
            int(np.nansum(rms * 1e3 <= 50.0)), len(records),
            int(np.sum(pk * 1e3 <= 100.0)), len(records),
            int(np.nansum(dfr * 1e3 <= 20.0)), len(records),
        ),
        flush=True,
    )
    print("[eval] dr_speed     mean=%.3f  min=%.3f  max=%.3f (m/s)" % (
        float(np.mean(spd)), float(np.min(spd)), float(np.max(spd))), flush=True)
    to = sum(1 for r in records if r.get("term_timeout"))
    print("[eval] outcomes     timeout=%d/%d  early=%d" % (to, len(records), len(records) - to), flush=True)
    print("[eval] reference    LQR 3.3-4.9 mm | PID 15.9-22.3 mm | P 42.6-63.3 mm", flush=True)
    print("[eval] action_delay %d step(s) = %.0f ms of dead time" % (
        delay, delay * 66.7), flush=True)

    # ---- formal gate -------------------------------------------------------------------
    # Median-first, pass-rate-second. At 40 episodes the per-episode pass fraction moves by
    # +-2 episodes between *identical* runs: the same K150k weights scored 80% on end<=50 mm
    # in one run and 75% in the next (2026-11-19), flipping the verdict on sampling noise
    # alone. The median is stable across re-runs, so it is the REQUIRED criterion; the 80%
    # pass rate is reported with a Wilson 95% CI and marked INCONCLUSIVE when the sample is
    # too small to distinguish it from the 80% target.
    def _wilson(k_ok: int, n: int, z: float = 1.96):
        if n <= 0:
            return 0.0, 0.0
        p = k_ok / n
        denom = 1.0 + z * z / n
        centre = p + z * z / (2 * n)
        spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom)

    gate_defs = [
        ("end<=50mm", disp * 1e3, 50.0, "mm"),
        ("rms<=50mm", rms * 1e3, 50.0, "mm"),
        ("peak<=100mm", pk * 1e3, 100.0, "mm"),
        ("drift<=20mm/s", dfr * 1e3, 20.0, "mm/s"),
    ]
    print("\n[eval] ==== GATE (median must be inside the bar) ====", flush=True)
    gate_detail = {}
    verdicts = []
    for name, vals, bar, unit in gate_defs:
        finite = vals[np.isfinite(vals)]
        n = int(finite.size)
        med = float(np.median(finite)) if n else float("nan")
        k_ok = int(np.sum(finite <= bar))
        frac = (k_ok / n) if n else 0.0
        lo, hi = _wilson(k_ok, n)
        if not (med <= bar):
            verdict = "FAIL"
        elif lo >= 0.8:
            verdict = "PASS"
        elif hi < 0.8:
            verdict = "FAIL"
        else:
            verdict = "INCONCLUSIVE"
        verdicts.append(verdict)
        gate_detail[name] = {
            "median": med,
            "bar": bar,
            "pass_frac": frac,
            "pass_ci95": [lo, hi],
            "verdict": verdict,
        }
        print(
            "[eval]   %-14s median=%8.1f %-4s bar=%5.1f | pass %3d/%3d=%4.0f%% "
            "(95%%CI %3.0f-%3.0f%%) | %s"
            % (name, med, unit, bar, k_ok, n, 100 * frac, 100 * lo, 100 * hi, verdict),
            flush=True,
        )
    if any(v == "FAIL" for v in verdicts):
        gate_result = "FAIL"
    elif all(v == "PASS" for v in verdicts):
        gate_result = "PASS"
    else:
        gate_result = "INCONCLUSIVE"
        print(
            "[eval] note: %d bar(s) inconclusive -- rerun with more episodes (>=100) or a "
            "second seed before reading this as pass/fail." % verdicts.count("INCONCLUSIVE"),
            flush=True,
        )
    print("[eval] GATE RESULT: %s" % gate_result, flush=True)

    with open(args_cli.out, "w") as f:
        json.dump(
            {
                "checkpoint": args_cli.checkpoint,
                "seed": args_cli.seed,
                "eval_redraw_per_run_dr": bool(int(args_cli.redraw_per_run_dr)),
                "action_delay_steps": delay,
                "action_delay_ms": delay * 66.7,
                "episodes": records,
                "summary": {
                    "disp_spawn_m_mean": m, "disp_spawn_m_median": md,
                    "disp_spawn_m_p90": p90, "disp_spawn_m_max": mx,
                    "peak_m_mean": m2, "peak_m_p90": p902, "peak_m_max": mx2,
                    "rms_m_mean": r3, "rms_m_median": r5, "rms_m_p90": r9,
                    "path_m_mean": t3, "path_m_median": t5, "path_m_p90": t9,
                    "late_drift_mps_mean": f3, "late_drift_mps_median": f5,
                    "late_drift_mps_p90": f9,
                    "est_err_m_mean": e3, "est_err_m_median": e5, "est_err_m_p90": e9,
                    "pass_end_50mm": int(np.sum(disp * 1e3 <= 50.0)),
                    "pass_rms_50mm": int(np.nansum(rms * 1e3 <= 50.0)),
                    "pass_peak_100mm": int(np.sum(pk * 1e3 <= 100.0)),
                    "pass_drift_20mms": int(np.nansum(dfr * 1e3 <= 20.0)),
                    "gate_frac": {
                        k: v["pass_frac"] for k, v in gate_detail.items()
                    },
                    "gate": gate_detail,
                    "gate_pass": gate_result == "PASS",
                    "gate_result": gate_result,
                    "n_episodes": len(records),
                },
                "reward_cfg": {
                    "sigma_flow": float(c.sigma_flow),
                    "sigma_anchor_vis_m": float(c.sigma_anchor_vis_m),
                    "sigma_visodom_m": float(c.sigma_visodom_m),
                    "sigma_visodom_obs_m": float(getattr(c, "sigma_visodom_obs_m", 1.0)),
                },
            },
            f,
            indent=2,
        )
    print(f"[eval] wrote {args_cli.out}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
