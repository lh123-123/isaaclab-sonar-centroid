# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Finite-step play for Direct-v0 SB3 SAC checkpoints, with raw-action histogram.

Mirrors the metrics printed by play_auv_tracking_drqv2.py and the Holo v0
evaluate() path so the three action distributions are directly comparable.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument(
    "--rew_scale_smooth",
    type=float,
    default=0.0,
    help="Must match training. The shipped v0 SAC checkpoint used 0.0.",
)
parser.add_argument("--dump", type=str, default="/tmp/sb3sac_v0_play_actions.npz")
parser.add_argument(
    "--episodes",
    type=int,
    default=None,
    help="Stop after this many completed episodes. Step budget = max(steps, episodes*250).",
)
parser.add_argument(
    "--summary",
    type=str,
    default=None,
    help="Write episode mean±std JSON here. Default: dump with .json suffix.",
)
parser.add_argument("--tag", type=str, default="", help="Label stored in the summary JSON.")
parser.add_argument(
    "--kf_measurement",
    type=str,
    default="sonar",
    choices=["gt", "sonar"],
    help="KF measurement: geometric GT (no cameras) or OceanSim sonar.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.kf_measurement == "sonar":
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

import math
import sys as _sys
from pathlib import Path as _Path

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_env import AUVTrackingEnvCfg

_sys.path.insert(0, str(_Path(__file__).resolve().parent / "lib"))
from auv_play_metrics import format_summary, summarize_play, write_summary_json


def main():
    cfg = AUVTrackingEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.ocean_current.enabled = False
    cfg.rew_scale_smooth = float(args_cli.rew_scale_smooth)
    cfg.kf_measurement = str(args_cli.kf_measurement)
    if not args_cli.headless:
        cfg.sim.render_interval = 4
    env = gym.make("Isaac-AUV-Tracking-Direct-v0", cfg=cfg)
    vec = Sb3VecEnvWrapper(env, fast_variant=True)
    core = env.unwrapped
    fov_half = 0.5 * math.radians(float(cfg.sensor_fov_deg))
    sensor_range = float(cfg.sensor_range)
    r_star = 2.0

    model = SAC.load(args_cli.checkpoint, device="cuda:0" if torch.cuda.is_available() else "cpu")
    n_episodes_target = args_cli.episodes
    step_budget = int(args_cli.steps)
    if n_episodes_target is not None:
        step_budget = max(step_budget, int(n_episodes_target) * 250)
    print(
        f"[play-sac-v0] loading {args_cli.checkpoint} "
        f"action_scale_v={cfg.action_scale_v} action_scale_w={cfg.action_scale_w} "
        f"target_speed={cfg.target_speed} r*={r_star} episodes={n_episodes_target} "
        f"step_budget={step_budget}",
        flush=True,
    )

    rews, auv_spd, tgt_spd, raw_av, raw_aw, raw_da, gt_range = [], [], [], [], [], [], []
    dones_hist, tot_hist, r_err_hist, gt_a_hist, logdet_hist = [], [], [], [], []
    last_act = None
    ep_rets, ep_lens, ep_tot = [], [], []
    cur_ret = np.zeros(vec.num_envs, dtype=np.float64)
    cur_len = np.zeros(vec.num_envs, dtype=np.int32)
    cur_tot = np.zeros(vec.num_envs, dtype=np.float64)

    def _vel_stats():
        lin_w = core._robot.data.root_lin_vel_w.detach()
        auv_xy = torch.linalg.norm(lin_w[:, :2], dim=-1)
        tgt_xy = torch.linalg.norm(core._target_state[:, 2:4], dim=-1)
        cmd = core._cmd_vel.detach()
        dxy = core._target_pos[:, :2] - core._robot.data.root_pos_w[:, :2]
        gt_r = torch.linalg.norm(dxy, dim=-1)
        agent_pos, agent_yaw = core._agent_pose_2d()
        _, _, gt_a = core.kf.get_observation(core._target_pos, agent_pos, agent_yaw)
        tot = (gt_r <= sensor_range) & (torch.abs(gt_a) <= fov_half)
        return {
            "auv_xy": float(auv_xy.mean()),
            "tgt_xy": float(tgt_xy.mean()),
            "cmd_v": float(cmd[:, 0].mean()),
            "cmd_w": float(cmd[:, 1].mean()),
            "gt_r": float(gt_r.mean()),
            "gt_a": float(gt_a.abs().mean()),
            "tot": float(tot.float().mean()),
            "r_err": float((gt_r - r_star).abs().mean()),
            "logdet": float(core._log_det_cov().mean()),
            "gt_r_b": gt_r,
            "gt_a_b": gt_a,
        }

    obs = vec.reset()
    for t in range(step_budget):
        actions, _ = model.predict(obs, deterministic=True)
        actions = np.asarray(actions, dtype=np.float32).reshape(vec.num_envs, -1)
        obs, rewards, dones, infos = vec.step(actions)
        r = np.asarray(rewards, dtype=np.float64).reshape(-1)
        rews.append(float(r.mean()))
        cur_ret += r
        cur_len += 1
        vs = _vel_stats()
        tot_b = ((vs["gt_r_b"] <= sensor_range) & (torch.abs(vs["gt_a_b"]) <= fov_half)).float()
        cur_tot += tot_b.detach().cpu().numpy()
        a_v = float(np.mean(actions[:, 0]))
        a_w = float(np.mean(actions[:, 1]))
        da = 0.0 if last_act is None else float(np.linalg.norm(actions - last_act, axis=-1).mean())
        last_act = actions.copy()
        raw_av.append(a_v)
        raw_aw.append(a_w)
        raw_da.append(da)
        auv_spd.append(vs["auv_xy"])
        tgt_spd.append(vs["tgt_xy"])
        gt_range.append(vs["gt_r"])
        tot_hist.append(vs["tot"])
        r_err_hist.append(vs["r_err"])
        gt_a_hist.append(vs["gt_a"])
        logdet_hist.append(vs["logdet"])
        done = np.asarray(dones, dtype=bool).reshape(-1)
        dones_hist.append(bool(done.any()) if vec.num_envs == 1 else bool(done[0]))
        if done.any():
            for i in np.where(done)[0]:
                n = max(int(cur_len[i]), 1)
                ep_rets.append(float(cur_ret[i]))
                ep_lens.append(int(cur_len[i]))
                ep_tot.append(float(cur_tot[i] / n))
                cur_ret[i] = 0.0
                cur_len[i] = 0
                cur_tot[i] = 0.0
            if n_episodes_target is not None and len(ep_rets) >= n_episodes_target:
                print(f"[play-sac-v0] reached {len(ep_rets)} episodes at step {t}", flush=True)
                break
        if t % args_cli.log_every == 0 or t == step_budget - 1:
            print(
                f"[play-sac-v0] step={t:04d} a=[{a_v:+.3f},{a_w:+.3f}] da={da:.3f} "
                f"cmd=[{vs['cmd_v']:+.3f},{vs['cmd_w']:+.3f}] "
                f"auv_xy={vs['auv_xy']:.3f} tgt_xy={vs['tgt_xy']:.3f} "
                f"gt_r={vs['gt_r']:.2f} tot={vs['tot']:.2f} r_err={vs['r_err']:.2f} "
                f"gt_a={vs['gt_a']:.2f} rew={rews[-1]:.3f}",
                flush=True,
            )

    av = np.asarray(raw_av)
    aw = np.asarray(raw_aw)
    edges = np.linspace(-1.0, 1.0, 11)

    def _hist(name, x):
        counts, _ = np.histogram(x, bins=edges)
        peak = max(int(counts.max()), 1)
        print(
            f"[a-dist] {name} n={len(x)} mean={float(x.mean()):+.3f} "
            f"|mean|={float(np.mean(np.abs(x))):.3f} std={float(x.std()):.3f} "
            f"frac|a|>0.9={float(np.mean(np.abs(x) > 0.9)):.2f}",
            flush=True,
        )
        for i, c in enumerate(counts):
            print(f"[a-dist] {edges[i]:+.1f}..{edges[i + 1]:+.1f} {c:4d} {'#' * int(round(40 * c / peak))}", flush=True)

    print(f"[play-sac-v0] steps={args_cli.steps} mean_step_rew={float(np.mean(rews)):.3f}", flush=True)
    _hist("a_v", av)
    _hist("a_w", aw)
    da = np.asarray(raw_da)
    print(
        f"[a-change] da mean={float(da.mean()):.3f} p90={float(np.percentile(da, 90)):.3f} "
        f"max={float(da.max()):.3f}  signflip_v={float(np.mean(np.diff(np.sign(av)) != 0)):.2f} "
        f"signflip_w={float(np.mean(np.diff(np.sign(aw)) != 0)):.2f}",
        flush=True,
    )
    dump_kw = dict(
        a_v=av,
        a_w=aw,
        da=da,
        auv_spd=np.asarray(auv_spd),
        tgt_spd=np.asarray(tgt_spd),
        gt_r=np.asarray(gt_range),
        rew=np.asarray(rews),
        done=np.asarray(dones_hist, dtype=np.bool_),
        tot=np.asarray(tot_hist),
        r_err=np.asarray(r_err_hist),
        gt_a=np.asarray(gt_a_hist),
        logdet=np.asarray(logdet_hist),
        ep_ret=np.asarray(ep_rets, dtype=np.float64),
        ep_len=np.asarray(ep_lens, dtype=np.float64),
        ep_tot=np.asarray(ep_tot, dtype=np.float64),
    )
    np.savez(args_cli.dump, **dump_kw)
    print(f"[a-dist] saved {args_cli.dump}", flush=True)
    print(
        f"[play-sac-v0] AUV |v_xy| mean={float(np.mean(auv_spd)):.3f} max={float(np.max(auv_spd)):.3f}  "
        f"TGT |v_xy| mean={float(np.mean(tgt_spd)):.3f} max={float(np.max(tgt_spd)):.3f}  "
        f"gt_r mean={float(np.mean(gt_range)):.2f}",
        flush=True,
    )
    summary_path = args_cli.summary or str(_Path(args_cli.dump).with_suffix(".json"))
    summary = summarize_play(dump_kw, label=args_cli.tag or args_cli.checkpoint)
    summary["tracking"] = {
        "tot_mean": float(np.mean(tot_hist)) if tot_hist else 0.0,
        "valid_mean": float("nan"),
        "flow_valid_mean": float("nan"),
        "r_err_mean": float(np.mean(r_err_hist)) if r_err_hist else 0.0,
        "gt_a_mean": float(np.mean(gt_a_hist)) if gt_a_hist else 0.0,
        "logdet_mean": float(np.mean(logdet_hist)) if logdet_hist else 0.0,
        "ep_tot_mean": float(np.mean(ep_tot)) if ep_tot else 0.0,
        "ep_tot_std": float(np.std(ep_tot, ddof=1)) if len(ep_tot) > 1 else 0.0,
        "ep_valid_mean": float("nan"),
        "n_episodes": len(ep_rets),
        "checkpoint": args_cli.checkpoint,
        "tag": args_cli.tag,
        "num_envs": int(vec.num_envs),
        "n_steps": len(rews),
        "r_star_m": r_star,
        "sensor_range_m": sensor_range,
        "fov_half_rad": fov_half,
        "r_err_note": "v0 r_err = |gt_r - 2.0|; v2 r_err = |centroid_r - 2.0|",
    }
    write_summary_json(summary, summary_path)
    print(format_summary(summary), flush=True)
    print(
        f"[play-sac-v0] tot={summary['tracking']['tot_mean']:.3f} "
        f"r_err={summary['tracking']['r_err_mean']:.3f} "
        f"gt_a={summary['tracking']['gt_a_mean']:.3f} rad "
        f"logdet={summary['tracking']['logdet_mean']:.3f}",
        flush=True,
    )
    if ep_rets:
        print(
            f"[play-sac-v0] ep_ret mean={float(np.mean(ep_rets)):.1f} "
            f"ep_len mean={float(np.mean(ep_lens)):.1f} "
            f"ep_tot mean={float(np.mean(ep_tot)):.3f} n={len(ep_rets)}",
            flush=True,
        )
    print(f"[eval] wrote {summary_path}", flush=True)
    env.close()
    print("[play-sac-v0] DONE", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
