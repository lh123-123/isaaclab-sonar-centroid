# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Headless play for Isaac-AUV-Tracking-Direct-v2 (FlowSAC)."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-AUV-Tracking-Direct-v2",
    help="Gym id: Direct-v2 / Direct-v2-image / Direct-v2-state",
)
parser.add_argument("--log_every", type=int, default=40)
parser.add_argument("--dump", type=str, default="/tmp/v2_flow_play_actions.npz")
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent / "lib"))
from auv_play_metrics import format_summary, summarize_play, write_summary_json
from isaaclab_tasks.direct.isaac_auv_tracking.agents.flow_sac_policy import FlowSACPolicy  # noqa: F401
from isaaclab_tasks.direct.isaac_auv_tracking.agents.sonar_encoder import (  # noqa: F401
    CompactSonarEncoder,
    FlowSonarFeaturesExtractor,
)
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_flow_env import (
    AUVTrackingFlowEnvCfg,
    AUVTrackingFlowImageEnvCfg,
    AUVTrackingFlowStateEnvCfg,
)

_TASK_CFG = {
    "Isaac-AUV-Tracking-Direct-v2": AUVTrackingFlowEnvCfg,
    "Isaac-AUV-Tracking-Direct-v2-image": AUVTrackingFlowImageEnvCfg,
    "Isaac-AUV-Tracking-Direct-v2-state": AUVTrackingFlowStateEnvCfg,
}


def main():
    if args_cli.task not in _TASK_CFG:
        raise ValueError(f"unsupported task {args_cli.task!r}; expected one of {sorted(_TASK_CFG)}")
    cfg = _TASK_CFG[args_cli.task]()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.ocean_current.enabled = False
    # High-res sonar: range_res 0.05→0.025, angular_res 2.0→1.0 → 140×121
    cfg.oceansim_sonar_cfg.range_res = 0.025
    cfg.oceansim_sonar_cfg.angular_res = 1.0
    cfg.oceansim_sonar_cfg.height = 140
    cfg.oceansim_sonar_cfg.width = 121
    if not args_cli.headless:
        cfg.sim.render_interval = 4
    env = gym.make(args_cli.task, cfg=cfg)
    vec = Sb3VecEnvWrapper(env, fast_variant=True)
    core = env.unwrapped
    fov_half = float(core._fov_half)
    sensor_range = float(cfg.sensor_range)
    r_star = float(cfg.flow_ref_range_m)

    print(
        f"[play-v2] loading {args_cli.checkpoint} num_envs={vec.num_envs} "
        f"r*={r_star} sensor_range={sensor_range}",
        flush=True,
    )
    agent = SAC.load(args_cli.checkpoint, env=vec, print_system_info=False)

    n_episodes_target = args_cli.episodes
    step_budget = int(args_cli.steps)
    if n_episodes_target is not None:
        step_budget = max(step_budget, int(n_episodes_target) * 250)
        print(f"[play-v2] episode target={n_episodes_target} step_budget={step_budget}", flush=True)

    def _stats():
        lin_w = core._robot.data.root_lin_vel_w.detach()
        auv_xy = torch.linalg.norm(lin_w[:, :2], dim=-1)
        tgt_xy = torch.linalg.norm(core._target_state[:, 2:4], dim=-1)
        dxy = core._target_pos[:, :2] - core._robot.data.root_pos_w[:, :2]
        gt_r = torch.linalg.norm(dxy, dim=-1)
        agent_pos, agent_yaw = core._agent_pose_2d()
        _, _, gt_a = core.kf.get_observation(core._target_pos, agent_pos, agent_yaw)
        in_range = gt_r <= sensor_range
        in_fov = torch.abs(gt_a) <= fov_half
        tot = in_range & in_fov
        return {
            "auv_xy": float(auv_xy.mean()),
            "tgt_xy": float(tgt_xy.mean()),
            "gt_r": float(gt_r.mean()),
            "gt_a": float(gt_a.abs().mean()),
            "tot": float(tot.float().mean()),
            "in_range": float(in_range.float().mean()),
            "r": float(core._flow_r.mean()),
            "a": float(core._flow_a.abs().mean()),
            "r_err": float((core._flow_r - r_star).abs().mean()),
            "valid": float(core._flow_valid.float().mean()),
            "flow_valid": float(core._flow_valid_flow.float().mean()),
            "logdet": float(core._log_det_cov().mean()),
        }

    obs = vec.reset()
    rews, auv_spd, tgt_spd, raw_av, raw_aw, raw_da, gt_range = [], [], [], [], [], [], []
    dones_hist, tot_hist, valid_hist, flow_hist, r_err_hist, gt_a_hist, logdet_hist = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    last_act = None
    ep_rets, ep_lens, ep_tot, ep_valid = [], [], [], []
    cur_ret = np.zeros(vec.num_envs, dtype=np.float64)
    cur_len = np.zeros(vec.num_envs, dtype=np.int32)
    cur_tot = np.zeros(vec.num_envs, dtype=np.float64)
    cur_valid = np.zeros(vec.num_envs, dtype=np.float64)

    for t in range(step_budget):
        actions, _ = agent.predict(obs, deterministic=True)
        actions = np.asarray(actions, dtype=np.float32).reshape(vec.num_envs, -1)
        obs, rewards, dones, infos = vec.step(actions)
        r = np.asarray(rewards, dtype=np.float64).reshape(-1)
        rews.append(float(r.mean()))
        cur_ret += r
        cur_len += 1
        vs = _stats()
        dxy = core._target_pos[:, :2] - core._robot.data.root_pos_w[:, :2]
        gt_r_b = torch.linalg.norm(dxy, dim=-1)
        agent_pos, agent_yaw = core._agent_pose_2d()
        _, _, gt_a_b = core.kf.get_observation(core._target_pos, agent_pos, agent_yaw)
        tot_b = ((gt_r_b <= sensor_range) & (torch.abs(gt_a_b) <= fov_half)).float()
        cur_tot += tot_b.detach().cpu().numpy()
        cur_valid += core._flow_valid.float().detach().cpu().numpy()

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
        valid_hist.append(vs["valid"])
        flow_hist.append(vs["flow_valid"])
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
                ep_valid.append(float(cur_valid[i] / n))
                cur_ret[i] = 0.0
                cur_len[i] = 0
                cur_tot[i] = 0.0
                cur_valid[i] = 0.0
            if n_episodes_target is not None and len(ep_rets) >= n_episodes_target:
                print(f"[play-v2] reached {len(ep_rets)} episodes at step {t}", flush=True)
                break
        if t % args_cli.log_every == 0 or t == step_budget - 1:
            print(
                f"[play-v2] step={t:04d} a=[{a_v:+.3f},{a_w:+.3f}] "
                f"gt_r={vs['gt_r']:.2f} tot={vs['tot']:.2f} "
                f"r_err={vs['r_err']:.2f} valid={vs['valid']:.2f} "
                f"flow={vs['flow_valid']:.2f} rew={rews[-1]:.3f}",
                flush=True,
            )

    av = np.asarray(raw_av)
    aw = np.asarray(raw_aw)
    da = np.asarray(raw_da)
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
        valid=np.asarray(valid_hist),
        flow_valid=np.asarray(flow_hist),
        r_err=np.asarray(r_err_hist),
        gt_a=np.asarray(gt_a_hist),
        logdet=np.asarray(logdet_hist),
        ep_ret=np.asarray(ep_rets, dtype=np.float64),
        ep_len=np.asarray(ep_lens, dtype=np.float64),
        ep_tot=np.asarray(ep_tot, dtype=np.float64),
        ep_valid=np.asarray(ep_valid, dtype=np.float64),
    )
    np.savez(args_cli.dump, **dump_kw)
    print(f"[a-dist] saved {args_cli.dump}", flush=True)

    summary_path = args_cli.summary or str(_Path(args_cli.dump).with_suffix(".json"))
    summary = summarize_play(dump_kw, label=args_cli.tag or args_cli.checkpoint)
    summary["tracking"] = {
        "tot_mean": float(np.mean(tot_hist)) if tot_hist else 0.0,
        "valid_mean": float(np.mean(valid_hist)) if valid_hist else 0.0,
        "flow_valid_mean": float(np.mean(flow_hist)) if flow_hist else 0.0,
        "r_err_mean": float(np.mean(r_err_hist)) if r_err_hist else 0.0,
        "gt_a_mean": float(np.mean(gt_a_hist)) if gt_a_hist else 0.0,
        "logdet_mean": float(np.mean(logdet_hist)) if logdet_hist else 0.0,
        "ep_tot_mean": float(np.mean(ep_tot)) if ep_tot else 0.0,
        "ep_tot_std": float(np.std(ep_tot, ddof=1)) if len(ep_tot) > 1 else 0.0,
        "ep_valid_mean": float(np.mean(ep_valid)) if ep_valid else 0.0,
        "n_episodes": len(ep_rets),
        "checkpoint": args_cli.checkpoint,
        "tag": args_cli.tag,
        "num_envs": int(vec.num_envs),
        "n_steps": len(rews),
        "r_star_m": r_star,
        "sensor_range_m": sensor_range,
        "fov_half_rad": fov_half,
    }
    write_summary_json(summary, summary_path)
    print(format_summary(summary), flush=True)
    print(
        f"[play-v2] tot={summary['tracking']['tot_mean']:.3f} "
        f"valid={summary['tracking']['valid_mean']:.3f} "
        f"flow={summary['tracking']['flow_valid_mean']:.3f} "
        f"r_err={summary['tracking']['r_err_mean']:.3f} "
        f"gt_a={summary['tracking']['gt_a_mean']:.3f} rad "
        f"logdet={summary['tracking']['logdet_mean']:.3f}",
        flush=True,
    )
    if ep_rets:
        print(
            f"[play-v2] ep_ret mean={float(np.mean(ep_rets)):.1f} "
            f"ep_len mean={float(np.mean(ep_lens)):.1f} "
            f"ep_tot mean={float(np.mean(ep_tot)):.3f} n={len(ep_rets)}",
            flush=True,
        )
    print(f"[eval] wrote {summary_path}", flush=True)
    env.close()
    print("[play-v2] DONE", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
