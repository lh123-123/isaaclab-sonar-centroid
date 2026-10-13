#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Headless play-eval for Isaac-AUV-Tracking-Direct-v4-cam (CustomSAC + MidCNN).

Reports episode return plus tracking geometry (GT range / |bearing| / lost) so the
overnight pipeline can recommend reward-weight changes.

  ./scripts/eval_auv_tracking_v4_cam.sh --checkpoint PATH --episodes 20 --headless
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval Direct-v4-cam SAC checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v4-cam")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--steps", type=int, default=None, help="Hard step cap (default: episodes*250).")
parser.add_argument("--seed", type=int, default=66)
parser.add_argument(
    "--flow_source",
    type=str,
    default=None,
    choices=["raft", "gt"],
    help="Override env flow_source (v5-flow only): 'raft' = RAFT-Small estimated flow, "
    "'gt' = analytic disk stamp ablation baseline. Default: use env cfg default.",
)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--summary", type=str, default="", help="JSON summary path.")
parser.add_argument("--dump", type=str, default="", help="Optional npz of per-step arrays.")
parser.add_argument(
    "--pace_hz",
    type=float,
    default=0.0,
    help=(
        "Cap the env-step rate at this many Hz by sleeping (0 = off, the headless default). "
        "GUI-only: the loop otherwise runs flat out and starves the viewport, which makes the "
        "window unresponsive. Does not change physics or the numbers reported."
    ),
)
parser.add_argument(
    "--render_interval",
    type=int,
    default=None,
    help=(
        "Override sim.render_interval (GUI only). Training used 24 with sim.dt=1/120, i.e. one "
        "render per env step, so the viewport can only update at 5 Hz no matter how fast the "
        "policy runs -- that is what makes the window look laggy. A divisor of decimation "
        "(e.g. 8 or 4) renders more often for smoother motion while the substep at the env-step "
        "boundary still renders, so the observation the policy sees is unchanged."
    ),
)
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
from isaaclab_tasks.direct.isaac_auv_tracking.agents.cam_encoder import MidCNNExtractor  # noqa: F401
from isaaclab_tasks.direct.isaac_auv_tracking.agents.custom_sac_policy import CustomSACPolicy  # noqa: F401
from isaaclab_tasks.direct.isaac_auv_tracking.sonar_meas import body_polar_from_xy
from isaaclab_tasks.utils import parse_env_cfg


def _gt_polar(raw) -> tuple[float, float, bool]:
    agent_xy, yaw = raw._agent_pose_2d()
    a_t, r_t = body_polar_from_xy(raw._target_pos, agent_xy, yaw)
    r = float(r_t[0].item())
    a = float(a_t[0].item())
    fov_half = float(raw._fov_half.item()) if hasattr(raw._fov_half, "item") else float(raw._fov_half)
    sensor_range = float(getattr(raw.cfg, "sensor_range", 4.0))
    observed = bool(getattr(raw, "_observed", torch.zeros(1, dtype=torch.bool))[0].item())
    in_fov = bool(r <= sensor_range and abs(a) <= fov_half)
    return r, a, observed or in_fov


def main() -> None:
    ckpt = Path(args_cli.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt.resolve())

    n_ep = int(args_cli.episodes)
    step_budget = int(args_cli.steps) if args_cli.steps is not None else max(400, n_ep * 250)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.flow_source is not None and hasattr(env_cfg, "flow_source"):
        env_cfg.flow_source = str(args_cli.flow_source)
        print(f"[eval_v4] flow_source override -> {env_cfg.flow_source}", flush=True)
    if args_cli.render_interval is not None:
        ri = int(args_cli.render_interval)
        dec = int(getattr(env_cfg, "decimation", ri))
        if ri <= 0:
            raise ValueError(f"--render_interval must be positive, got {ri}")
        if dec % ri != 0:
            print(
                f"[eval_v4] WARNING: render_interval={ri} does not divide decimation={dec}. The last "
                "render before an env step may fall on a different physics substep than in training, "
                "so the observation can differ. Prefer a divisor of decimation.",
                flush=True,
            )
        else:
            hz = 1.0 / (ri * float(env_cfg.sim.dt))
            print(
                f"[eval_v4] render_interval {env_cfg.sim.render_interval} -> {ri} "
                f"({hz:.0f} Hz viewport; env-step render unchanged)",
                flush=True,
            )
        env_cfg.sim.render_interval = ri
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    vec = Sb3VecEnvWrapper(env, fast_variant=True)
    # The 12-channel stack is not an SB3 image (channel check only allows 1/3/4),
    # so the saved Box comes back as float32 while the live space stays uint8.
    # Same bounds and shape; the encoder still accepts uint8 frames.
    image_space = vec.observation_space.spaces["image"]
    if image_space.dtype != np.float32:
        vec.observation_space.spaces["image"] = gym.spaces.Box(
            low=np.zeros(image_space.shape, dtype=np.float32),
            high=np.full(image_space.shape, 255.0, dtype=np.float32),
            dtype=np.float32,
        )
    print(f"[eval_v4] loading {ckpt.resolve()}", flush=True)
    agent = SAC.load(str(ckpt), vec, print_system_info=False)
    obs = vec.reset()

    out_dir = ckpt.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args_cli.summary) if str(args_cli.summary).strip() else out_dir / f"eval_{ckpt.stem}.json"
    dump_path = Path(args_cli.dump) if str(args_cli.dump).strip() else out_dir / f"eval_{ckpt.stem}.npz"

    rews: list[float] = []
    abs_a: list[float] = []
    gt_r: list[float] = []
    lost: list[float] = []
    s_ratio: list[float] = []  # R8 尺寸比 s = cue_w / teach_w（仅 v5-flow 系有 _size_ratio）
    a_v_hist: list[float] = []
    a_w_hist: list[float] = []
    da_hist: list[float] = []
    v_hist: list[float] = []
    yawrate_hist: list[float] = []
    dones_hist: list[bool] = []
    last_act = None

    ep_rets: list[float] = []
    ep_lens: list[int] = []
    ep_med_abs_a: list[float] = []
    ep_mean_r: list[float] = []
    ep_frac_r_gt3: list[float] = []
    ep_lost: list[float] = []
    ep_med_s: list[float] = []
    cur_ret = 0.0
    cur_len = 0
    cur_abs_a: list[float] = []
    cur_r: list[float] = []
    cur_lost: list[float] = []
    cur_s: list[float] = []

    t0 = time.perf_counter()
    pace_dt = 1.0 / float(args_cli.pace_hz) if float(args_cli.pace_hz) > 0.0 else 0.0
    if pace_dt > 0.0:
        print(f"[eval_v4] pacing env steps to {args_cli.pace_hz:g} Hz (GUI mode)", flush=True)
    for t in range(step_budget):
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
        actions = np.asarray(actions, dtype=np.float32).reshape(vec.num_envs, -1)
        with torch.inference_mode():
            obs, rewards, dones, _infos = vec.step(actions)

        # GUI-only pacing: keep the viewport responsive by not letting the step loop saturate
        # the renderer. No effect on physics or on any reported metric.
        if pace_dt > 0.0:
            sleep_for = (t0 + (t + 1) * pace_dt) - time.perf_counter()
            if sleep_for > 0.0:
                time.sleep(sleep_for)

        r = float(np.asarray(rewards).reshape(-1)[0])
        done = bool(np.asarray(dones).reshape(-1)[0])
        gt_range, gt_az, vis = _gt_polar(raw)
        abs_az = abs(math.degrees(gt_az))
        is_lost = 0.0 if vis else 1.0
        # R8 尺寸比：env 内部 s = cue_w / teach_w（teach 无效时 s=1 中性）。别的任务没有该
        # 属性，记 NaN，汇总时用 NaN-safe 统计，判线只对 v5-flow 系生效。
        if hasattr(raw, "_size_ratio"):
            s_now = float(torch.as_tensor(raw._size_ratio).reshape(-1)[0].item())
        else:
            s_now = float("nan")

        rews.append(r)
        abs_a.append(abs_az)
        gt_r.append(gt_range)
        lost.append(is_lost)
        s_ratio.append(s_now)
        a_v_hist.append(float(actions[0, 0]))
        a_w_hist.append(float(actions[0, 1]))
        # R9 运动分析：实测机体前向速度（env._ego_vel_state() 返回 (N,2) [v_surge, yaw_rate]；
        # 无该方法的任务记 NaN）。
        if hasattr(raw, "_ego_vel_state"):
            _ego = raw._ego_vel_state()
            v_hist.append(float(_ego[0, 0].item()))
            yawrate_hist.append(float(_ego[0, 1].item()))
        else:
            v_hist.append(float("nan"))
            yawrate_hist.append(float("nan"))
        da = 0.0 if last_act is None else float(np.linalg.norm(actions - last_act))
        da_hist.append(da)
        last_act = actions.copy()
        dones_hist.append(done)

        cur_ret += r
        cur_len += 1
        cur_abs_a.append(abs_az)
        cur_r.append(gt_range)
        cur_lost.append(is_lost)
        cur_s.append(s_now)

        if done:
            ep_rets.append(cur_ret)
            ep_lens.append(cur_len)
            ep_med_abs_a.append(float(np.median(cur_abs_a)))
            ep_mean_r.append(float(np.mean(cur_r)))
            ep_frac_r_gt3.append(float(np.mean(np.asarray(cur_r) > 3.0)))
            ep_lost.append(float(np.mean(cur_lost)))
            ep_med_s.append(float(np.nanmedian(cur_s)) if cur_s else float("nan"))
            cur_ret = 0.0
            cur_len = 0
            cur_abs_a, cur_r, cur_lost, cur_s = [], [], [], []
            if len(ep_rets) >= n_ep:
                print(f"[eval_v4] reached {len(ep_rets)} episodes at step={t}", flush=True)
                break

        if t % max(int(args_cli.log_every), 1) == 0:
            s_str = "" if math.isnan(s_now) else f" s={s_now:.2f}"
            print(
                f"[eval_v4] step={t:04d} rew={r:+.3f} gt_r={gt_range:.2f} "
                f"|α|={abs_az:.1f}° lost={is_lost:.0f}{s_str} ep={len(ep_rets)}/{n_ep}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    summary = {
        "checkpoint": str(ckpt.resolve()),
        "task": args_cli.task,
        "kf_measurement": getattr(raw.cfg, "kf_measurement", None),
        "n_steps": len(rews),
        "n_episodes": len(ep_rets),
        "elapsed_s": round(elapsed, 1),
        "step_rew_mean": float(np.mean(rews)) if rews else float("nan"),
        "ep_ret_mean": float(np.mean(ep_rets)) if ep_rets else float("nan"),
        "ep_ret_std": float(np.std(ep_rets, ddof=1)) if len(ep_rets) > 1 else 0.0,
        "ep_len_mean": float(np.mean(ep_lens)) if ep_lens else float("nan"),
        "med_abs_bearing_deg": float(np.median(ep_med_abs_a)) if ep_med_abs_a else float(np.median(abs_a) if abs_a else float("nan")),
        "mean_abs_bearing_deg": float(np.mean(abs_a)) if abs_a else float("nan"),
        "mean_gt_r": float(np.mean(gt_r)) if gt_r else float("nan"),
        "frac_r_gt3": float(np.mean(np.asarray(gt_r) > 3.0)) if gt_r else float("nan"),
        "lost_rate": float(np.mean(lost)) if lost else float("nan"),
        # R8 判线新指标：尺寸比中位 ≈ 1（NaN = 任务无 _size_ratio，如 v4-cam/v6）
        "med_size_ratio": float(np.nanmedian(ep_med_s)) if ep_med_s else float("nan"),
        "da_mean": float(np.mean(da_hist)) if da_hist else float("nan"),
        "da_p90": float(np.percentile(da_hist, 90)) if da_hist else float("nan"),
        "a_w_sat": float(np.mean(np.abs(np.asarray(a_w_hist)) > 0.9)) if a_w_hist else float("nan"),
        "reward_cfg": {
            "rew_penalty_range": float(getattr(raw.cfg, "rew_penalty_range", 0.0)),
            "rew_penalty_bearing": float(getattr(raw.cfg, "rew_penalty_bearing", 0.0)),
            "rew_penalty_action_delta": float(getattr(raw.cfg, "rew_penalty_action_delta", 0.0)),
            "rew_penalty_lost": float(getattr(raw.cfg, "rew_penalty_lost", 0.0)),
        },
        "episodes": [
            {
                "ep_ret": ep_rets[i],
                "ep_len": ep_lens[i],
                "med_abs_bearing_deg": ep_med_abs_a[i],
                "mean_gt_r": ep_mean_r[i],
                "frac_r_gt3": ep_frac_r_gt3[i],
                "lost_rate": ep_lost[i],
                "med_size_ratio": ep_med_s[i],
            }
            for i in range(len(ep_rets))
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(
        dump_path,
        rew=np.asarray(rews),
        abs_bearing_deg=np.asarray(abs_a),
        gt_r=np.asarray(gt_r),
        lost=np.asarray(lost),
        size_ratio=np.asarray(s_ratio),
        a_v=np.asarray(a_v_hist),
        a_w=np.asarray(a_w_hist),
        v_surge=np.asarray(v_hist),
        yaw_rate=np.asarray(yawrate_hist),
        da=np.asarray(da_hist),
        done=np.asarray(dones_hist, dtype=np.bool_),
    )
    print(
        f"[eval_v4] ep_ret={summary['ep_ret_mean']:.1f}±{summary['ep_ret_std']:.1f} "
        f"med_|α|={summary['med_abs_bearing_deg']:.1f}° mean_r={summary['mean_gt_r']:.2f} "
        f"frac_r>3={summary['frac_r_gt3']:.1%} lost={summary['lost_rate']:.1%} "
        f"da_p90={summary['da_p90']:.3f}",
        flush=True,
    )
    if not math.isnan(summary["med_size_ratio"]):
        print(f"[eval_v4] med_size_ratio={summary['med_size_ratio']:.3f} (判线≈1)", flush=True)
    print(f"[eval_v4] wrote {summary_path.resolve()}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
