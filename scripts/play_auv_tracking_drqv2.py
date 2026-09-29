# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Finite-step play for Direct-v1 DrQ-v2 checkpoints (.pt) with sonar frame stack."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--feature_dim", type=int, default=50)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument(
    "--frame_stack",
    type=int,
    default=0,
    help="0 = read from checkpoint meta (fallback 3).",
)
parser.add_argument(
    "--rew_scale_smooth",
    type=float,
    default=0.0,
    help="Must match training. 1.0 for calm+smooth DrQ runs.",
)
parser.add_argument(
    "--action_scale_v",
    type=float,
    default=-1.0,
    help="Override cfg.action_scale_v (cmd = a_v * scale). <0 keeps cfg (1.5).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.direct.isaac_auv_tracking.agents.drqv2 import DrQV2Agent, SonarFrameStack
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_vision_env import AUVTrackingVisionEnvCfg


def _policy_images_from_obs(obs) -> np.ndarray:
    if isinstance(obs, dict):
        img = obs["image"]
    else:
        img = obs
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def main():
    cfg = AUVTrackingVisionEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.ocean_current.enabled = False
    cfg.rew_scale_smooth = float(args_cli.rew_scale_smooth)
    cfg.target_soft_start_s = 3.0
    cfg.target_easy_s = 5.0
    if args_cli.action_scale_v >= 0.0:
        cfg.action_scale_v = float(args_cli.action_scale_v)
    # Training uses render_interval=decimation=24 (~5 viewport FPS). Drop it for GUI.
    if not args_cli.headless:
        cfg.sim.render_interval = 4
    env = gym.make("Isaac-AUV-Tracking-Direct-v1", cfg=cfg)
    vec = Sb3VecEnvWrapper(env, fast_variant=True)

    # Peek checkpoint for frame_stack / obs_shape before building agent.
    ckpt = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    frame_stack = int(args_cli.frame_stack) if args_cli.frame_stack > 0 else int(ckpt.get("frame_stack", 3))
    obs_shape = tuple(ckpt.get("obs_shape", (frame_stack, 64, 64)))
    if obs_shape[0] != frame_stack:
        frame_stack = int(obs_shape[0])

    action_dim = int(np.prod(vec.action_space.shape))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent = DrQV2Agent(
        obs_shape=obs_shape,
        action_shape=(action_dim,),
        device=device,
        feature_dim=args_cli.feature_dim,
        hidden_dim=args_cli.hidden_dim,
        use_tb=False,
    )
    print(
        f"[play-drqv2] loading {args_cli.checkpoint} frame_stack={frame_stack} "
        f"obs_shape={obs_shape} action_scale_v={cfg.action_scale_v} "
        f"action_scale_w={cfg.action_scale_w}",
        flush=True,
    )
    agent.load(args_cli.checkpoint)
    agent.train(False)

    stacker = SonarFrameStack(num_envs=args_cli.num_envs, frame_stack=frame_stack, height=64, width=64)
    obs = vec.reset()
    stacked = stacker.reset(_policy_images_from_obs(obs))
    core = env.unwrapped

    rews = []
    ep_rets = []
    ep_lens = []
    auv_spd = []
    tgt_spd = []
    auv_cmd_v = []
    auv_cmd_w = []
    body_vx = []
    body_wz = []
    raw_av = []
    raw_aw = []
    pwm_drive = []
    gt_range = []
    logdets = []
    last_act = None
    cur_ret = np.zeros(vec.num_envs, dtype=np.float64)
    cur_len = np.zeros(vec.num_envs, dtype=np.int32)

    def _cov_stats():
        """Current KF P after step (predict already applied)."""
        logdet = core._log_det_cov().detach()
        diag = torch.diagonal(core.kf.cov, dim1=1, dim2=2).detach().clamp(min=0.0)
        obs_frac = float(core._observed.float().mean())
        return (
            float(logdet.mean()),
            float(logdet.min()),
            float(logdet.max()),
            float(diag[:, 0].sqrt().mean()),
            float(diag[:, 1].sqrt().mean()),
            float(diag[:, 2].sqrt().mean()),
            obs_frac,
        )

    def _vel_stats():
        # Chain: a ∈[-1,1] → cmd = a*scale → PID → body vx / world |v_xy|
        lin_w = core._robot.data.root_lin_vel_w.detach()
        lin_b = core._robot.data.root_lin_vel_b.detach()
        ang_b = core._robot.data.root_ang_vel_b.detach()
        auv_xy = torch.linalg.norm(lin_w[:, :2], dim=-1)
        tgt_xy = torch.linalg.norm(core._target_state[:, 2:4], dim=-1)
        cmd = core._cmd_vel.detach()
        pwm = core._pwm.detach()
        dxy = core._target_pos[:, :2] - core._robot.data.root_pos_w[:, :2]
        rng = torch.linalg.norm(dxy, dim=-1)
        return {
            "auv_xy": float(auv_xy.mean()),
            "tgt_xy": float(tgt_xy.mean()),
            "cmd_v": float(cmd[:, 0].mean()),
            "cmd_w": float(cmd[:, 1].mean()),
            "body_vx": float(lin_b[:, 0].mean()),
            "body_vy": float(lin_b[:, 1].mean()),
            "body_vz": float(lin_b[:, 2].mean()),
            "body_wz": float(ang_b[:, 2].mean()),
            "pwm01": float(pwm[:, :2].abs().mean()),
            "gt_r": float(rng.mean()),
        }

    for t in range(args_cli.steps):
        actions = agent.act(stacked, step=10**9, eval_mode=True)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        obs, rewards, dones, infos = vec.step(actions)
        raw = _policy_images_from_obs(obs)
        stacked = stacker.update(raw)
        done = np.asarray(dones, dtype=bool).reshape(-1)
        if done.any():
            stacker.on_dones(raw, done)
            stacked = stacker.get()
        r = np.asarray(rewards, dtype=np.float64).reshape(-1)
        rews.append(float(r.mean()))
        cur_ret += r
        cur_len += 1
        vs = _vel_stats()
        a_v = float(np.mean(actions[:, 0]))
        a_w = float(np.mean(actions[:, 1]))
        da = 0.0 if last_act is None else float(np.linalg.norm(actions - last_act, axis=-1).mean())
        last_act = np.asarray(actions, dtype=np.float64).copy()
        auv_spd.append(vs["auv_xy"])
        tgt_spd.append(vs["tgt_xy"])
        auv_cmd_v.append(vs["cmd_v"])
        auv_cmd_w.append(vs["cmd_w"])
        body_vx.append(vs["body_vx"])
        body_wz.append(vs["body_wz"])
        raw_av.append(a_v)
        raw_aw.append(a_w)
        pwm_drive.append(vs["pwm01"])
        gt_range.append(vs["gt_r"])
        ld, ld_min, ld_max, sx, sy, svx, obs_frac = _cov_stats()
        logdets.append(ld)
        if done.any():
            for i in np.where(done)[0]:
                ep_rets.append(float(cur_ret[i]))
                ep_lens.append(int(cur_len[i]))
                cur_ret[i] = 0.0
                cur_len[i] = 0
        if t % args_cli.log_every == 0 or t == args_cli.steps - 1:
            print(
                f"[spd] step={t:04d} a=[{a_v:+.3f},{a_w:+.3f}] da={da:.3f} "
                f"cmd=[{vs['cmd_v']:+.3f},{vs['cmd_w']:+.3f}] "
                f"body_v=[{vs['body_vx']:+.3f},{vs['body_vy']:+.3f},{vs['body_vz']:+.3f}] "
                f"wz={vs['body_wz']:+.3f} world_xy={vs['auv_xy']:.3f} "
                f"pwm01={vs['pwm01']:.2f} tgt={vs['tgt_xy']:.3f} gt_r={vs['gt_r']:.2f} "
                f"rew={rews[-1]:.3f} obs={obs_frac:.2f}",
                flush=True,
            )

    print(
        f"[play-drqv2] steps={args_cli.steps} mean_step_rew={float(np.mean(rews)):.3f} "
        f"n_episodes={len(ep_rets)}",
        flush=True,
    )
    av = np.asarray(raw_av)
    aw = np.asarray(raw_aw)

    def _hist(name: str, x: np.ndarray, edges: np.ndarray) -> None:
        counts, _ = np.histogram(x, bins=edges)
        width = 40
        peak = max(int(counts.max()), 1)
        print(f"[a-dist] {name} n={len(x)} mean={float(x.mean()):+.3f} "
              f"|mean|={float(np.mean(np.abs(x))):.3f} std={float(x.std()):.3f}", flush=True)
        for i, c in enumerate(counts):
            bar = "#" * int(round(width * c / peak))
            print(f"[a-dist] {edges[i]:+.1f}..{edges[i+1]:+.1f} {c:4d} {bar}", flush=True)

    edges = np.linspace(-1.0, 1.0, 11)
    _hist("a_v", av, edges)
    _hist("a_w", aw, edges)
    sat_v = float(np.mean(np.abs(av) > 0.9))
    mid_v = float(np.mean(np.abs(av) < 0.4))
    print(
        f"[a-dist] a_v sat(|a|>0.9)={sat_v:.2f} mid(|a|<0.4)={mid_v:.2f} "
        f"sign+={float(np.mean(av > 0)):.2f}",
        flush=True,
    )
    dump = "/tmp/drqv2_play_actions.npz"
    np.savez(dump, a_v=av, a_w=aw)
    print(f"[a-dist] saved {dump}", flush=True)

    cv = np.asarray(auv_cmd_v)
    bvx = np.asarray(body_vx)
    wxy = np.asarray(auv_spd)
    print(
        f"[spd-sum] |a_v| mean={float(np.mean(np.abs(av))):.3f} p90={float(np.percentile(np.abs(av), 90)):.3f} "
        f"sat(|a_v|>0.9)={float(np.mean(np.abs(av) > 0.9)):.2f} "
        f"|a_w| mean={float(np.mean(np.abs(raw_aw))):.3f}",
        flush=True,
    )
    print(
        f"[spd-sum] cmd_v mean={float(np.mean(cv)):+.3f} |cmd_v| mean={float(np.mean(np.abs(cv))):.3f} "
        f"max={float(np.max(np.abs(cv))):.3f} sat(|cmd|>=scale-0.05)="
        f"{float(np.mean(np.abs(cv) >= (core.cfg.action_scale_v - 0.05))):.2f}",
        flush=True,
    )
    print(
        f"[spd-sum] body_vx mean={float(np.mean(bvx)):+.3f} |body_vx| mean={float(np.mean(np.abs(bvx))):.3f} "
        f"max={float(np.max(np.abs(bvx))):.3f}  track_err |cmd-vx| mean="
        f"{float(np.mean(np.abs(cv - bvx))):.3f}",
        flush=True,
    )
    print(
        f"[spd-sum] world |v_xy| mean={float(np.mean(wxy)):.3f} "
        f"p50={float(np.percentile(wxy, 50)):.3f} p90={float(np.percentile(wxy, 90)):.3f} "
        f"max={float(np.max(wxy)):.3f}  pwm01 mean={float(np.mean(pwm_drive)):.2f} "
        f"gt_r mean={float(np.mean(gt_range)):.2f}",
        flush=True,
    )
    print(
        f"[play-drqv2] AUV |v_xy| mean={float(np.mean(auv_spd)):.3f} "
        f"max={float(np.max(auv_spd)):.3f}  cmd_v mean={float(np.mean(auv_cmd_v)):.3f} "
        f"max={float(np.max(np.abs(auv_cmd_v))):.3f}",
        flush=True,
    )
    print(
        f"[play-drqv2] cov logdet mean={float(np.mean(logdets)):.2f} "
        f"min={float(np.min(logdets)):.2f} max={float(np.max(logdets)):.2f} "
        f"(init P=50I → logdet≈{4 * np.log(50.0):.1f})",
        flush=True,
    )
    print(
        f"[play-drqv2] TGT |v_xy| mean={float(np.mean(tgt_spd)):.3f} "
        f"max={float(np.max(tgt_spd)):.3f}  (cfg cruise={core.cfg.target_speed} "
        f"max={core.cfg.target_max_speed})",
        flush=True,
    )
    if ep_rets:
        print(
            f"[play-drqv2] ep_ret mean={float(np.mean(ep_rets)):.1f} "
            f"min={float(np.min(ep_rets)):.1f} max={float(np.max(ep_rets)):.1f} "
            f"ep_len mean={float(np.mean(ep_lens)):.1f}",
            flush=True,
        )
    else:
        print(
            f"[play-drqv2] no full episode; partial_ret={float(cur_ret.mean()):.1f} "
            f"len={int(cur_len.mean())}",
            flush=True,
        )
    env.close()
    print("[play-drqv2] DONE", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
