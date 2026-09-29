# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Train DrQ-v2 on Isaac-AUV-Tracking-Direct-v1 (sonar polar + temporal frame stack)."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DrQ-v2 on AUV Tracking Direct-v1")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--total_steps", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=66)
parser.add_argument("--buffer_size", type=int, default=50_000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--nstep", type=int, default=3)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--feature_dim", type=int, default=50)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--learning_starts", type=int, default=1000)
parser.add_argument("--update_every_steps", type=int, default=2)
parser.add_argument("--num_expl_steps", type=int, default=2000)
parser.add_argument("--stddev_schedule", type=str, default="linear(1.0,0.1,500000)")
parser.add_argument(
    "--frame_stack",
    type=int,
    default=3,
    help="Temporal sonar frames stacked on channel axis (DrQ-style; default 3).",
)
parser.add_argument("--log_every", type=int, default=2000)
parser.add_argument("--ckpt_every", type=int, default=50_000)
parser.add_argument("--log_dir", type=str, default="")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="",
    help="Resume weights from checkpoint_*.pt (watchdog / manual continue).",
)
parser.add_argument(
    "--rew_scale_smooth",
    type=float,
    default=0.0,
    help="Reward weight on exp(-||a-a_prev||). 1.0 matches Holo c_smooth yaml.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.direct.isaac_auv_tracking.agents.drqv2 import (
    DrQV2Agent,
    NStepReplayBuffer,
    SonarFrameStack,
)
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_vision_env import AUVTrackingVisionEnvCfg


def _policy_images_from_obs(obs) -> np.ndarray:
    """Raw Direct-v1 policy image (N,3,64,64) uint8."""
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
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    cfg = AUVTrackingVisionEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.ocean_current.enabled = False
    cfg.rew_scale_smooth = float(args_cli.rew_scale_smooth)
    # Cold-start schedule + lobe-center init (already on cfg; keep explicit).
    cfg.target_soft_start_s = 3.0
    cfg.target_easy_s = 5.0
    env = gym.make("Isaac-AUV-Tracking-Direct-v1", cfg=cfg)
    vec = Sb3VecEnvWrapper(env, fast_variant=True)

    h, w = 64, 64
    stacker = SonarFrameStack(
        num_envs=args_cli.num_envs,
        frame_stack=args_cli.frame_stack,
        height=h,
        width=w,
    )
    obs_shape = stacker.obs_shape  # (K, 64, 64)
    action_dim = int(np.prod(vec.action_space.shape))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    agent = DrQV2Agent(
        obs_shape=obs_shape,
        action_shape=(action_dim,),
        device=device,
        lr=args_cli.lr,
        feature_dim=args_cli.feature_dim,
        hidden_dim=args_cli.hidden_dim,
        num_expl_steps=args_cli.num_expl_steps,
        update_every_steps=args_cli.update_every_steps,
        stddev_schedule=args_cli.stddev_schedule,
        use_tb=True,
    )
    replay = NStepReplayBuffer(
        capacity=args_cli.buffer_size,
        obs_shape=obs_shape,
        action_dim=action_dim,
        batch_size=args_cli.batch_size,
        nstep=args_cli.nstep,
        gamma=args_cli.gamma,
        num_envs=args_cli.num_envs,
    )

    global_step = 0
    if args_cli.checkpoint:
        ckpt_path = Path(args_cli.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--checkpoint not found: {ckpt_path}")
        loaded = agent.load(str(ckpt_path))
        global_step = int(loaded.get("global_step", 0))
        if global_step <= 0:
            # Fallback: checkpoint_50000.pt → 50000
            stem = ckpt_path.stem
            if stem.startswith("checkpoint_") and stem.split("_", 1)[1].isdigit():
                global_step = int(stem.split("_", 1)[1])
        print(f"[drqv2] resumed {ckpt_path} @ step={global_step}", flush=True)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(args_cli.log_dir) if args_cli.log_dir else Path(f"logs/drqv2/Isaac-AUV-Tracking-Direct-v1/{stamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "meta.txt").write_text(
        f"frame_stack={args_cli.frame_stack}\nobs_shape={obs_shape}\n"
        f"nstep={args_cli.nstep}\nnum_envs={args_cli.num_envs}\n"
        f"resume_from={args_cli.checkpoint or 'none'}\n"
        f"start_step={global_step}\n"
        f"ocean_current=off\n"
        f"rew_scale_smooth={args_cli.rew_scale_smooth}\n"
        f"cold_start=soft_start_{cfg.target_soft_start_s}s easy_{cfg.target_easy_s}s "
        f"a2t=[{cfg.target_init_dist_min},{cfg.target_init_dist_max}]x"
        f"+/-{cfg.target_init_ang_max:.3f}\n"
        "note=sonar polar channel-0 temporal stack (not RGBxK)\n",
        encoding="utf-8",
    )
    writer = SummaryWriter(str(log_dir / "tb"))
    print(f"[drqv2] log_dir={log_dir.resolve()}", flush=True)
    print(
        f"[drqv2] frame_stack={args_cli.frame_stack} obs_shape={obs_shape} "
        f"encoder.repr_dim={agent.encoder.repr_dim} num_envs={args_cli.num_envs}",
        flush=True,
    )

    obs = vec.reset()
    raw = _policy_images_from_obs(obs)
    stacked = stacker.reset(raw)

    ep_ret = np.zeros(args_cli.num_envs, dtype=np.float64)
    ep_len = np.zeros(args_cli.num_envs, dtype=np.int32)
    finished_rets: list[float] = []

    t0 = time.time()
    last_log = global_step
    ckpt_extra = {"frame_stack": args_cli.frame_stack, "task": "Isaac-AUV-Tracking-Direct-v1"}

    while global_step < args_cli.total_steps:
        actions = agent.act(stacked, step=global_step, eval_mode=False)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        next_obs, rewards, dones, infos = vec.step(actions)
        next_raw = _policy_images_from_obs(next_obs)
        rewards_np = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones_np = np.asarray(dones, dtype=bool).reshape(-1)

        # Advance stack with post-step frame; auto-reset envs get reset obs in next_raw.
        next_stacked = stacker.update(next_raw)
        if dones_np.any():
            # Re-seed stack for finished envs with their reset observation.
            stacker.on_dones(next_raw, dones_np)
            next_stacked = stacker.get()

        replay.add_batch(stacked, actions, rewards_np, next_stacked, dones_np)

        ep_ret += rewards_np
        ep_len += 1
        for i in np.where(dones_np)[0]:
            finished_rets.append(float(ep_ret[i]))
            writer.add_scalar("rollout/ep_rew", float(ep_ret[i]), global_step)
            writer.add_scalar("rollout/ep_len", int(ep_len[i]), global_step)
            ep_ret[i] = 0.0
            ep_len[i] = 0

        stacked = next_stacked
        global_step += args_cli.num_envs

        metrics = {}
        if global_step >= args_cli.learning_starts and len(replay) >= args_cli.batch_size:
            batch = replay.sample()
            metrics = agent.update(batch, step=global_step)

        if global_step - last_log >= args_cli.log_every or global_step >= args_cli.total_steps:
            last_log = global_step
            elapsed = time.time() - t0
            fps = global_step / max(elapsed, 1e-6)
            mean_ep = float(np.mean(finished_rets[-20:])) if finished_rets else float("nan")
            writer.add_scalar("rollout/ep_rew_mean", mean_ep, global_step)
            writer.add_scalar("time/fps", fps, global_step)
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, global_step)
            print(
                f"[drqv2] step={global_step}/{args_cli.total_steps} "
                f"buffer={len(replay)} fps={fps:.1f} ep_rew_mean={mean_ep:.1f} "
                f"critic_loss={metrics.get('critic_loss', float('nan'))}",
                flush=True,
            )

        if global_step > 0 and global_step % args_cli.ckpt_every < args_cli.num_envs:
            ckpt_path = log_dir / f"checkpoint_{global_step}.pt"
            agent.save(str(ckpt_path), extra={**ckpt_extra, "global_step": global_step})
            print(f"[drqv2] saved {ckpt_path}", flush=True)

    final = log_dir / "checkpoint_final.pt"
    agent.save(str(final), extra={**ckpt_extra, "global_step": global_step})
    print(f"[drqv2] done → {final}", flush=True)
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
