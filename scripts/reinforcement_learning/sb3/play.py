# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "SAC"],
    help="RL algorithm used to train the checkpoint (PPO or SAC).",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--kf_measurement",
    type=str,
    default=None,
    choices=["gt", "sonar"],
    help="AUV-Tracking Direct-v0 KF measurement: geometric GT (fast) or OceanSim sonar.",
)
parser.add_argument("--steps", type=int, default=None, help="Number of env steps to play (default: until app closes).")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# cameras required for video, Direct-v1 vision, or OceanSim sonar KF
_needs_cameras = bool(args_cli.video)
if args_cli.task is not None and "AUV-Tracking" in args_cli.task:
    if "Direct-v1" in args_cli.task:
        _needs_cameras = True
    elif args_cli.kf_measurement != "gt":
        _needs_cameras = True
if _needs_cameras:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with stable-baselines agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.kf_measurement is not None and hasattr(env_cfg, "kf_measurement"):
        env_cfg.kf_measurement = args_cli.kf_measurement

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", train_task_name)
    log_root_path = os.path.abspath(log_root_path)
    # checkpoint and log_dir stuff
    if args_cli.use_pretrained_checkpoint:
        checkpoint_path = get_published_pretrained_checkpoint("sb3", train_task_name)
        if not checkpoint_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # FIXME: last checkpoint doesn't seem to really use the last one'
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint, sort_alpha=False)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    vec_norm_path = checkpoint_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl")
    vec_norm_path = Path(vec_norm_path)

    # normalize environment (if needed)
    if vec_norm_path.exists():
        print(f"Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False
    elif bool(agent_cfg.get("normalize_input", False)):
        print(f"Normalizing input, clip_obs={agent_cfg.get('clip_obs', 100.0)}")
        env = VecNormalize(
            env,
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=float(agent_cfg.pop("clip_obs", 100.0)),
        )
    # Drop any leftover norm keys so they don't leak into the agent constructor.
    for _k in ("normalize_input", "normalize_value", "clip_obs"):
        agent_cfg.pop(_k, None)

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    algo_cls = {"PPO": PPO, "SAC": SAC}[args_cli.algorithm.upper()]
    agent = algo_cls.load(checkpoint_path, env, print_system_info=True)

    dt = env.unwrapped.step_dt
    core = env.unwrapped
    if not hasattr(core, "_robot"):
        core = getattr(core, "unwrapped", core)

    # reset environment
    obs = env.reset()
    timestep = 0
    rew_sum = 0.0
    done_count = 0
    auv_spd, tgt_spd, cmd_vs = [], [], []
    max_steps = args_cli.steps if args_cli.steps is not None else (args_cli.video_length if args_cli.video else None)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
        timestep += 1
        r_mean = float(rewards.mean()) if hasattr(rewards, "mean") else float(rewards)
        rew_sum += r_mean
        n_done = int(dones.sum()) if hasattr(dones, "sum") else int(bool(dones))
        done_count += n_done
        v_auv = v_tgt = cmd_v = cmd_w = float("nan")
        if hasattr(core, "_robot") and hasattr(core, "_target_state"):
            lin_w = core._robot.data.root_lin_vel_w
            v_auv = float(torch.linalg.norm(lin_w[:, :2], dim=-1).mean())
            v_tgt = float(torch.linalg.norm(core._target_state[:, 2:4], dim=-1).mean())
            cmd = core._cmd_vel
            cmd_v = float(cmd[:, 0].mean())
            cmd_w = float(cmd[:, 1].mean())
            auv_spd.append(v_auv)
            tgt_spd.append(v_tgt)
            cmd_vs.append(cmd_v)
        if timestep % 50 == 0 or n_done:
            print(
                f"[play] step={timestep:4d}  mean_rew={r_mean:7.3f}  "
                f"avg_rew={rew_sum / timestep:7.3f}  dones={done_count}  "
                f"auv_xy={v_auv:.3f} m/s  tgt_xy={v_tgt:.3f} m/s  cmd_v={cmd_v:.3f} cmd_w={cmd_w:.3f}",
                flush=True,
            )
        if args_cli.video and timestep >= args_cli.video_length:
            break
        if max_steps is not None and timestep >= max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    avg = rew_sum / max(timestep, 1)
    print(f"[play] done steps={timestep}  mean_step_rew={avg:.3f}  approx_ep_rew={avg * 200:.1f}  env_dones={done_count}")
    if auv_spd:
        import numpy as np
        print(
            f"[play] AUV |v_xy| mean={float(np.mean(auv_spd)):.3f} max={float(np.max(auv_spd)):.3f} "
            f"cmd_v mean={float(np.mean(cmd_vs)):.3f} max={float(np.max(np.abs(cmd_vs))):.3f}",
            flush=True,
        )
        print(
            f"[play] TGT |v_xy| mean={float(np.mean(tgt_spd)):.3f} max={float(np.max(tgt_spd)):.3f}",
            flush=True,
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
