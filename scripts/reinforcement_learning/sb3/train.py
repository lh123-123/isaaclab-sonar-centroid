# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
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
    help="RL algorithm (PPO or SAC).",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--kf_measurement",
    type=str,
    default=None,
    choices=["gt", "sonar"],
    help="AUV-Tracking Direct-v0 KF measurement: geometric GT (fast) or OceanSim sonar.",
)
parser.add_argument("--rew_speed", type=float, default=None, help="Override rew_scale_speed (v2 speed match).")
parser.add_argument("--rew_ref", type=float, default=None, help="Override rew_scale_ref (v2 reference distance).")
parser.add_argument("--rew_ref_sigma", type=float, default=None, help="Override rew_ref_sigma (v2 ref gaussian width).")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
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
# cameras required for video and for OceanSim sonar / Direct-v1 vision
_needs_cameras = bool(args_cli.video)
if args_cli.task is not None and "AUV-Tracking" in args_cli.task:
    if "Direct-v1" in args_cli.task or "Direct-v2" in args_cli.task:
        _needs_cameras = True
    elif "-cam" in args_cli.task:
        # A camera task (Direct-v3-cam / Direct-v4-cam ...): the RGB camera *is* the policy
        # observation. These use geometric-GT KF measurements, so the sonar branch below must
        # not be what decides this.
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


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import importlib
import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime
from typing import Any

import omni
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps

from isaaclab_tasks.direct.isaac_auv_tracking.agents.gate_log_callback import CriticGateLogCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize


def _resolve_sb3_class(spec: Any) -> Any:
    """Resolve a class from an import path string (Holo SB3_trainer style)."""
    if not isinstance(spec, str):
        return spec
    if "." not in spec:
        return spec
    module_path, class_name = spec.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _prepare_sb3_policy(agent_cfg: dict) -> tuple[Any, dict]:
    """Pop policy + resolve features_extractor_class import paths in policy_kwargs."""
    policy_arch = _resolve_sb3_class(agent_cfg.pop("policy"))
    policy_kwargs = agent_cfg.get("policy_kwargs")
    if isinstance(policy_kwargs, dict):
        fe = policy_kwargs.get("features_extractor_class")
        if isinstance(fe, str):
            policy_kwargs["features_extractor_class"] = _resolve_sb3_class(fe)
    return policy_arch, agent_cfg

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # max iterations for training (PPO-style: iterations * n_steps * num_envs)
    if args_cli.max_iterations is not None:
        if "n_steps" in agent_cfg:
            agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs
        else:
            # SAC: treat max_iterations as total timesteps override
            agent_cfg["n_timesteps"] = args_cli.max_iterations

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.kf_measurement is not None and hasattr(env_cfg, "kf_measurement"):
        env_cfg.kf_measurement = args_cli.kf_measurement
    if args_cli.rew_speed is not None and hasattr(env_cfg, "rew_scale_speed"):
        env_cfg.rew_scale_speed = args_cli.rew_speed
    if args_cli.rew_ref is not None and hasattr(env_cfg, "rew_scale_ref"):
        env_cfg.rew_scale_ref = args_cli.rew_ref
    if args_cli.rew_ref_sigma is not None and hasattr(env_cfg, "rew_ref_sigma"):
        env_cfg.rew_ref_sigma = args_cli.rew_ref_sigma

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    # read configurations about the agent-training
    policy_arch, agent_cfg = _prepare_sb3_policy(agent_cfg)
    n_timesteps = agent_cfg.pop("n_timesteps")

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    if norm_args and norm_args.get("normalize_input"):
        print(f"Normalizing input, {norm_args=}")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=norm_args["normalize_input"],
            norm_reward=norm_args.get("normalize_value", False),
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # create agent from stable baselines
    algo = args_cli.algorithm.upper()
    # Optional Holo-style action noise for SAC (σ=0.05 on 2D PID actions).
    # Use per-action shape (act_dim,) so it broadcasts to (num_envs, act_dim) — never bake in num_envs.
    use_action_noise = bool(agent_cfg.pop("use_action_noise", True))
    noise_std = float(agent_cfg.pop("action_noise_std", 0.05))
    act_shape = tuple(int(x) for x in env.action_space.shape)
    act_dim = int(act_shape[-1]) if act_shape else 1

    def _make_action_noise() -> NormalActionNoise:
        return NormalActionNoise(mean=np.zeros(act_dim), sigma=np.ones(act_dim) * noise_std)

    action_noise = None
    if algo == "SAC" and use_action_noise:
        action_noise = _make_action_noise()
        agent_cfg["action_noise"] = action_noise

    algo_cls = {"PPO": PPO, "SAC": SAC}[algo]
    resume_training = args_cli.checkpoint is not None
    if resume_training:
        print(f"[INFO] Resuming weights from: {args_cli.checkpoint}")
        agent = algo_cls.load(args_cli.checkpoint, env=env, print_system_info=True)
        # Keep TB / ckpts in this run's log_dir (not the old run folder).
        agent.tensorboard_log = log_dir
        # Old runs may have saved action_noise as (old_num_envs, act_dim), e.g. (32,2) vs (16,2).
        if algo == "SAC" and use_action_noise:
            agent.action_noise = _make_action_noise()
            print(f"[INFO] Rebuilt action_noise for num_envs={env.num_envs}, act_dim={act_dim}")
        print(f"[INFO] Loaded num_timesteps={getattr(agent, 'num_timesteps', '?')} → continue to {n_timesteps}")
    else:
        agent = algo_cls(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)

    # callbacks for agent
    # CheckpointCallback.save_freq counts VecEnv steps (1 call ≈ num_envs transitions).
    # Without dividing by num_envs, e.g. 3e6//20=1.5e5 with 32 envs ⇒ first ckpt at 4.8e6
    # (never reached) — so a crash leaves no model.zip.
    num_envs = int(env.num_envs)
    ckpt_transitions = max(50_000, int(n_timesteps // 20))  # ~every 5% of the run
    ckpt_every = max(1, ckpt_transitions // num_envs)
    print(
        f"[INFO] Checkpoint every {ckpt_every} VecEnv steps "
        f"(≈{ckpt_every * num_envs} transitions, {num_envs} envs)"
    )
    checkpoint_callback = CheckpointCallback(save_freq=ckpt_every, save_path=log_dir, name_prefix="model", verbose=2)
    callbacks = [checkpoint_callback, LogEveryNTimesteps(n_steps=args_cli.log_interval), CriticGateLogCallback()]

    # train the agent
    # reset_num_timesteps=False: keep counter so learn() stops at total n_timesteps (true resume).
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=None,
            reset_num_timesteps=not resume_training,
        )
    # save the final model
    agent.save(os.path.join(log_dir, "model"))
    print("Saving to:")
    print(os.path.join(log_dir, "model.zip"))

    if isinstance(env, VecNormalize):
        print("Saving normalization")
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
