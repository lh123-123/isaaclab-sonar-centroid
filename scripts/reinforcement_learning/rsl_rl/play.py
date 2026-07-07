# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug_print_y",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Print robot/target y during play (default: enabled, use --no-debug_print_y to disable).",
)
parser.add_argument(
    "--debug_print_interval", type=int, default=5, help="Y debug print interval in environment steps."
)
parser.add_argument(
    "--dy_sign_guard",
    action="store_true",
    default=False,
    help="Enable optional y-tracking action sign guard in env (LH-2 debug).",
)
parser.add_argument(
    "--dy_sign_guard_deadband",
    type=float,
    default=0.05,
    help="Deadband (meters) for dy_sign_guard near alignment.",
)
parser.add_argument(
    "--sonar_ui",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable OceanSim sonar viewport UI when supported (default: enabled, use --no-sonar_ui to disable).",
)
parser.add_argument(
    "--stochastic_policy",
    action="store_true",
    default=False,
    help=(
        "Sample actions from the Gaussian policy (same as training rollouts) instead of using the mean. "
        "Use when deterministic play looks idle but training metrics look fine (mean collapse)."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Optional LH-2 debug hooks (applied only when the task cfg defines these fields).
    if hasattr(env_cfg, "debug_print_y"):
        env_cfg.debug_print_y = args_cli.debug_print_y
    if hasattr(env_cfg, "run_mode"):
        env_cfg.run_mode = "play"
    if hasattr(env_cfg, "debug_print_interval"):
        env_cfg.debug_print_interval = max(1, int(args_cli.debug_print_interval))
    if hasattr(env_cfg, "dy_sign_guard"):
        env_cfg.dy_sign_guard = bool(args_cli.dy_sign_guard)
    if hasattr(env_cfg, "dy_sign_guard_deadband"):
        env_cfg.dy_sign_guard_deadband = float(args_cli.dy_sign_guard_deadband)
    if args_cli.sonar_ui and hasattr(env_cfg, "oceansim_sonar_cfg"):
        sonar_cfg = getattr(env_cfg, "oceansim_sonar_cfg")
        if hasattr(sonar_cfg, "sonar_initialize_viewport"):
            sonar_cfg.sonar_initialize_viewport = True

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference (mean action) or stochastic rollout (sample)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    if args_cli.stochastic_policy:
        pol_nn = ppo_runner.alg.policy
        dev = env.unwrapped.device
        if agent_cfg.empirical_normalization:
            obs_norm = ppo_runner.obs_normalizer

            def policy(obs_tensor):  # noqa: E306
                return pol_nn.act(obs_norm(obs_tensor.to(dev)))

        else:

            def policy(obs_tensor):  # noqa: E306
                return pol_nn.act(obs_tensor.to(dev))

        print("[INFO]: Using stochastic policy.act() (sampled actions), not act_inference mean.")

    _pol = ppo_runner.alg.policy
    if hasattr(_pol, "std"):
        print(f"[INFO] Policy exploration std (checkpoint): {_pol.std.detach().cpu().flatten().tolist()}")
    elif hasattr(_pol, "log_std"):
        print(
            "[INFO] Policy exploration std (checkpoint): "
            f"{torch.exp(_pol.log_std).detach().cpu().flatten().tolist()}"
        )

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit (exporter assumes a flat obs -> MLP actor; skip if incompatible e.g. CNN + MLP fuse)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )
    except Exception as exc:
        print(f"[WARN] Skipped JIT/ONNX export (custom vision / fused policy often unsupported by default exporter): {exc}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # Stochastic act() uses RNG: avoid wrapping it in inference_mode (can make samples look "stuck" on some setups).
        if args_cli.stochastic_policy:
            with torch.no_grad():
                actions = policy(obs)
        else:
            with torch.inference_mode():
                actions = policy(obs)
        with torch.inference_mode():
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
