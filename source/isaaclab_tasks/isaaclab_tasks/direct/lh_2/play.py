# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play script for underwater target tracking environment."""

import argparse
import os

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play underwater target tracking")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Lh-2-Direct-v0", help="Task name")
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint"
)
parser.add_argument("--save_video", action="store_true", help="Save video of the episode")
parser.add_argument(
    "--debug_print_y",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Print legacy y-axis debug during play (default: off; angle-diff uses env cfg print_ball_world_angle).",
)
parser.add_argument("--debug_print_interval", type=int, default=5, help="Y-print interval (steps) during play.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.run_mode = "play"
    env_cfg.debug_print_y = args_cli.debug_print_y
    env_cfg.debug_print_interval = max(1, args_cli.debug_print_interval)
    agent_cfg.device = env_cfg.sim.device

    env = DirectRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    resume_path = None
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        try:
            resume_path = get_checkpoint_path(log_root_path)
        except ValueError as exc:
            print(f"[INFO] No checkpoint under {log_root_path}: {exc}")

    policy = None
    if resume_path is not None:
        print(f"[INFO] Loading checkpoint from: {resume_path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        print("[INFO] No checkpoint; stepping with zero actions.")

    obs, _ = env.get_observations()
    dev = env.unwrapped.device

    print("[INFO] Running (Ctrl+C to stop).")
    try:
        while simulation_app.is_running():
            if policy is not None:
                with torch.no_grad():
                    actions = policy(obs)
            else:
                actions = torch.zeros(env.num_envs, env.num_actions, device=dev)

            obs, _, _, _ = env.step(actions)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
