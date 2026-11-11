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
    "--render_interval",
    type=int,
    default=None,
    help="Physics steps between viewport/camera renders. Training uses 24 (~5 viewport FPS). "
    "GUI defaults to 6 (~20 fps, smoother). Headless leaves the task default alone.",
)
parser.add_argument(
    "--viewport_cam",
    choices=["robot", "persp", "none"],
    default=None,
    help="GUI viewport camera. Default for *-cam tasks: robot onboard LookFollowCamera0.",
)
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

# cameras required for video, Direct-v1 vision, camera tasks, or OceanSim sonar KF
_needs_cameras = bool(args_cli.video)
if args_cli.task is not None and "AUV-Tracking" in args_cli.task:
    if "Direct-v1" in args_cli.task:
        _needs_cameras = True
    elif "-cam" in args_cli.task:
        # Direct-v3-cam / Direct-v4-cam: the RGB camera is the policy observation.
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
import math
import os
import random
import time
import torch

import numpy as np

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

    # GUI viewport stutter: training uses render_interval == decimation == 24, so the image
    # advances 0.2 s of sim time per frame (~5 fps). Rendering is cheap relative to the PID /
    # hydrodynamics substep chain; drop the interval so the viewport updates more often.
    # Headless play / eval keeps the task default so metrics stay comparable to training.
    # *-cam tasks pay more per render (tiled camera); still prefer frequent presents — probe
    # measured ~+8 ms/extra frame vs a much smoother view (interval 6→2 ≈ 4→12 presents/step).
    ui_mode = not bool(getattr(args_cli, "headless", False))
    render_interval = args_cli.render_interval
    if render_interval is None and ui_mode:
        cam_task = args_cli.task is not None and "-cam" in args_cli.task
        render_interval = 2 if cam_task else 4
    if render_interval is not None:
        prev = int(env_cfg.sim.render_interval)
        env_cfg.sim.render_interval = int(render_interval)
        print(
            f"[play] render_interval: {prev} -> {env_cfg.sim.render_interval} "
            f"({'GUI smooth' if ui_mode else 'override'})",
            flush=True,
        )

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

    # Point the GUI viewport at the onboard camera for *-cam tasks (same call as the v4 probe).
    if ui_mode and args_cli.task is not None and "-cam" in args_cli.task:
        _vp = args_cli.viewport_cam if args_cli.viewport_cam is not None else "robot"
        if _vp != "none":
            try:
                from omni.kit.viewport.utility import get_viewport_from_window_name

                vp = get_viewport_from_window_name("Viewport")
                if vp is None:
                    print("[play] WARN: no Viewport window; leaving camera untouched", flush=True)
                else:
                    core0 = env.unwrapped
                    if hasattr(core0, "unwrapped"):
                        core0 = core0.unwrapped
                    if _vp == "persp":
                        target = "/OmniverseKit_Persp"
                    elif hasattr(core0, "_robot_camera_prim_paths"):
                        target = core0._robot_camera_prim_paths()[0]
                    else:
                        target = "/World/envs/env_0/Robot/warpauv_base_link/LookFollowCamera0"
                    vp.set_active_camera(target)
                    print(f"[play] GUI viewport camera -> {target}", flush=True)
            except Exception as exc:  # noqa: BLE001 - viewport nicety must not abort play
                print(f"[play] WARN: could not set viewport camera: {exc}", flush=True)

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
    auv_spd, tgt_spd, cmd_vs, ep_ages = [], [], [], []
    # Extra diagnostics for reward redesign (range / FOV / col / actions / belief).
    diag_gt_r, diag_gt_a, diag_tot, diag_obs = [], [], [], []
    diag_logdet, diag_is_col, diag_av, diag_aw, diag_rew = [], [], [], [], []
    diag_r_belief, diag_a_belief = [], []
    # World-space trajectories for offline motion plots (per env, per step).
    diag_agent_pos, diag_tgt_pos, diag_agent_yaw = [], [], []
    # v5-flow reward decomposition (only filled when the env exposes the flow attrs).
    diag_r_bear, diag_r_size, diag_delta_pen, diag_flow_a, diag_size_ratio = [], [], [], [], []
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
            # Per-env speeds (not just the mean) so cold-start vs free-RRT phases stay separable.
            auv_xy = torch.linalg.norm(lin_w[:, :2], dim=-1)
            tgt_xy = torch.linalg.norm(core._target_state[:, 2:4], dim=-1)
            cmd = core._cmd_vel
            v_auv = float(auv_xy.mean())
            v_tgt = float(tgt_xy.mean())
            cmd_v = float(cmd[:, 0].mean())
            cmd_w = float(cmd[:, 1].mean())
            auv_spd.extend(auv_xy.detach().cpu().tolist())
            tgt_spd.extend(tgt_xy.detach().cpu().tolist())
            cmd_vs.extend(cmd[:, 0].detach().cpu().tolist())
            if hasattr(core, "episode_length_buf"):
                # episode_length_buf counts control steps; soft_start_s=3 / easy_s=5 at dt=0.2 s
                # ⇒ ages <15 / <25. Ages come from episode_length_buf.
                ep_ages.extend(core.episode_length_buf.detach().cpu().tolist())
            else:
                ep_ages.extend([0] * int(auv_xy.numel()))

            # Geometry vs camera FOV / KF belief (what the reward actually cares about).
            dxy = core._target_pos[:, :2] - core._robot.data.root_pos_w[:, :2]
            gt_r = torch.linalg.norm(dxy, dim=-1)
            agent_pos, agent_yaw = core._agent_pose_2d()
            _, _, gt_a = core.kf.get_observation(core._target_pos, agent_pos, agent_yaw)
            fov_half = float(core._fov_half)
            sensor_r = float(core.cfg.sensor_range)
            tot = (gt_r <= sensor_r) & (gt_a.abs() <= fov_half)
            act = torch.as_tensor(actions, device=core.device, dtype=torch.float32).reshape(core.num_envs, -1)
            diag_gt_r.extend(gt_r.detach().cpu().tolist())
            diag_gt_a.extend(gt_a.abs().detach().cpu().tolist())
            diag_tot.extend(tot.float().detach().cpu().tolist())
            diag_obs.extend(core._observed.float().detach().cpu().tolist())
            diag_logdet.extend(core._log_det_cov().detach().cpu().tolist())
            diag_is_col.extend(core._is_col.float().detach().cpu().tolist())
            diag_av.extend(act[:, 0].detach().cpu().tolist())
            diag_aw.extend(act[:, 1].detach().cpu().tolist())
            rew_t = rewards if torch.is_tensor(rewards) else torch.as_tensor(rewards)
            diag_rew.extend(rew_t.detach().float().cpu().reshape(-1).tolist())
            # Belief polar from policy state when available (asymmetric critic input).
            if isinstance(obs, dict) and "state" in obs:
                st = obs["state"]
                st_t = st if torch.is_tensor(st) else torch.as_tensor(st)
                diag_r_belief.extend(st_t[..., 0].detach().float().cpu().reshape(-1).tolist())
                diag_a_belief.extend(st_t[..., 1].abs().detach().float().cpu().reshape(-1).tolist())
            elif hasattr(core, "_build_policy_obs"):
                # Fallback: rebuild belief obs (r, alpha, logdet, observed).
                bel = core._build_policy_obs()
                diag_r_belief.extend(bel[:, 0].detach().cpu().tolist())
                diag_a_belief.extend(bel[:, 1].abs().detach().cpu().tolist())
            # Trajectories: agent world pos + yaw, target world pos (2D RRT plane).
            agent_pos_w = core._robot.data.root_pos_w.detach().cpu()
            diag_agent_pos.extend(agent_pos_w.reshape(-1).tolist())
            tgt_p = core._target_pos.detach().cpu()
            if tgt_p.shape[-1] < 3:
                tgt_p = torch.cat([tgt_p, torch.zeros(*tgt_p.shape[:-1], 3 - tgt_p.shape[-1])], dim=-1)
            diag_tgt_pos.extend(tgt_p.reshape(-1).tolist())
            _, agent_yaw_arr = core._agent_pose_2d()
            diag_agent_yaw.extend(torch.as_tensor(agent_yaw_arr).detach().cpu().reshape(-1).tolist())
            # v5-flow reward decomposition (R8 first-principles formula), only when the
            # attributes exist so v0/v4-cam plays stay unaffected.
            if hasattr(core, "_flow_est_a") and hasattr(core, "_size_ratio"):
                sig_b = max(float(getattr(core.cfg, "rew_bearing_sigma_deg", 15.0)) / 57.29578, 1.0e-6)
                flow_a = core._flow_est_a.detach().float()
                r_bear = torch.exp(-((flow_a / sig_b) ** 2))
                sig_s = max(float(getattr(core.cfg, "rew_size_sigma", 0.5)), 1.0e-6)
                r_size = torch.exp(-((torch.log(core._size_ratio.detach().float().clamp(min=0.05)) / sig_s) ** 2))
                act = torch.as_tensor(actions, device=core.device, dtype=torch.float32).reshape(core.num_envs, -1)
                smooth = torch.sum((act - core._prev_actions.detach().float()) ** 2, dim=-1)
                diag_r_bear.extend(r_bear.cpu().tolist())
                diag_r_size.extend(r_size.cpu().tolist())
                diag_delta_pen.extend(
                    (float(getattr(core.cfg, "rew_penalty_action_delta_v5", 0.03)) * smooth).cpu().tolist()
                )
                diag_flow_a.extend(torch.rad2deg(flow_a).cpu().tolist())
                diag_size_ratio.extend(core._size_ratio.detach().float().cpu().tolist())
        if timestep % 50 == 0 or n_done:
            tot_m = float(np.mean(diag_tot[-core.num_envs :])) if diag_tot else float("nan")
            col_m = float(np.mean(diag_is_col[-core.num_envs :])) if diag_is_col else float("nan")
            r_m = float(np.mean(diag_gt_r[-core.num_envs :])) if diag_gt_r else float("nan")
            print(
                f"[play] step={timestep:4d}  mean_rew={r_mean:7.3f}  "
                f"avg_rew={rew_sum / timestep:7.3f}  dones={done_count}  "
                f"auv_xy={v_auv:.3f} m/s  tgt_xy={v_tgt:.3f} m/s  cmd_v={cmd_v:.3f} cmd_w={cmd_w:.3f}  "
                f"gt_r={r_m:.2f} tot={tot_m:.2f} col={col_m:.2f}",
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

        a = np.asarray(auv_spd, dtype=np.float64)
        t = np.asarray(tgt_spd, dtype=np.float64)
        c = np.asarray(cmd_vs, dtype=np.float64)
        age = np.asarray(ep_ages, dtype=np.float64) if ep_ages else np.zeros(len(a))
        # Control-step dt ≈ 0.2 s (decimation=24 @ 120 Hz). soft_start_s=3 → age<15;
        # easy_s=5 → age<25; free RRT after that. Ages come from episode_length_buf.
        cold = age < 15
        easy = (age >= 15) & (age < 25)
        free = age >= 25

        def _pct(x):
            return tuple(float(np.percentile(x, p)) for p in (10, 50, 90))

        def _blk(name, mask):
            if not np.any(mask):
                return
            aa, tt, cc = a[mask], t[mask], c[mask]
            gap = aa - tt
            print(
                f"[play][spd {name}] n={int(mask.sum())}  "
                f"auv p10/50/90={_pct(aa)[0]:.3f}/{_pct(aa)[1]:.3f}/{_pct(aa)[2]:.3f}  "
                f"tgt p10/50/90={_pct(tt)[0]:.3f}/{_pct(tt)[1]:.3f}/{_pct(tt)[2]:.3f}  "
                f"cmd_v p10/50/90={_pct(cc)[0]:.3f}/{_pct(cc)[1]:.3f}/{_pct(cc)[2]:.3f}  "
                f"auv-tgt mean={float(gap.mean()):+.3f}  frac|auv-tgt|>0.2={float(np.mean(np.abs(gap) > 0.2)):.2f}",
                flush=True,
            )

        print(
            f"[play] AUV |v_xy| mean={float(a.mean()):.3f} max={float(a.max()):.3f} "
            f"cmd_v mean={float(c.mean()):.3f} max={float(np.max(np.abs(c))):.3f}",
            flush=True,
        )
        print(
            f"[play] TGT |v_xy| mean={float(t.mean()):.3f} max={float(t.max()):.3f}  "
            f"(v0 cruise=0.5 ceil=0.6; action_scale_v=1.5 ⇒ cmd_v∈[-1.5,1.5]; rew_scale_speed=0)",
            flush=True,
        )
        _blk("all", np.ones(len(a), dtype=bool))
        _blk("cold0-3s", cold)
        _blk("easy3-5s", easy)
        _blk("free>5s", free)

    if diag_gt_r:

        gr = np.asarray(diag_gt_r, dtype=np.float64)
        ga = np.asarray(diag_gt_a, dtype=np.float64)
        tot = np.asarray(diag_tot, dtype=np.float64)
        obs_f = np.asarray(diag_obs, dtype=np.float64)
        ld = np.asarray(diag_logdet, dtype=np.float64)
        col = np.asarray(diag_is_col, dtype=np.float64)
        av = np.asarray(diag_av, dtype=np.float64)
        aw = np.asarray(diag_aw, dtype=np.float64)
        rw = np.asarray(diag_rew, dtype=np.float64)
        age = np.asarray(ep_ages, dtype=np.float64) if ep_ages else np.zeros(len(gr))
        free = age >= 25

        def _pct(x):
            return tuple(float(np.percentile(x, p)) for p in (10, 50, 90))

        print("[play][diag] === tracking / FOV / collision (reward-relevant) ===", flush=True)
        print(
            f"[play][diag] gt_r p10/50/90={_pct(gr)[0]:.2f}/{_pct(gr)[1]:.2f}/{_pct(gr)[2]:.2f}  "
            f"|gt_a|deg p10/50/90={_pct(np.degrees(ga))[0]:.1f}/{_pct(np.degrees(ga))[1]:.1f}/{_pct(np.degrees(ga))[2]:.1f}  "
            f"fov_half={math.degrees(float(core._fov_half)):.1f}deg  sensor_r={float(core.cfg.sensor_range):.1f}m",
            flush=True,
        )
        print(
            f"[play][diag] tot(in FOV+range)={float(tot.mean()):.3f}  observed(KF)={float(obs_f.mean()):.3f}  "
            f"is_col={float(col.mean()):.3f}  frac_r<1.0={float((gr < 1.0).mean()):.3f}  "
            f"frac_r>3.5={float((gr > 3.5).mean()):.3f}  frac_|a|>15deg={float((ga > math.radians(15)).mean()):.3f}",
            flush=True,
        )
        print(
            f"[play][diag] logdet p10/50/90={_pct(ld)[0]:.2f}/{_pct(ld)[1]:.2f}/{_pct(ld)[2]:.2f}  "
            f"rew p10/50/90={_pct(rw)[0]:.2f}/{_pct(rw)[1]:.2f}/{_pct(rw)[2]:.2f}  "
            f"frac_rew<0={float((rw < 0).mean()):.3f}",
            flush=True,
        )
        print(
            f"[play][diag] a_v p10/50/90={_pct(av)[0]:+.2f}/{_pct(av)[1]:+.2f}/{_pct(av)[2]:+.2f}  "
            f"a_w p10/50/90={_pct(aw)[0]:+.2f}/{_pct(aw)[1]:+.2f}/{_pct(aw)[2]:+.2f}  "
            f"frac|a_v|>0.9={float((np.abs(av) > 0.9).mean()):.2f}  frac|a_w|>0.9={float((np.abs(aw) > 0.9).mean()):.2f}  "
            f"frac a_v<0={float((av < 0).mean()):.2f}",
            flush=True,
        )
        # Preferred standoff ~2 m (init band mid); report free-phase only so cold-start leash doesn't dominate.
        if free.any():
            print(
                f"[play][diag free>5s] gt_r p50={float(np.median(gr[free])):.2f}  "
                f"|gt_a|deg p50={float(np.degrees(np.median(ga[free]))):.1f}  "
                f"tot={float(tot[free].mean()):.3f}  col={float(col[free].mean()):.3f}  "
                f"|r-2| mean={float(np.abs(gr[free] - 2.0).mean()):.2f}",
                flush=True,
            )
        if diag_r_belief:
            rb = np.asarray(diag_r_belief, dtype=np.float64)
            ab = np.asarray(diag_a_belief, dtype=np.float64)
            print(
                f"[play][diag] belief_r p50={float(np.median(rb)):.2f}  "
                f"|belief_a|deg p50={float(np.degrees(np.median(ab))):.1f}  "
                f"|belief_r - gt_r| mean={float(np.abs(rb - gr[: len(rb)]).mean()):.2f}",
                flush=True,
            )
        # Dump for offline analysis / canvas.
        dump_path = "/tmp/auv_v4_cam_play_diag.npz"
        np.savez(
            dump_path,
            gt_r=gr,
            gt_a=ga,
            tot=tot,
            observed=obs_f,
            logdet=ld,
            is_col=col,
            a_v=av,
            a_w=aw,
            rew=rw,
            auv_spd=np.asarray(auv_spd, dtype=np.float64),
            tgt_spd=np.asarray(tgt_spd, dtype=np.float64),
            cmd_v=np.asarray(cmd_vs, dtype=np.float64),
            ep_age=age,
            agent_pos=np.asarray(diag_agent_pos, dtype=np.float64).reshape(len(gr), -1),
            tgt_pos=np.asarray(diag_tgt_pos, dtype=np.float64).reshape(len(gr), -1),
            agent_yaw=np.asarray(diag_agent_yaw, dtype=np.float64),
            r_bear=np.asarray(diag_r_bear, dtype=np.float64),
            r_size=np.asarray(diag_r_size, dtype=np.float64),
            delta_pen=np.asarray(diag_delta_pen, dtype=np.float64),
            flow_a_deg=np.asarray(diag_flow_a, dtype=np.float64),
            size_ratio=np.asarray(diag_size_ratio, dtype=np.float64),
        )
        print(f"[play][diag] saved {dump_path}", flush=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
