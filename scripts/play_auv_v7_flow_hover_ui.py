#!/usr/bin/env python3
"""GUI playback of a Direct-v7-flow-hover SAC checkpoint (deterministic).

  ./isaaclab.sh -p scripts/play_auv_v7_flow_hover_ui.py --checkpoint PATH
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play v7 flow-hover checkpoint in the GUI.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=450)
parser.add_argument("--pace_hz", type=float, default=15.0)
parser.add_argument(
    "--current",
    type=str,
    default="random",
    help="Ocean-current field type. Training used random.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import numpy as np
import torch
from stable_baselines3 import SAC

import isaaclab_tasks  # noqa: F401
import gymnasium as gym

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.direct.isaac_auv_tracking.agents.cam_encoder import MidCNNExtractor  # noqa: F401
from isaaclab_tasks.direct.isaac_auv_tracking.agents.custom_sac_policy import CustomSACPolicy  # noqa: F401
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)

cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.use_real_usd = False
cfg.ocean_current.type = args_cli.current
env = gym.make("Isaac-AUV-Tracking-Direct-v7-flow-hover", cfg=cfg)
vec = Sb3VecEnvWrapper(env, fast_variant=True)
raw = env.unwrapped

if not args_cli.headless:
    try:
        from omni.kit.viewport.utility import get_viewport_from_window_name

        cam_path = f"{raw.scene.env_prim_paths[0]}/Robot/{raw.cfg.view_camera_name}"
        vp = get_viewport_from_window_name("Viewport")
        if vp is not None:
            vp.set_active_camera(cam_path)
            print(f"[v7-play] viewport camera -> {cam_path}", flush=True)
    except Exception as exc:
        print(f"[v7-play] viewport camera switch failed: {exc}", flush=True)

print(f"[v7-play] loading {args_cli.checkpoint}", flush=True)
agent = SAC.load(args_cli.checkpoint, vec, print_system_info=False)
obs = vec.reset()
pos0 = raw._robot.data.root_pos_w[0].detach().cpu().clone()
print("[v7-play] viewport open. deterministic policy, ocean current on.", flush=True)

t0 = time.perf_counter()
pace_dt = 1.0 / args_cli.pace_hz if args_cli.pace_hz > 0 else 0.0
last_pos = pos0.clone()
ep_rows: list[dict] = []
for step in range(args_cli.steps):
    with torch.inference_mode():
        actions, _ = agent.predict(obs, deterministic=True)
    actions = np.asarray(actions, dtype=np.float32).reshape(vec.num_envs, -1)
    with torch.inference_mode():
        obs, rewards, dones, infos = vec.step(actions)
    if pace_dt > 0.0:
        sleep_for = (t0 + (step + 1) * pace_dt) - time.perf_counter()
        if sleep_for > 0.0:
            time.sleep(sleep_for)
    done = bool(np.asarray(dones).reshape(-1)[0])
    # On the terminating step the env has already reset, so root_pos_w is the new spawn.
    # last_pos is the pose after the previous control step.
    if step % 30 == 0 and not done:
        pos = raw._robot.data.root_pos_w[0].detach().cpu()
        drift = pos - pos0
        st = obs["state"][0] if isinstance(obs, dict) else obs[0]["state"]
        st = np.asarray(st).reshape(-1)
        flow_n = float(st[8])
        print(
            f"[v7-play] step={step:4d} rew={float(np.asarray(rewards).reshape(-1)[0]):.3f} "
            f"act=({actions[0,0]:+.2f},{actions[0,1]:+.2f},{actions[0,2]:+.2f}) "
            f"flow_n={flow_n:.4f} drift=({float(drift[0]):+.3f},{float(drift[1]):+.3f},{float(drift[2]):+.3f})",
            flush=True,
        )
        last_pos = pos.clone()
    elif not done:
        last_pos = raw._robot.data.root_pos_w[0].detach().cpu().clone()
    if done:
        end = last_pos - pos0
        ep = {}
        if isinstance(infos, (list, tuple)) and infos:
            ep = infos[0].get("episode") or {}
        horiz = float(torch.linalg.norm(end[:2]))
        print(
            f"[v7-play] episode {len(ep_rows)} end step={step} "
            f"len={ep.get('l', float('nan'))} rew={ep.get('r', float('nan'))} "
            f"drift_mag={ep.get('drift_mag', float('nan'))} flow_n={ep.get('flow_rms_n', float('nan'))} "
            f"log_disp_end={ep.get('disp_end_m', float('nan'))}m "
            f"anchor_sim={ep.get('anchor_sim', float('nan'))} "
            f"anchor_disp={ep.get('anchor_disp_m', float('nan'))}m "
            f"disp=({float(end[0]):+.3f},{float(end[1]):+.3f},{float(end[2]):+.3f}) horiz={horiz:.3f}m "
            f"timeout={ep.get('term_timeout', float('nan'))} att={ep.get('term_att', float('nan'))}",
            flush=True,
        )
        ep_rows.append({
            "horiz": horiz,
            "z": float(end[2]),
            "rew": ep.get("r"),
            "drift_mag": ep.get("drift_mag"),
            "disp_end": ep.get("disp_end_m"),
            "anchor_sim": ep.get("anchor_sim"),
            "anchor_disp": ep.get("anchor_disp_m"),
        })
        pos0 = raw._robot.data.root_pos_w[0].detach().cpu().clone()
        last_pos = pos0.clone()

if ep_rows:
    horiz = np.array([r["horiz"] for r in ep_rows], dtype=np.float64)
    rews = np.array([r["rew"] if r["rew"] is not None else np.nan for r in ep_rows], dtype=np.float64)
    dm = np.array([r["drift_mag"] if r["drift_mag"] is not None else np.nan for r in ep_rows], dtype=np.float64)
    de = np.array([r["disp_end"] if r["disp_end"] is not None else np.nan for r in ep_rows], dtype=np.float64)
    asim = np.array([r["anchor_sim"] if r["anchor_sim"] is not None else np.nan for r in ep_rows], dtype=np.float64)
    adisp = np.array([r["anchor_disp"] if r["anchor_disp"] is not None else np.nan for r in ep_rows], dtype=np.float64)
    print(
        f"[v7-play] SUMMARY n={len(ep_rows)} horiz={horiz.mean():.3f}±{horiz.std():.3f} m "
        f"rew={np.nanmean(rews):.1f} drift_mag={np.nanmean(dm):.4f} log_disp_end={np.nanmean(de):.3f} m "
        f"anchor_sim={np.nanmean(asim):.3f} anchor_disp={np.nanmean(adisp):.3f} m",
        flush=True,
    )

env.close()
print("[v7-play] done", flush=True)
