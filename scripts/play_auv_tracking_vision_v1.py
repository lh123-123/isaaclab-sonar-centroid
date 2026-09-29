# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Finite-step headless play for Isaac-AUV-Tracking-Direct-v1 (CustomSAC)."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument(
    "--rew_scale_smooth",
    type=float,
    default=0.0,
    help="Reward weight on action smoothness. 0 = off (08-10 training).",
)
parser.add_argument("--dump", type=str, default="/tmp/v1_vision_play_actions.npz")
parser.add_argument(
    "--episodes",
    type=int,
    default=None,
    help="Stop after this many completed episodes. Step budget becomes max(steps, episodes*250).",
)
parser.add_argument(
    "--summary",
    type=str,
    default=None,
    help="Write episode mean±std JSON here. Default: dump with .json suffix.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import io
import zipfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent / "lib"))
from auv_play_metrics import format_summary, summarize_play, write_summary_json
from isaaclab_tasks.direct.isaac_auv_tracking.agents import sonar_encoder as sonar_encoder_mod
from isaaclab_tasks.direct.isaac_auv_tracking.agents.custom_sac_policy import (  # noqa: F401
    GatedResidualSACPolicy,
    HybridImageStateSACPolicy,
)
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_vision_env import AUVTrackingVisionEnvCfg


class CompactSonarEncoder(BaseFeaturesExtractor):
    """08-10 compact CNN Actor encoder (not the current ResNet50)."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 512,
        resnet_output_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=1)
        image_space = observation_space["image"]
        n_input = int(image_space.shape[0])
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_input, int(image_space.shape[1]), int(image_space.shape[2]))
            n_flat = int(self.cnn(dummy).shape[1])
        self.trunk = nn.Sequential(
            nn.Linear(n_flat, int(resnet_output_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(resnet_output_dim), int(features_dim)),
            nn.LayerNorm(int(features_dim)),
            nn.Tanh(),
        )
        self._features_dim = int(features_dim)

    def forward(self, observations: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        images = observations["image"] if isinstance(observations, dict) else observations
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()
            if float(images.max()) > 1.5:
                images = images / 255.0
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        if images.shape[1] == 3:
            images = (images - mean) / std
        return self.trunk(self.cnn(images))


def _checkpoint_is_compact_cnn(path: str) -> bool:
    with zipfile.ZipFile(path) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=False)
    return any(str(k).startswith("actor.features_extractor.cnn.") for k in sd)


def main():
    cfg = AUVTrackingVisionEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.ocean_current.enabled = False
    cfg.rew_scale_smooth = float(args_cli.rew_scale_smooth)
    if not args_cli.headless:
        cfg.sim.render_interval = 4
    env = gym.make("Isaac-AUV-Tracking-Direct-v1", cfg=cfg)
    vec = Sb3VecEnvWrapper(env, fast_variant=True)
    core = env.unwrapped

    if _checkpoint_is_compact_cnn(args_cli.checkpoint):
        sonar_encoder_mod.SonarEncoder = CompactSonarEncoder
        print("[play] compact-CNN checkpoint: patched SonarEncoder for load", flush=True)

    print(
        f"[play] loading {args_cli.checkpoint} ocean_current=False "
        f"rew_scale_smooth={cfg.rew_scale_smooth}",
        flush=True,
    )
    agent = SAC.load(args_cli.checkpoint, env=vec, print_system_info=False)

    def _obs_state_tensor(obs):
        if not isinstance(obs, dict) or "state" not in obs:
            return None
        state = obs["state"]
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, device=agent.device)
        else:
            state = state.to(agent.device)
        if state.dim() > 2:
            state = state.reshape(state.shape[0], -1)
        return state

    def _gate_mean(obs):
        critic = getattr(agent.policy, "critic", None)
        if critic is None or not hasattr(critic, "gates"):
            return None
        state = _obs_state_tensor(obs)
        if state is None:
            return None
        with torch.no_grad():
            g = torch.sigmoid(critic.gates[0](state.float()))
        return float(g.mean().cpu())

    n_episodes_target = args_cli.episodes
    step_budget = int(args_cli.steps)
    if n_episodes_target is not None:
        step_budget = max(step_budget, int(n_episodes_target) * 250)
        print(f"[play] episode target={n_episodes_target} step_budget={step_budget}", flush=True)

    def _vel_stats():
        lin_w = core._robot.data.root_lin_vel_w.detach()
        auv_xy = torch.linalg.norm(lin_w[:, :2], dim=-1)
        tgt_xy = torch.linalg.norm(core._target_state[:, 2:4], dim=-1)
        cmd = core._cmd_vel.detach()
        dxy = core._target_pos[:, :2] - core._robot.data.root_pos_w[:, :2]
        return {
            "auv_xy": float(auv_xy.mean()),
            "tgt_xy": float(tgt_xy.mean()),
            "cmd_v": float(cmd[:, 0].mean()),
            "cmd_w": float(cmd[:, 1].mean()),
            "gt_r": float(torch.linalg.norm(dxy, dim=-1).mean()),
        }

    obs = vec.reset()
    rews, auv_spd, tgt_spd, raw_av, raw_aw, raw_da, gt_range, dones_hist, gates = [], [], [], [], [], [], [], [], []
    last_act = None
    ep_rets = []
    ep_lens = []
    cur_ret = np.zeros(vec.num_envs, dtype=np.float64)
    cur_len = np.zeros(vec.num_envs, dtype=np.int32)

    for t in range(step_budget):
        actions, _ = agent.predict(obs, deterministic=True)
        actions = np.asarray(actions, dtype=np.float32).reshape(vec.num_envs, -1)
        obs, rewards, dones, infos = vec.step(actions)
        r = np.asarray(rewards, dtype=np.float64).reshape(-1)
        rews.append(float(r.mean()))
        cur_ret += r
        cur_len += 1
        vs = _vel_stats()
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
        gmean = _gate_mean(obs)
        if gmean is not None:
            gates.append(gmean)
        done = np.asarray(dones, dtype=bool).reshape(-1)
        dones_hist.append(bool(done.any()) if vec.num_envs == 1 else bool(done[0]))
        if done.any():
            for i in np.where(done)[0]:
                ep_rets.append(float(cur_ret[i]))
                ep_lens.append(int(cur_len[i]))
                cur_ret[i] = 0.0
                cur_len[i] = 0
            if n_episodes_target is not None and len(ep_rets) >= n_episodes_target:
                print(f"[play] reached {len(ep_rets)} episodes at step {t}", flush=True)
                break
        if t % args_cli.log_every == 0 or t == step_budget - 1:
            print(
                f"[play] step={t:04d} a=[{a_v:+.3f},{a_w:+.3f}] da={da:.3f} "
                f"cmd=[{vs['cmd_v']:+.3f},{vs['cmd_w']:+.3f}] "
                f"auv_xy={vs['auv_xy']:.3f} tgt_xy={vs['tgt_xy']:.3f} "
                f"gt_r={vs['gt_r']:.2f} rew={rews[-1]:.3f}",
                flush=True,
            )

    av = np.asarray(raw_av)
    aw = np.asarray(raw_aw)
    edges = np.linspace(-1.0, 1.0, 11)

    def _hist(name, x):
        counts, _ = np.histogram(x, bins=edges)
        peak = max(int(counts.max()), 1)
        print(
            f"[a-dist] {name} n={len(x)} mean={float(x.mean()):+.3f} "
            f"|mean|={float(np.mean(np.abs(x))):.3f} std={float(x.std()):.3f} "
            f"frac|a|>0.9={float(np.mean(np.abs(x) > 0.9)):.2f}",
            flush=True,
        )
        for i, c in enumerate(counts):
            print(
                f"[a-dist] {edges[i]:+.1f}..{edges[i + 1]:+.1f} {c:4d} "
                f"{'#' * int(round(40 * c / peak))}",
                flush=True,
            )

    print(f"[play] steps={len(rews)} mean_step_rew={float(np.mean(rews)):.3f} n_episodes={len(ep_rets)}", flush=True)
    _hist("a_v", av)
    _hist("a_w", aw)
    da = np.asarray(raw_da)
    print(
        f"[a-change] da mean={float(da.mean()):.3f} p90={float(np.percentile(da, 90)):.3f} "
        f"max={float(da.max()):.3f}  signflip_v={float(np.mean(np.diff(np.sign(av)) != 0)):.2f} "
        f"signflip_w={float(np.mean(np.diff(np.sign(aw)) != 0)):.2f}",
        flush=True,
    )
    dump_kw = dict(
        a_v=av,
        a_w=aw,
        da=da,
        auv_spd=np.asarray(auv_spd),
        tgt_spd=np.asarray(tgt_spd),
        gt_r=np.asarray(gt_range),
        rew=np.asarray(rews),
        done=np.asarray(dones_hist, dtype=np.bool_),
    )
    if gates:
        dump_kw["gate"] = np.asarray(gates)
    np.savez(args_cli.dump, **dump_kw)
    print(f"[a-dist] saved {args_cli.dump}", flush=True)
    summary_path = args_cli.summary or str(_Path(args_cli.dump).with_suffix(".json"))
    summary = summarize_play(dump_kw, label=args_cli.checkpoint)
    write_summary_json(summary, summary_path)
    print(format_summary(summary), flush=True)
    print(f"[eval] wrote {summary_path}", flush=True)
    print(
        f"[play] AUV |v_xy| mean={float(np.mean(auv_spd)):.3f} max={float(np.max(auv_spd)):.3f}  "
        f"TGT |v_xy| mean={float(np.mean(tgt_spd)):.3f} max={float(np.max(tgt_spd)):.3f}",
        flush=True,
    )
    if ep_rets:
        print(
            f"[play] ep_ret mean={float(np.mean(ep_rets)):.1f} "
            f"min={float(np.min(ep_rets)):.1f} max={float(np.max(ep_rets)):.1f} "
            f"ep_len mean={float(np.mean(ep_lens)):.1f}",
            flush=True,
        )
    else:
        print(f"[play] no full episode; partial_ret={float(cur_ret.mean()):.1f} len={int(cur_len.mean())}", flush=True)
    env.close()
    print("[play] DONE", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
