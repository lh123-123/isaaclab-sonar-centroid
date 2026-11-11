#!/usr/bin/env python3
"""GUI playback plus an Isaac window of the optical flow the policy sees.

The viewport stays on the ROV camera. A second Kit window, "v7 optical flow",
shows the 64x64 camera, the flow direction (hue) and strength (brightness),
and the magnitude. Black flow means the picture is still.

  ./isaaclab.sh -p scripts/play_auv_v7_flow_vis_ui.py --checkpoint PATH
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play v7 and show the live optical-flow field.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1792)
parser.add_argument("--pace_hz", type=float, default=15.0)
parser.add_argument("--flow_scale_px", type=float, default=2.0, help="pixels that map to full brightness")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import numpy as np
import omni.ui as ui
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

_PANEL = 384


def _hsv_to_rgb(hue: np.ndarray, value: np.ndarray) -> np.ndarray:
    """hue and value in [0, 1]. Saturation is full. Returns uint8 RGB."""
    h = np.mod(hue, 1.0)
    v = np.clip(value, 0.0, 1.0)
    sector = (h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = np.zeros_like(v)
    q = v * (1.0 - f)
    t = v * f
    choices_r = np.stack([v, q, p, p, t, v], axis=0)
    choices_g = np.stack([t, v, v, q, p, p], axis=0)
    choices_b = np.stack([p, p, t, v, v, q], axis=0)
    rgb = np.stack(
        [np.choose(sector, choices_r), np.choose(sector, choices_g), np.choose(sector, choices_b)],
        axis=-1,
    )
    return (rgb * 255.0).astype(np.uint8)


def _up(img: np.ndarray, factor: int = 6) -> np.ndarray:
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def _to_rgba(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    return np.dstack((rgb, alpha))


def _push(provider: ui.ByteImageProvider, rgb: np.ndarray, held: list) -> None:
    rgba = np.ascontiguousarray(_to_rgba(rgb))
    held.append(rgba)
    del held[:-3]
    provider.set_bytes_data(rgba.flatten().data, [rgba.shape[1], rgba.shape[0]])


def _camera_rgb(obs) -> np.ndarray:
    image = np.asarray(obs["image"])
    if image.ndim == 4:
        image = image[0]
    rgb = np.asarray(image[:3], dtype=np.float32)
    if float(rgb.max()) > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)


def _flow_rgb(du_px: np.ndarray, dv_px: np.ndarray, scale_px: float) -> tuple[np.ndarray, np.ndarray, float]:
    mag = np.sqrt(du_px * du_px + dv_px * dv_px)
    hue = (np.arctan2(dv_px, du_px) + np.pi) / (2.0 * np.pi)
    value = np.clip(mag / max(scale_px, 1e-3), 0.0, 1.0)
    color = _hsv_to_rgb(hue, value)
    gray = np.repeat((value * 255.0).astype(np.uint8)[..., None], 3, axis=-1)
    return color, gray, float(mag.mean())


def main() -> None:
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.use_real_usd = False
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
                print(f"[v7-flow-ui] viewport camera -> {cam_path}", flush=True)
        except Exception as exc:
            print(f"[v7-flow-ui] viewport camera switch failed: {exc}", flush=True)

    print(f"[v7-flow-ui] loading {args_cli.checkpoint}", flush=True)
    agent = SAC.load(args_cli.checkpoint, vec, print_system_info=False)
    obs = vec.reset()
    fw = float(raw.cfg.flow_render_width)
    fh = float(raw.cfg.flow_render_height)

    blank = np.zeros((_PANEL, _PANEL, 3), dtype=np.uint8)
    providers = [ui.ByteImageProvider() for _ in range(3)]
    held: list[np.ndarray] = []
    for provider in providers:
        _push(provider, blank, held)
    window = ui.Window("v7 optical flow", width=_PANEL * 3 + 24, height=_PANEL + 80)
    with window.frame:
        with ui.VStack(spacing=4):
            caption = ui.Label("camera | flow direction and strength | magnitude. black = still")
            with ui.HStack(spacing=4):
                for provider in providers:
                    ui.ImageWithProvider(provider, width=_PANEL, height=_PANEL)
    print("[v7-flow-ui] flow window created inside Isaac.", flush=True)

    t0 = time.perf_counter()
    pace_dt = 1.0 / args_cli.pace_hz if args_cli.pace_hz > 0 else 0.0
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
        actions = np.asarray(actions, dtype=np.float32).reshape(vec.num_envs, -1)
        with torch.inference_mode():
            obs, _rewards, _dones, _infos = vec.step(actions)

        fi = raw._flow_image[0].detach().cpu().numpy()
        du_px = fi[0] * fw
        dv_px = fi[1] * fh
        flow_rgb, mag_rgb, mean_mag = _flow_rgb(du_px, dv_px, float(args_cli.flow_scale_px))
        rms = float(raw._flow_rms[0].item())
        _push(providers[0], _up(_camera_rgb(obs)), held)
        _push(providers[1], _up(flow_rgb), held)
        _push(providers[2], _up(mag_rgb), held)
        caption.text = (
            f"camera | flow hue (rms {rms:.2f} px) | magnitude (mean {mean_mag:.2f} px). black = still"
        )

        if pace_dt > 0.0:
            sleep_for = (t0 + (step + 1) * pace_dt) - time.perf_counter()
            if sleep_for > 0.0:
                time.sleep(sleep_for)
        if step % 30 == 0:
            print(
                f"[v7-flow-ui] step={step:4d} flow_rms={rms:.3f}px "
                f"mean_du={float(du_px.mean()):+.3f} mean_dv={float(dv_px.mean()):+.3f}",
                flush=True,
            )

    env.close()
    print("[v7-flow-ui] done", flush=True)


if __name__ == "__main__":
    main()
