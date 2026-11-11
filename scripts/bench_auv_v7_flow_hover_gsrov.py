"""Throughput probe for the GS-ROV station-keeping env.

Times env construction and N random-action steps for both the CAD hull and the cuboid
proxy so we can tell whether the prepared asset is cheap enough to train on.

Usage:
    python bench_auv_v7_flow_hover_gsrov.py --num_envs 64 --steps 40
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GS-ROV v7 flow-hover throughput probe.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--compare-proxy", action="store_true", help="Also time the cuboid proxy.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import torch

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)


def run(use_real_usd: bool) -> None:
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.use_real_usd = use_real_usd

    t0 = time.perf_counter()
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    build = time.perf_counter() - t0

    env.reset()
    act = torch.zeros(env.num_envs, 3)
    for _ in range(3):  # warm up shaders / compilation
        env.step(act)

    t0 = time.perf_counter()
    for _ in range(args_cli.steps):
        env.step(act)
    elapsed = time.perf_counter() - t0

    label = "gsrov_usd" if use_real_usd else "cuboid_proxy"
    fps = args_cli.steps * env.num_envs / elapsed
    print(
        f"[bench] {label:12s} envs={env.num_envs:4d} build={build:6.2f}s "
        f"step={1000.0 * elapsed / args_cli.steps:7.2f}ms "
        f"throughput={fps:9.1f} env-steps/s",
        flush=True,
    )
    env.close()


def main() -> None:
    run(True)
    if args_cli.compare_proxy:
        run(False)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
