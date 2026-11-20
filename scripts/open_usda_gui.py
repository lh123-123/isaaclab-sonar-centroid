#!/usr/bin/env python3
"""Open a USD/USDA stage in the Isaac Sim GUI and keep the window open.

Usage:
  ./isaaclab.sh -p scripts/open_usda_gui.py <file.usda|.usd|.usdc>

The launcher CLI treats the first positional arg as an experience (.kit) file,
so passing a stage path there does NOT open the scene. This script instead
starts a normal GUI SimulationApp and calls omni.usd open_stage explicitly,
then keeps the Kit event loop running so the window stays interactive.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Open a USD stage in Isaac Sim GUI.")
    parser.add_argument("stage", type=str, help="Path to .usda/.usd/.usdc to open.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--experience",
        default="/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.full.kit",
        help="Kit experience file. Full is the stable GUI; the default "
        "'Isaac-Sim Python' experience can crash the RTX render thread on this GPU.",
    )
    args_cli, _ = parser.parse_known_args()

    stage_path = os.path.abspath(args_cli.stage)
    if not os.path.exists(stage_path):
        raise FileNotFoundError(f"stage not found: {stage_path}")

    from isaacsim import SimulationApp

    sim_cfg = {"headless": False, "width": args_cli.width, "height": args_cli.height}
    exp = args_cli.experience
    if os.path.isfile(exp):
        sim_cfg["experience"] = exp
    else:
        print(f"[open_usda_gui] experience not found ({exp}), using default", flush=True)

    simulation_app = SimulationApp(sim_cfg)

    import omni.usd

    ok = omni.usd.get_context().open_stage(stage_path)
    if ok is False:
        raise RuntimeError(f"failed to open stage: {stage_path}")
    print(f"[open_usda_gui] stage opened: {stage_path}", flush=True)

    # Keep the interactive window alive.
    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()