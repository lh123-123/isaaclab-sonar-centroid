# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Open the editable *copy* of the WarpAUV asset in the Isaac Sim GUI, for manual prim editing.

Which file this opens
---------------------
``source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/data/warpauv_cam/warpauv.usd``

That is a byte-identical duplicate of ``data/warpauv/warpauv.usd`` (the asset every current
experiment uses via ``assets/warpauv.py``), so prims added here cannot perturb v0/v1/v2. The copy
is self-contained -- ``UsdUtils.ComputeAllDependencies`` resolves to the file itself, with no
external references and no absolute paths -- so edits saved into it stay local.

Why open the asset directly (instead of spawning it through ``InteractiveScene``)
--------------------------------------------------------------------------------
This follows ``scripts/tools/convert_urdf.py``, the workspace's "open a USD then keep the GUI
alive" pattern. Opening the asset as the *root* layer makes the edit target unambiguous, so a
plain ``File > Save`` (Ctrl+S) lands in the asset layer. If we instead spawned the asset as a
reference inside a generated scene, a new prim would be authored into the session layer and would
silently vanish on close.

The stage has no light of its own, so a dome light is authored into the *session* layer (never
into the asset) purely so the hull is visible while you work. It is not saved with the asset.

Usage::

    ./scripts/open_warpauv_cam_asset.sh              # GUI, ready for you to add prims
    ./scripts/open_warpauv_cam_asset.sh --headless   # open + report prim tree, then exit
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Open the editable WarpAUV copy for manual prim editing.")
parser.add_argument(
    "--usd_path",
    type=str,
    default=None,
    help="USD to open. Defaults to the task-local copy (data/warpauv_cam/warpauv.usd).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app
from pxr import Gf, Usd, UsdLux, UsdPhysics


def _default_usd_path() -> str:
    """Resolve the task-local WarpAUV copy next to this repo's task sources."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(
        repo_root,
        "source",
        "isaaclab_tasks",
        "isaaclab_tasks",
        "direct",
        "isaac_auv_tracking",
        "data",
        "warpauv_cam",
        "warpauv.usd",
    )


def _add_viewport_light(stage: Usd.Stage) -> None:
    """Author a dome light into the session layer so the hull is visible but the asset stays clean."""
    session = stage.GetSessionLayer()
    if session is None:
        return
    previous = stage.GetEditTarget()
    try:
        stage.SetEditTarget(Usd.EditTarget(session))
        light_path = "/ViewportLight"
        if not stage.GetPrimAtPath(light_path).IsValid():
            light = UsdLux.DomeLight.Define(stage, light_path)
            light.CreateIntensityAttr(3000.0)
            light.CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))
    finally:
        stage.SetEditTarget(previous)


def _print_prim_tree(stage: Usd.Stage, max_depth: int = 3) -> None:
    """Print the asset hierarchy so the prim path to extend is obvious."""
    print("[asset] ---- prim tree ----", flush=True)
    for prim in stage.Traverse():
        depth = len(prim.GetPath().pathString.strip("/").split("/")) - 1
        if depth > max_depth:
            continue
        flags = []
        if prim.IsInstance():
            flags.append("instance")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            flags.append("RigidBodyAPI")
        if prim.GetTypeName() == "Camera":
            flags.append("Camera")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"[asset] {'  ' * depth}{prim.GetPath()}{suffix}", flush=True)


def main() -> None:
    usd_path = args_cli.usd_path or _default_usd_path()
    usd_path = os.path.abspath(usd_path)
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"WarpAUV copy not found: {usd_path}")

    # Determine if there is a GUI to update (same check as scripts/tools/convert_urdf.py).
    carb_settings_iface = carb.settings.get_settings()
    local_gui = carb_settings_iface.get("/app/window/enabled")
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    print(f"[asset] opening: {usd_path}", flush=True)
    stage_utils.open_stage(usd_path)
    stage = stage_utils.get_current_stage()
    _add_viewport_light(stage)

    default_prim = stage.GetDefaultPrim()
    print(f"[asset] default prim: {default_prim.GetPath() if default_prim else '<none>'}", flush=True)
    print("[asset] add your prim under the rigid-body link, e.g.", flush=True)
    print("[asset]   /warpauv/warpauv_base_link/<YourPrim>", flush=True)
    print("[asset] then File > Save (Ctrl+S) to write it into the asset layer.", flush=True)
    _print_prim_tree(stage)

    if local_gui or livestream_gui:
        print("[asset] GUI is up -- add your prim, then Ctrl+S. Close the window to exit.", flush=True)
        app = omni.kit.app.get_app_interface()
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                app.update()
    else:
        print("[asset] headless: prim tree reported, exiting.", flush=True)


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
