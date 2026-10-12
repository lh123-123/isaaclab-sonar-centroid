# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""READ-ONLY root-cause probe: why is the Direct-v3-cam camera's render pose frozen?

Hypothesis under test
---------------------
``debug_auv_cam_frames.py`` and ``debug_auv_cam_probe.py`` showed the rendered image never
contains the target AUV even when it is ~1 m in front of the camera's *expected* pose, and that
the image does not change when the robot is teleported (floor band row stays 53, dark% stays
exactly 15.6%). Meanwhile ``TiledCamera.data.pos_w`` is frozen at the authored spawn pose
``(0.35, 0, 8.05)`` -- note: in WORLD coordinates, with no environment-origin offset applied.

Two candidate mechanisms, which need different fixes:

  (a) The camera prim is a child of the robot, and under ``replicate_physics=True`` PhysX writes
      robot poses into Fabric while the USD xform stays authored. The camera therefore inherits a
      frozen parent transform -> frozen render pose. (In this case the camera's USD *local*
      transform is correct; its *world* transform is stale.)
  (b) The offset is applied in a frame that does not compose with the env origin, so the camera
      is authored at world ``(0.35, 0, 8.05)`` regardless of the robot. (In this case the camera
      local transform is fine but the parent chain is wrong.)

Method
------
This script only READS. It prints, for ``Robot`` / ``Robot/warpauv_base_link/LookFollowCamera0`` /
``Target``:

* the USD local transform, the USD world transform (via ``UsdGeom.XformCache``), and whether the
  prim is instanceable / an instance / has a Fabric-backed xform;
* the simulated (Fabric/PhysX) pose that the same prim should be at;
* the same set of values again AFTER teleporting the robot, so we can see exactly which layers
  the simulation updates.

It also dumps the env-origin bookkeeping so we can tell where the origin is (not) applied.

Usage::

    ./scripts/debug_cam_xform_compose.sh --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _fmt(v, prec: int = 4) -> str:
    arr = np.asarray(v, dtype=np.float64).ravel()
    return "(" + ", ".join(f"{x:+.{prec}f}" for x in arr) + ")"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only USD transform composition probe.")
    parser.add_argument("--out_dir", type=str, default="logs/auv_tracking/v3_cam_xform")
    args, unknown = parser.parse_known_args()

    import torch

    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]] + unknown
    live_parser = argparse.ArgumentParser()
    live_parser.add_argument("--out_dir", type=str, default=args.out_dir)
    live_parser.add_argument("--hang_timeout", type=float, default=150.0)
    AppLauncher.add_app_launcher_args(live_parser)
    live_args = live_parser.parse_args()
    live_args.enable_cameras = True

    import faulthandler

    faulthandler.enable()
    if float(live_args.hang_timeout) > 0:
        faulthandler.dump_traceback_later(float(live_args.hang_timeout), exit=True)

    app_launcher = AppLauncher(live_args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_cam_env import AUVTrackingCamEnvCfg

    cfg = AUVTrackingCamEnvCfg()
    cfg.scene.num_envs = 1
    # Ship-config check: cam_sync_usd_pose is True by default, so the probe observes exactly what
    # training sees. The camera now lives inside the asset at
    # .../Robot/warpauv_base_link/LookFollowCamera0, so its USD world pose should equal the robot's
    # simulated pose composed with the baked mount offset (0.35, 0, 0.05) and the "world" rotation.
    cfg.cam_sync_usd_pose = True
    env = gym.make("Isaac-AUV-Tracking-Direct-v3-cam", cfg=cfg)
    raw = env.unwrapped
    dev = raw.device

    print("[xform] resetting...", flush=True)
    env.reset()
    print("[xform] reset done", flush=True)

    stage = raw.sim.stage
    paths = {
        "World": "/World",
        "envs": "/World/envs",
        "env_0": "/World/envs/env_0",
        "Robot(wrapper)": "/World/envs/env_0/Robot",
        "Camera": "/World/envs/env_0/Robot/warpauv_base_link/LookFollowCamera0",
        "Target": "/World/envs/env_0/Target",
        "Target(body)": "/World/envs/env_0/Target/warpauv_base_link",
    }
    # The sonar is parented to the *inner* rigid-body prim, not the spawn wrapper
    # (see shared_platform.py::_resolve_rigid_body_prim_path). Discover the robot's rigid-body
    # prim the same way so we can compare USD freshness on both levels.
    from isaaclab.sim.utils import get_all_matching_child_prims
    from pxr import UsdPhysics as _UsdPhysics

    body_prims = get_all_matching_child_prims(
        "/World/envs/env_0/Robot", predicate=lambda p: p.HasAPI(_UsdPhysics.RigidBodyAPI)
    )
    print(f"[xform] robot rigid-body prims (RigidBodyAPI): {[p.GetPath().pathString for p in body_prims]}", flush=True)
    if body_prims:
        paths["Robot(body)"] = body_prims[0].GetPath().pathString
        paths["SonarMount(would-be)"] = body_prims[0].GetPath().pathString + "/SonarMount"

    print("\n[xform] ================ static structure ================", flush=True)
    print(f"[xform] scene.env_origins (sim) = {raw.scene.env_origins.detach().cpu().numpy()}", flush=True)
    print(f"[xform] scene.env_prim_paths     = {raw.scene.env_prim_paths}", flush=True)

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for label, path in paths.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            print(f"[xform] {label:8s} {path:34s}  <MISSING>", flush=True)
            continue
        xf = UsdGeom.Xformable(prim)
        flags = []
        if prim.IsInstanceable():
            flags.append("instanceable")
        if prim.IsInstance():
            flags.append("INSTANCE")
        if prim.IsInPrototype():
            flags.append("in-prototype")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            flags.append("RigidBodyAPI")
        if xf:
            ops = xf.GetOrderedXformOps()
            ops_s = ",".join(o.GetOpName() for o in ops) if ops else "-"
        else:
            ops_s = "(not Xformable)"
        print(f"[xform] {label:8s} {path:34s}  {' '.join(flags) or '-':28s} ops=[{ops_s}]", flush=True)
        if xf:
            local = xf.GetLocalTransformation()
            lq = local.ExtractRotationQuat()
            lq_np = np.array([lq.GetReal(), *lq.GetImaginary()], dtype=np.float64)
            print(f"[xform]          local xform: T={_fmt(local.ExtractTranslation())} "
                  f"quat(wxyz)={_fmt(lq_np)}", flush=True)
        world = cache.GetLocalToWorldTransform(prim)
        print(f"[xform]          WORLD xform: T={_fmt(world.ExtractTranslation())}", flush=True)

    # camera prim USD optical attributes (validates the intrinsics we rely on)
    cam_prim = stage.GetPrimAtPath(paths["Camera"])
    if cam_prim and cam_prim.IsValid() and cam_prim.IsA(UsdGeom.Camera):
        cam = UsdGeom.Camera(cam_prim)
        print(
            f"[xform] camera USD: focalLength={cam.GetFocalLengthAttr().Get()}"
            f" hAperture={cam.GetHorizontalApertureAttr().Get()}"
            f" vAperture={cam.GetVerticalApertureAttr().Get()}"
            f" clip={cam.GetClippingRangeAttr().Get()}"
            f" proj={cam.GetProjectionAttr().Get()}",
            flush=True,
        )
    intr = raw._tiled_camera.data.intrinsic_matrices[0].detach().cpu().numpy()
    print(f"[xform] camera intrinsics K=\n{np.array2string(intr, precision=3)}", flush=True)

    def record(tag: str) -> dict:
        """Snapshot USD world transforms and simulated poses for the key prims."""
        sim = {
            "Robot": raw._robot.data.root_pos_w[0, :3].detach().cpu().numpy(),
            "Target": raw._target.data.root_pos_w[0, :3].detach().cpu().numpy(),
        }
        try:
            sim["Camera"] = raw._tiled_camera.data.pos_w[0, :3].detach().cpu().numpy()
        except Exception as exc:  # pragma: no cover
            sim["Camera"] = np.array([np.nan, np.nan, np.nan])
            print(f"[xform] camera pos_w unavailable: {exc}", flush=True)
        usd = {}
        for k, p in paths.items():
            pr = stage.GetPrimAtPath(p)
            if pr and pr.IsValid():
                usd[k] = np.array(cache.GetLocalToWorldTransform(pr).ExtractTranslation())
        # Camera pose derived from the robot's SIMULATED pose + mount offset (the pose the
        # renderer *should* use if the camera rides the robot).
        import torch as _t

        from isaaclab.utils.math import quat_apply

        off = _t.tensor(cfg.tiled_camera.offset.pos, dtype=_t.float32)
        rq = raw._robot.data.root_quat_w[0].detach().cpu()
        derived = (raw._robot.data.root_pos_w[0, :3].detach().cpu()
                   + quat_apply(rq.unsqueeze(0), off.unsqueeze(0)).squeeze(0)).numpy()
        print(f"\n[xform] ---------------- {tag} ----------------", flush=True)
        for k in usd:
            line = f"[xform] {k:20s} usd_world={_fmt(usd[k])}"
            if k in sim:
                line += f"  sim={_fmt(sim[k])}  |sim-usd|={np.linalg.norm(sim[k] - usd[k]):8.3f}"
            print(line, flush=True)
        print(f"[xform] expected camera from robot sim pose + offset = {_fmt(derived)}", flush=True)
        print(f"[xform]   usd_world(Camera) - derived = {_fmt(usd['Camera'] - derived)}", flush=True)
        print(f"[xform]   sim(Camera)       - derived = {_fmt(sim['Camera'] - derived)}", flush=True)
        return {"sim": sim, "usd": usd, "derived": derived}

    before = record("after reset")

    # Teleport the robot and step so the sim definitely writes its new pose.
    e0 = torch.tensor([0], device=dev, dtype=torch.long)
    zero = torch.zeros((raw.num_envs, raw.action_space.shape[-1]), device=dev)
    new_pos = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0]], device=dev)
    print("\n[xform] teleporting Robot -> (0,0,2) and stepping 3x...", flush=True)
    for _ in range(3):
        raw._robot.write_root_pose_to_sim(new_pos, e0)
        env.step(zero)

    after = record("after teleport Robot->(0,0,2)")

    print("\n[xform] ================ verdict ================", flush=True)
    for k in ("Robot", "Camera", "Target"):
        d_usd = np.linalg.norm(after["usd"][k] - before["usd"][k])
        d_sim = np.linalg.norm(after["sim"][k] - before["sim"][k])
        print(
            f"[xform] {k:7s} moved: sim={d_sim:8.3f} m  usd={d_usd:8.3f} m"
            f"   -> {'USD follows sim' if d_usd > 0.5 else 'USD FROZEN (sim moved, USD did not)'}",
            flush=True,
        )
    print(
        "[xform] interpretation: if Camera USD is frozen while Camera sim would have moved, the\n"
        "         camera inherits a stale parent xform (mechanism a). If Camera usd_world equals\n"
        "         offset with no env-origin contribution, the parent chain is wrong (mechanism b).",
        flush=True,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    import os
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    os._exit(0)
