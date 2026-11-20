"""Render a USD stage to a PNG thumbnail so asset problems can be eyeballed headlessly.

Usage:
    python render_usd_thumb.py MODEL.usd OUT.png [--res 640]
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render a USD thumbnail.")
parser.add_argument("stage", type=str)
parser.add_argument("out", type=str)
parser.add_argument("--res", type=int, default=640)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import omni.usd
import omni.replicator.core as rep
import omni.kit.app
from pxr import Gf, Usd, UsdGeom, UsdLux

app = omni.kit.app.get_app()


def bbox(stage) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    return rng.GetMin(), rng.GetMax()


def main() -> None:
    stage_path = os.path.abspath(args_cli.stage)
    omni.usd.get_context().open_stage(stage_path)
    for _ in range(60):
        app.update()

    stage = omni.usd.get_context().get_stage()
    mn, mx = bbox(stage)
    center = (mn + mx) / 2.0
    radius = max((mx - mn).GetLength() / 2.0, 1e-3)
    print(f"[thumb] bbox {mn} .. {mx}  radius {radius:.3f}", flush=True)

    # Camera: 3/4 view from +X/+Y/+Z, framed on the bbox.
    dist = radius * 3.0
    eye = Gf.Vec3d(center[0] + dist * 0.7, center[1] - dist * 0.7, center[2] + dist * 0.55)
    look = (center - eye).GetNormalized()
    up = Gf.Vec3d(0, 0, 1)
    right = Gf.Cross(look, up).GetNormalized()
    up2 = Gf.Cross(right, look).GetNormalized()
    rot = Gf.Matrix4d(
        right[0], right[1], right[2], 0,
        up2[0], up2[1], up2[2], 0,
        -look[0], -look[1], -look[2], 0,
        0, 0, 0, 1,
    )

    cam_path = "/World/ThumbCam"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateFocalLengthAttr(24.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(radius * 0.01, radius * 100.0))
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(rot)
    xf.AddTranslateOp().Set(eye)

    dome = UsdLux.DomeLight.Define(stage, "/World/ThumbDome")
    dome.CreateIntensityAttr(1500.0)
    key = UsdLux.SphereLight.Define(stage, "/World/ThumbKey")
    UsdGeom.Xformable(key.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(center[0] + dist, center[1] - dist, center[2] + dist)
    )
    key.CreateIntensityAttr(5.0e5)
    key.CreateRadiusAttr(radius * 0.5)

    for _ in range(20):
        app.update()

    res = (args_cli.res, args_cli.res)
    rp = rep.create.render_product(cam_path, res)
    writer = rep.WriterRegistry.get("BasicWriter")
    out_dir = os.path.abspath(os.path.dirname(args_cli.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args_cli.out))[0]
    writer.initialize(output_dir=out_dir, rgb=True)
    writer.attach([rp])

    rep.orchestrator.run_until_complete(num_frames=4, warmup_frames=4)

    produced = [f for f in os.listdir(out_dir) if f.startswith("rgb_") and f.endswith(".png")]
    if produced:
        src = os.path.join(out_dir, sorted(produced)[-1])
        os.replace(src, os.path.abspath(args_cli.out))
        print(f"[thumb] wrote {os.path.abspath(args_cli.out)}", flush=True)
    else:
        print("[thumb] no rgb written", flush=True)

    rep.orchestrator.stop()
    app.close()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
