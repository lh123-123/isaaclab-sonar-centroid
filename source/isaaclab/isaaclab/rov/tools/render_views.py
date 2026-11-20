"""Render a USD model from several views with Isaac Sim (headless) via Replicator."""

import os
import sys

usd_path = os.path.abspath(sys.argv[1])
out_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else "/tmp/rov_views"
os.makedirs(out_dir, exist_ok=True)

from isaacsim.simulation_app import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": 960, "height": 720})
app.update()

import omni.usd
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, Gf

ctx = omni.usd.get_context()
ctx.open_stage(usd_path)
for _ in range(10):
    app.update()

stage = ctx.get_stage()
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
rng = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
mn, mx = rng.GetMin(), rng.GetMax()
cen = (mn + mx) / 2.0
size = mx - mn
radius = 0.5 * max(size[0], size[1], size[2])
print("bbox:", mn, mx, "center:", cen, "size:", size)

# ensure lights + a ground-less environment
rep.create.light(light_type="Sphere", intensity=3000, position=(cen[0] + radius, cen[1] + radius, cen[2] + 2 * radius), scale=radius)

views = {
    "iso": (cen + Gf.Vec3d(1.6, -1.6, 1.2) * radius * 2.2, (0, 0, 0)),
    "top": (cen + Gf.Vec3d(0.0, 0.0, radius * 4.0), (0, 0, 0)),
    "front_x": (cen + Gf.Vec3d(radius * 4.0, 0.0, 0.0), (0, 0, 0)),
    "front_y": (cen + Gf.Vec3d(0.0, radius * 4.0, 0.0), (0, 0, 0)),
}

for name, (pos, rot) in views.items():
    cam = rep.create.camera(position=(pos[0], pos[1], pos[2]), look_at=(cen[0], cen[1], cen[2]))
    rp = rep.create.render_product(cam, (960, 720))
    view_dir = os.path.join(out_dir, name)
    os.makedirs(view_dir, exist_ok=True)
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=view_dir, rgb=True)
    writer.attach([rp])
    for _ in range(8):
        app.update()
    rep.orchestrator.run()
    for _ in range(8):
        app.update()
    rep.orchestrator.wait_until_complete()
    print("rendered", name)
    writer.detach()
    rp.destroy()
    app.update()

app.close()
print("DONE", out_dir)
