"""Per-mesh local bbox inside one thruster prototype (world frame of first thruster)."""

import sys
import numpy as np
from pxr import Usd, UsdGeom, Gf

path = sys.argv[1]
stage = Usd.Stage.Open(path)
cache = UsdGeom.XformCache(Usd.TimeCode.Default())

root = stage.GetDefaultPrim()
wrapper = root
while True:
    ch = [c for c in wrapper.GetChildren() if c.GetTypeName() == "Xform"]
    if len(ch) == 1 and not wrapper.IsInstance():
        wrapper = ch[0]
    else:
        break

thr = None
for c in wrapper.GetChildren():
    if "推进器装配" in c.GetDisplayName():
        thr = c
        break

print("thruster:", thr.GetName())
for p in Usd.PrimRange(thr, Usd.TraverseInstanceProxies()):
    if p.GetTypeName() != "Mesh":
        continue
    xf = cache.GetLocalToWorldTransform(p)
    pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
    if not pts:
        continue
    arr = np.array([tuple(xf.Transform(Gf.Vec3d(pt))) for pt in pts])
    mn, mx = arr.min(axis=0), arr.max(axis=0)
    size = mx - mn
    cen = (mn + mx) / 2
    # rotational symmetry check: closest two dims
    s = np.sort(size)
    print(f"  mesh {p.GetName():28s} n={len(pts):6d} size={np.round(size,2)} center={np.round(cen,2)}")
