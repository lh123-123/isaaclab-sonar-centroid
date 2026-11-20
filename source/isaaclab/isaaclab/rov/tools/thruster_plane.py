"""Detect duct plane: compare radial spread in XY / YZ / ZX for thruster prototype."""

import sys
import numpy as np
from pxr import Usd, UsdGeom, Gf

path = sys.argv[1]
stage = Usd.Stage.Open(path)

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

# collect prototype-space points (drop instance transform)
proto = thr.GetPrototype()
pts = []


def rec(p, xf):
    if p.GetTypeName() == "Mesh":
        a = UsdGeom.Mesh(p).GetPointsAttr().Get()
        if a:
            for pt in a:
                pts.append(tuple(xf.Transform(Gf.Vec3d(pt))))
    child_xf = xf
    if p.GetTypeName() == "Xform":
        child_xf = xf * UsdGeom.Xformable(p).GetLocalTransformation(Usd.TimeCode.Default())
    for c in p.GetChildren():
        rec(c, child_xf)


rec(proto, Gf.Matrix4d(1))
P = np.array(pts)
print("points:", len(P), "local size:", np.round(P.max(0) - P.min(0), 2))
for name, (i, j) in {"XY (axis Z)": (0, 1), "YZ (axis X)": (1, 2), "ZX (axis Y)": (2, 0)}.items():
    r = np.sqrt(P[:, i] ** 2 + P[:, j] ** 2)
    rmax = r.max()
    # how many points lie near a circle of radius ~0.8*rmax? measure relative std of outer ring
    outer = r[r > 0.75 * rmax]
    print(f"  {name}: rmax={rmax:6.2f} outer_count={len(outer):6d} outer_r_std={outer.std():5.2f} "
          f"axial_extent={np.ptp(P[:, 3 - i - j]):6.2f}")
