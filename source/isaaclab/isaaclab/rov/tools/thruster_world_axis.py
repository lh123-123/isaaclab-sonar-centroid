"""Per-thruster world-space PCA to find thrust axis direction."""

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


def gather_world_points(prim):
    pts = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() != "Mesh":
            continue
        xf = cache.GetLocalToWorldTransform(p)
        a = UsdGeom.Mesh(p).GetPointsAttr().Get()
        if a:
            for pt in a:
                pts.append(tuple(xf.Transform(Gf.Vec3d(pt))))
    return np.array(pts)


i = 0
for c in wrapper.GetChildren():
    if "推进器装配" not in c.GetDisplayName():
        continue
    i += 1
    pts = gather_world_points(c)
    if len(pts) == 0:
        print(f"#{i} {c.GetName()}: NO POINTS")
        continue
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(c).ComputeAlignedRange()
    cen = (bb.GetMin() + bb.GetMax()) / 2.0
    size = bb.GetMax() - bb.GetMin()
    mean = pts.mean(axis=0)
    X = pts - mean
    cov = X.T @ X / len(pts)
    w, v = np.linalg.eigh(cov)
    axis = v[:, 0]
    if axis[2] < 0:
        axis = -axis
    print(f"#{i} {c.GetName()}  c=({cen[0]:7.2f},{cen[1]:7.2f},{cen[2]:7.2f}) "
          f"s=({size[0]:6.2f},{size[1]:6.2f},{size[2]:6.2f})  "
          f"axis=({axis[0]:6.3f},{axis[1]:6.3f},{axis[2]:6.3f})  eig={np.round(w,0)}")
