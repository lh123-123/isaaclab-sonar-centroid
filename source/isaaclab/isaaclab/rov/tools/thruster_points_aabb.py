"""Actual point AABB + PCA axis per thruster (world)."""

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


def pts_world(prim):
    out = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() != "Mesh":
            continue
        xf = cache.GetLocalToWorldTransform(p)
        a = UsdGeom.Mesh(p).GetPointsAttr().Get()
        if a:
            for pt in a:
                out.append(tuple(xf.Transform(Gf.Vec3d(pt))))
    return np.array(out)


i = 0
for c in wrapper.GetChildren():
    if "推进器装配" not in c.GetDisplayName():
        continue
    i += 1
    P = pts_world(c)
    mn, mx = P.min(0), P.max(0)
    size = mx - mn
    cen = (mn + mx) / 2
    X = P - P.mean(0)
    cov = X.T @ X / len(P)
    w, v = np.linalg.eigh(cov)
    print(f"#{i} n={len(P):6d} size={np.round(size,2)} center={np.round(cen,2)} eig={np.round(w,0)} axis={np.round(v[:,0],3)}")
