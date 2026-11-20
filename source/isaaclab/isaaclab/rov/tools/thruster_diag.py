"""Diagnostic: coordinate and radial distributions of thruster prototype points."""

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

# gather prototype-space points (identity)
P = []


def rec(p, xf):
    if p.GetTypeName() == "Mesh":
        a = UsdGeom.Mesh(p).GetPointsAttr().Get()
        if a:
            for pt in a:
                P.append(tuple(xf.Transform(Gf.Vec3d(pt))))
    child_xf = xf
    if p.GetTypeName() == "Xform":
        child_xf = xf * UsdGeom.Xformable(p).GetLocalTransformation(Usd.TimeCode.Default())
    for c in p.GetChildren():
        rec(c, child_xf)


rec(thr.GetPrototype(), Gf.Matrix4d(1))
P = np.array(P)
P = P - P.mean(0)

print("n:", len(P))
for k, name in enumerate("XYZ"):
    q = np.percentile(P[:, k], [0, 5, 25, 50, 75, 95, 100])
    print(f"{name}: {np.round(q, 1)}  std={P[:,k].std():.1f}")

for name, (i, j) in {"radial XY": (0, 1), "radial YZ": (1, 2), "radial ZX": (2, 0)}.items():
    r = np.sqrt(P[:, i] ** 2 + P[:, j] ** 2)
    q = np.percentile(r, [5, 25, 50, 75, 95, 100])
    print(f"{name}: r quantiles {np.round(q,1)}")

# unique-ish coordinate clustering along each axis (histogram peaks)
for k, name in enumerate("XYZ"):
    hist, edges = np.histogram(P[:, k], bins=20)
    peaks = [(round((edges[m] + edges[m + 1]) / 2, 1), int(hist[m])) for m in np.argsort(hist)[-4:]]
    print(f"{name} histogram top bins (center,count): {peaks}")
