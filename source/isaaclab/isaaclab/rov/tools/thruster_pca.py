"""PCA on thruster prototype vertices to find symmetry (thrust) axis."""

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

# gather unique prototypes for thruster assemblies
protos = {}
order = []
for c in wrapper.GetChildren():
    if "推进器装配" not in c.GetDisplayName():
        continue
    proto = c.GetPrototype() if c.IsInstance() else None
    if proto is None:
        continue
    key = str(proto.GetPath())
    if key not in protos:
        protos[key] = proto
        order.append(key)


def gather_points(prim):
    pts = []

    def rec(p, xf):
        if p.GetTypeName() == "Mesh":
            m = UsdGeom.Mesh(p)
            a = m.GetPointsAttr().Get()
            if a:
                for pt in a:
                    pts.append(tuple(xf.Transform(Gf.Vec3d(pt))))
        child_xf = xf
        if p.GetTypeName() == "Xform":
            child_xf = xf * UsdGeom.Xformable(p).GetLocalTransformation(Usd.TimeCode.Default())
        for ch in p.GetChildren():
            rec(ch, child_xf)

    rec(prim, Gf.Matrix4d(1))
    return np.array(pts)


for key in order:
    proto = protos[key]
    pts = gather_points(proto)
    if len(pts) == 0:
        continue
    mean = pts.mean(axis=0)
    X = pts - mean
    cov = X.T @ X / len(pts)
    w, v = np.linalg.eigh(cov)
    # eigenvalues ascending; eigenvectors columns
    axis = v[:, 0]  # smallest variance
    print(f"\nproto {key}: points={len(pts)}")
    print("  eigenvalues:", np.round(w, 2))
    print("  bbox size:", np.round(pts.max(axis=0) - pts.min(axis=0), 2))
    print("  smallest-var axis (local):", np.round(axis, 3))
    for k in range(3):
        print(f"  eig{k}: value={w[k]:.1f} vec={np.round(v[:,k],3)}")
