"""Register each thruster instance to the prototype via Kabsch to recover rotation.

Then map the prototype thrust axis (local +Y) to world.

Usage:
    conda run -n isaaclab python thruster_kabsch.py MODEL.usd
"""

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

thrusters = [c for c in wrapper.GetChildren() if "推进器装配" in c.GetDisplayName()]


def gather_world(prim):
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


def gather_local(prim):
    out = []

    def rec(p, xf):
        if p.GetTypeName() == "Mesh":
            a = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if a:
                for pt in a:
                    out.append(tuple(xf.Transform(Gf.Vec3d(pt))))
        child_xf = xf
        if p.GetTypeName() == "Xform":
            child_xf = xf * UsdGeom.Xformable(p).GetLocalTransformation(Usd.TimeCode.Default())
        for c in p.GetChildren():
            rec(c, child_xf)

    rec(prim, Gf.Matrix4d(1))
    return np.array(out)


proto = thrusters[0].GetPrototype()
L = gather_local(proto)
Lc = L - L.mean(0)
H = Lc.T @ Lc

for i, c in enumerate(thrusters, 1):
    W = gather_world(c)
    if len(W) != len(L):
        print(f"#{i}: point count mismatch {len(W)} vs {len(L)}")
        continue
    Wc = W - W.mean(0)
    # Kabsch: R = V U^T from SVD(Wc^T Lc) mapping Lc -> Wc
    C = Wc.T @ Lc
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    # residual
    resid = np.sqrt(((Lc @ R.T - Wc) ** 2).sum(1).mean())
    axis = R @ np.array([0.0, 1.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    ctr = W.mean(0)
    print(f"#{i} {c.GetName():34s} center=({ctr[0]:7.2f},{ctr[1]:7.2f},{ctr[2]:7.2f}) "
          f"resid={resid:6.3f} thrust_axis(localY->world)=({axis[0]:6.3f},{axis[1]:6.3f},{axis[2]:6.3f})")
    print("     R=", np.round(R, 4).tolist())
