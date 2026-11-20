"""Determine thruster local geometry bbox and map axis to world."""

import sys
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


def local_bbox(prim):
    mn = Gf.Vec3d(1e30, 1e30, 1e30)
    mx = Gf.Vec3d(-1e30, -1e30, -1e30)
    found = False

    def rec(p, xf):
        nonlocal mn, mx, found
        if p.GetTypeName() == "Mesh":
            m = UsdGeom.Mesh(p)
            pts = m.GetPointsAttr().Get()
            if pts:
                found = True
                for pt in pts:
                    q = xf.Transform(Gf.Vec3d(pt))
                    for i in range(3):
                        mn[i] = min(mn[i], q[i])
                        mx[i] = max(mx[i], q[i])
        # accumulate child local transform if present
        child_xf = xf
        x = UsdGeom.Xformable(p)
        if p.GetTypeName() == "Xform":
            m = x.GetLocalTransformation(Usd.TimeCode.Default())
            child_xf = xf * m
        for c in p.GetChildren():
            rec(c, child_xf)

    rec(prim, Gf.Matrix4d(1))
    return (mn, mx) if found else (None, None)


seen = set()
i = 0
for c in wrapper.GetChildren():
    if "推进器装配" not in c.GetDisplayName():
        continue
    i += 1
    proto = c.GetPrototype() if c.IsInstance() else None
    key = proto.GetPath() if proto else None
    print(f"\n=== thruster #{i} {c.GetName()} ===")
    if proto and key not in seen:
        seen.add(key)
        # local bbox from prototype children (prototype itself is identity)
        mn = Gf.Vec3d(1e30, 1e30, 1e30); mx = Gf.Vec3d(-1e30, -1e30, -1e30)
        for ch in proto.GetChildren():
            a, b = local_bbox(ch)
            if a:
                for k in range(3):
                    mn[k] = min(mn[k], a[k]); mx[k] = max(mx[k], b[k])
        print("  prototype:", key, "local bbox size:", tuple(round(v, 2) for v in (mx - mn)))
    # instance transform columns
    xf = UsdGeom.Xformable(c)
    ops = xf.GetOrderedXformOps()
    if ops:
        m = ops[0].Get()
        cols = [Gf.Vec3d(m[0][k], m[1][k], m[2][k]) for k in range(3)]
        trans = Gf.Vec3d(m[3][0], m[3][1], m[3][2])
        print("  world local-axes:", [tuple(round(v, 3) for v in col.GetNormalized()) for col in cols])
