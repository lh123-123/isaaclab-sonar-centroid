"""Dump thruster assembly transforms + children for orientation analysis."""

import sys
from pxr import Usd, UsdGeom, Gf

path = sys.argv[1]
stage = Usd.Stage.Open(path)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])

root = stage.GetDefaultPrim()
wrapper = root
while True:
    ch = [c for c in wrapper.GetChildren() if c.GetTypeName() == "Xform"]
    if len(ch) == 1 and not wrapper.IsInstance():
        wrapper = ch[0]
    else:
        break

i = 0
for c in wrapper.GetChildren():
    if "推进器装配" not in c.GetDisplayName():
        continue
    i += 1
    rng = cache.ComputeWorldBound(c).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    cen = (mn + mx) / 2.0
    size = mx - mn
    print(f"\n=== thruster #{i} {c.GetName()} ({c.GetDisplayName()}) ===")
    print(f"  center=({cen[0]:.2f},{cen[1]:.2f},{cen[2]:.2f}) size=({size[0]:.2f},{size[1]:.2f},{size[2]:.2f})")
    xf = UsdGeom.Xformable(c)
    for op in xf.GetOrderedXformOps():
        print("  op:", op.GetOpName(), op.Get())
    # children within prototype
    if c.IsInstance():
        proto = c.GetPrototype()
        print("  prototype:", proto.GetPath())
        for ch in proto.GetChildren():
            n = ch.GetDisplayName() or ch.GetName()
            print("    child:", n, ch.GetTypeName(), "inst=", ch.IsInstance())
            if ch.IsInstance():
                for g in ch.GetPrototype().GetChildren()[:6]:
                    print("       g:", g.GetDisplayName() or g.GetName(), g.GetTypeName())
