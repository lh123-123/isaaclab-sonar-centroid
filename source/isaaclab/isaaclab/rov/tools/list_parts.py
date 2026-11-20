"""Inspect converted CAD USD: list parts by displayName with world bboxes."""

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

print("container:", wrapper.GetPath())
rows = []
for c in wrapper.GetChildren():
    name = c.GetDisplayName()
    rng = cache.ComputeWorldBound(c).ComputeAlignedRange()
    if rng.IsEmpty():
        continue
    mn, mx = rng.GetMin(), rng.GetMax()
    size = mx - mn
    cen = (mn + mx) / 2.0
    rows.append((name, c.GetPath().name, cen, size))

kw = sys.argv[2] if len(sys.argv) > 2 else ""
for name, pname, cen, size in rows:
    if kw and kw not in name:
        continue
    print(f"{name:28s} c=({cen[0]:8.2f},{cen[1]:8.2f},{cen[2]:8.2f}) s=({size[0]:7.2f},{size[1]:7.2f},{size[2]:7.2f})")

print("total top-level parts:", len(rows))
