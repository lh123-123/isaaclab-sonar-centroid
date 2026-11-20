"""World directions of each thruster's local axes (full local-to-world)."""

import sys
from pxr import Usd, UsdGeom, Gf

path = sys.argv[1]
stage = Usd.Stage.Open(path)
cache = UsdGeom.XformCache(Usd.TimeCode.Default())
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

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
    m = cache.GetLocalToWorldTransform(c)
    cols = []
    for k in range(3):
        v = Gf.Vec3d(m[0][k], m[1][k], m[2][k])
        n = v.GetNormalized()
        cols.append(n)
    rng = bbox_cache.ComputeWorldBound(c).ComputeAlignedRange()
    cen = (rng.GetMin() + rng.GetMax()) / 2.0
    size = rng.GetMax() - rng.GetMin()
    print(f"#{i} c=({cen[0]:7.2f},{cen[1]:7.2f},{cen[2]:7.2f}) s=({size[0]:6.2f},{size[1]:6.2f},{size[2]:6.2f})")
    for name, v in zip("XYZ", cols):
        print(f"     world dir of local {name}: ({v[0]:6.3f},{v[1]:6.3f},{v[2]:6.3f})")
