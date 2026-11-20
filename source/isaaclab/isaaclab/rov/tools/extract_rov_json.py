"""Extract ROV geometry + 8 thruster transforms into a JSON config.

Usage:
    conda run -n isaaclab python extract_rov_json.py MODEL.usd OUT.json
"""

import json
import sys
from pxr import Usd, UsdGeom, Gf

usd_path = sys.argv[1]
out_path = sys.argv[2]

stage = Usd.Stage.Open(usd_path)
cache = UsdGeom.XformCache(Usd.TimeCode.Default())
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

root = stage.GetDefaultPrim()
wrapper = root
while True:
    ch = [c for c in wrapper.GetChildren() if c.GetTypeName() == "Xform"]
    if len(ch) == 1 and not wrapper.IsInstance():
        wrapper = ch[0]
    else:
        break

# --- whole-model stats ---
rng = bbox_cache.ComputeWorldBound(root).ComputeAlignedRange()
gmn, gmx = rng.GetMin(), rng.GetMax()


def mesh_volume_centroid(mesh_prim, xf):
    m = UsdGeom.Mesh(mesh_prim)
    pts = m.GetPointsAttr().Get()
    faces = m.GetFaceVertexIndicesAttr().Get()
    counts = m.GetFaceVertexCountsAttr().Get()
    if not pts or not faces or not counts:
        return 0.0, Gf.Vec3d(0, 0, 0)
    vol = 0.0
    csum = Gf.Vec3d(0, 0, 0)
    idx = 0
    for c in counts:
        if c == 3:
            tris = [(faces[idx], faces[idx + 1], faces[idx + 2])]
        elif c == 4:
            tris = [(faces[idx], faces[idx + 1], faces[idx + 2]),
                    (faces[idx], faces[idx + 2], faces[idx + 3])]
        else:
            tris = [(faces[idx], faces[idx + i], faces[idx + i + 1]) for i in range(1, c - 1)]
        for a, b, d in tris:
            A = xf.Transform(Gf.Vec3d(pts[a])); B = xf.Transform(Gf.Vec3d(pts[b])); C = xf.Transform(Gf.Vec3d(pts[d]))
            tv = Gf.Dot(A, Gf.Cross(B, C)) / 6.0
            vol += tv
            csum += ((A + B + C) / 4.0) * tv
        idx += c
    return vol, (csum / vol if abs(vol) > 1e-9 else Gf.Vec3d(0, 0, 0))


total_vol = 0.0
wsum = Gf.Vec3d(0, 0, 0)
nmesh = 0
ntri = 0
for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
    if prim.GetTypeName() != "Mesh":
        continue
    xf = cache.GetLocalToWorldTransform(prim)
    vol, cen = mesh_volume_centroid(prim, xf)
    total_vol += vol
    wsum += cen * vol
    nmesh += 1
    counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
    if counts:
        ntri += sum(max(0, c - 2) for c in counts)

com = wsum / total_vol if abs(total_vol) > 1e-9 else Gf.Vec3d(0, 0, 0)

# --- thrusters ---
thrusters = []
for c in wrapper.GetChildren():
    if "推进器装配" not in c.GetDisplayName():
        continue
    m = cache.GetLocalToWorldTransform(c)
    tr = bbox_cache.ComputeWorldBound(c).ComputeAlignedRange()
    cen = (tr.GetMin() + tr.GetMax()) / 2.0
    size = tr.GetMax() - tr.GetMin()
    axes = {}
    for k, name in enumerate("XYZ"):
        v = Gf.Vec3d(m[0][k], m[1][k], m[2][k]).GetNormalized()
        axes[name] = [round(v[0], 5), round(v[1], 5), round(v[2], 5)]
    # local Y is the thrust axis (cylinder axis: matches 101 mm motor length)
    thrust_axis = axes["Y"]
    rel = (cen - com) / 1000.0
    thrusters.append({
        "prim": c.GetName(),
        "center_mm": [round(cen[0], 2), round(cen[1], 2), round(cen[2], 2)],
        "center_rel_com_m": [round(rel[0], 5), round(rel[1], 5), round(rel[2], 5)],
        "bbox_size_mm": [round(size[0], 2), round(size[1], 2), round(size[2], 2)],
        "world_local_axes": axes,
        "thrust_axis_world": thrust_axis,
        "matrix_world": [[round(m[r][k], 6) for k in range(4)] for r in range(4)],
    })

data = {
    "source_usd": usd_path,
    "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage),
    "upAxis": UsdGeom.GetStageUpAxis(stage),
    "mesh_count": nmesh,
    "triangle_count": ntri,
    "bbox_min_mm": [round(gmn[i], 2) for i in range(3)],
    "bbox_max_mm": [round(gmx[i], 2) for i in range(3)],
    "bbox_size_mm": [round(gmx[i] - gmn[i], 2) for i in range(3)],
    "signed_volume_mm3": round(total_vol, 1),
    "signed_volume_L": round(total_vol / 1e6, 4),
    "com_mm": [round(com[i], 2) for i in range(3)],
    "com_m": [round(com[i] / 1000.0, 5) for i in range(3)],
    "thrusters": thrusters,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(json.dumps({k: v for k, v in data.items() if k != "thrusters"}, ensure_ascii=False, indent=2))
print("\nthrusters:")
for t in thrusters:
    print(f"  {t['prim']:34s} rel=({t['center_rel_com_m']}) axis={t['thrust_axis_world']}")
print("\nwrote:", out_path)
