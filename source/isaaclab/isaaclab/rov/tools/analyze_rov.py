"""Analyze a converted CAD USD (handles instancing).

Computes world bbox, total signed volume, COM, and per-thruster data.

Usage:
    conda run -n isaaclab python analyze_rov.py MODEL.usd
"""

import sys
from pxr import Usd, UsdGeom, Gf, UsdUtils

path = sys.argv[1]
stage = Usd.Stage.Open(path)

print("stage:", path)
print("defaultPrim:", stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None)
print("metersPerUnit:", UsdGeom.GetStageMetersPerUnit(stage))
print("upAxis:", UsdGeom.GetStageUpAxis(stage))

xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())


def mesh_volume_centroid(mesh_prim, xf):
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get()
    faces = mesh.GetFaceVertexIndicesAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    if not pts or not faces or not counts:
        return 0.0, Gf.Vec3d(0, 0, 0), None, None
    mn = Gf.Vec3d(1e30, 1e30, 1e30)
    mx = Gf.Vec3d(-1e30, -1e30, -1e30)
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
            A = xf.Transform(Gf.Vec3d(pts[a]))
            B = xf.Transform(Gf.Vec3d(pts[b]))
            C = xf.Transform(Gf.Vec3d(pts[d]))
            tv = Gf.Dot(A, Gf.Cross(B, C)) / 6.0
            vol += tv
            csum += ((A + B + C) / 4.0) * tv
        idx += c
    for pt in pts:
        q = xf.Transform(Gf.Vec3d(pt))
        for i in range(3):
            mn[i] = min(mn[i], q[i])
            mx[i] = max(mx[i], q[i])
    centroid = csum / vol if abs(vol) > 1e-9 else (mn + mx) / 2.0
    return vol, centroid, mn, mx


# world bbox via BBoxCache over the whole model
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
root = stage.GetDefaultPrim()
rng = bbox_cache.ComputeWorldBound(root).ComputeAlignedRange()
gmn, gmx = rng.GetMin(), rng.GetMax()
print("bbox min:", tuple(round(v, 2) for v in gmn))
print("bbox max:", tuple(round(v, 2) for v in gmx))
print("bbox size:", tuple(round(v, 2) for v in (gmx - gmn)))

total_vol = 0.0
wsum = Gf.Vec3d(0, 0, 0)
nmesh = 0
ntri = 0
for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
    if prim.GetTypeName() != "Mesh":
        continue
    xf = xform_cache.GetLocalToWorldTransform(prim)
    vol, cen, mn, mx = mesh_volume_centroid(prim, xf)
    counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
    nmesh += 1
    if counts:
        ntri += sum(max(0, c - 2) for c in counts)
    total_vol += vol
    wsum += cen * vol

com = wsum / total_vol if abs(total_vol) > 1e-9 else Gf.Vec3d(0, 0, 0)
print("meshes:", nmesh, "triangles:", ntri)
print("signed volume mm^3:", round(total_vol, 1), " L:", round(total_vol / 1e6, 4))
print("COM mm:", tuple(round(v, 2) for v in com))
print("COM m:", tuple(round(v / 1000.0, 5) for v in com))
