"""Analyze a converted CAD USD: bounds, volume, COM, per-part stats.

Usage:
    conda run -n isaaclab python analyze_usd.py MODEL.usd [--parts]
"""

import sys
from pxr import Usd, UsdGeom, Gf

path = sys.argv[1]
show_parts = "--parts" in sys.argv

stage = Usd.Stage.Open(path)
print("stage:", path)
print("defaultPrim:", stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None)
print("metersPerUnit:", UsdGeom.GetStageMetersPerUnit(stage))
print("upAxis:", UsdGeom.GetStageUpAxis(stage))


def mesh_stats(mesh_prim):
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get()
    faces = mesh.GetFaceVertexIndicesAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    if not pts:
        return 0.0, None, None, 0
    mn = Gf.Vec3d(1e30, 1e30, 1e30)
    mx = Gf.Vec3d(-1e30, -1e30, -1e30)
    for p in pts:
        for i in range(3):
            mn[i] = min(mn[i], p[i])
            mx[i] = max(mx[i], p[i])
    vol = 0.0
    if faces and counts:
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
                A = Gf.Vec3d(pts[a]); B = Gf.Vec3d(pts[b]); C = Gf.Vec3d(pts[d])
                vol += Gf.Dot(A, Gf.Cross(B, C)) / 6.0
            idx += c
    return vol, mn, mx, len(pts)


root = stage.GetDefaultPrim()
inst = root
# Unwrap single-child wrapper chain and instances to reach real geometry
for _ in range(3):
    if inst.IsInstance():
        proto = inst.GetPrototype()
        if proto:
            inst = proto
    children = [c for c in inst.GetChildren() if c.GetTypeName() not in ("Scope",)]
    if len(children) == 1 and children[0].GetTypeName() == "Xform":
        inst = children[0]
    else:
        break

print("geometry root:", inst.GetPath())

objects = []
gmn = Gf.Vec3d(1e30, 1e30, 1e30)
gmx = Gf.Vec3d(-1e30, -1e30, -1e30)
total_vol = 0.0
wsum = Gf.Vec3d(0, 0, 0)
nmesh = 0


def collect(prim, prefix=""):
    global gmn, gmx, total_vol, wsum, nmesh
    if prim.GetTypeName() == "Mesh":
        vol, mn, mx, npts = mesh_stats(prim)
        nmesh += 1
        if mn is not None:
            for i in range(3):
                gmn[i] = min(gmn[i], mn[i])
                gmx[i] = max(gmx[i], mx[i])
        if vol:
            center = (mn + mx) / 2.0
            total_vol += vol
            wsum += center * vol
        objects.append((prefix or prim.GetName(), vol, mn, mx, npts))
    for ch in prim.GetChildren():
        collect(ch, prefix)


collect(inst)

size = gmx - gmn
com = wsum / total_vol if total_vol else Gf.Vec3d(0, 0, 0)
print("meshes:", nmesh)
print("bbox min:", gmn)
print("bbox max:", gmx)
print("bbox size:", size)
print("volume mm^3:", total_vol, " L:", total_vol / 1e6)
print("COM:", com)

if show_parts:
    print("\n--- parts (name, vol mm^3, min, max, points) ---")
    for name, vol, mn, mx, npts in objects:
        print(name, f"{vol:.0f}", mn, mx, npts)
