"""Build a training-ready ROV data JSON from the converted GS-ROV_drl USD.

- Thruster centers from actual point AABBs.
- Thruster thrust axes from area-weighted face-normal covariance
  (cylindrical duct/motor surfaces => axis = min normal-variance direction).
- Sign convention: vertical thrusters positive = +Z (up);
  horizontal thrusters positive = along the computed diagonal (x component > 0).
- Emits a 6x8 thrust allocation matrix (TAM): [force; torque] per unit thrust.

Usage:
    conda run -n isaaclab python build_rov_data.py MODEL.usd OUT.json
"""

import json
import sys

import numpy as np
from pxr import Gf, Usd, UsdGeom

usd_path = sys.argv[1]
out_path = sys.argv[2]

stage = Usd.Stage.Open(usd_path)
cache = UsdGeom.XformCache(Usd.TimeCode.Default())

root = stage.GetDefaultPrim()
wrapper = root
while True:
    children = [c for c in wrapper.GetChildren() if c.GetTypeName() == "Xform"]
    if len(children) == 1 and not wrapper.IsInstance():
        wrapper = children[0]
    else:
        break


def mesh_points_world(prim):
    points = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() != "Mesh":
            continue
        xf = cache.GetLocalToWorldTransform(p)
        attr = UsdGeom.Mesh(p).GetPointsAttr().Get()
        if attr:
            points.extend(tuple(xf.Transform(Gf.Vec3d(pt))) for pt in attr)
    return np.asarray(points)


def mesh_volume_centroid(prim):
    volume = 0.0
    moment = Gf.Vec3d(0, 0, 0)
    triangles = 0
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() != "Mesh":
            continue
        xf = cache.GetLocalToWorldTransform(p)
        mesh = UsdGeom.Mesh(p)
        pts = mesh.GetPointsAttr().Get()
        faces = mesh.GetFaceVertexIndicesAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        if not pts or not faces or not counts:
            continue
        offset = 0
        for count in counts:
            ids = faces[offset:offset + count]
            offset += count
            triangles += max(0, count - 2)
            if count == 3:
                tris = [(ids[0], ids[1], ids[2])]
            elif count == 4:
                tris = [(ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])]
            else:
                tris = [(ids[0], ids[i], ids[i + 1]) for i in range(1, count - 1)]
            for a, b, c in tris:
                A = xf.Transform(Gf.Vec3d(pts[a]))
                B = xf.Transform(Gf.Vec3d(pts[b]))
                C = xf.Transform(Gf.Vec3d(pts[c]))
                tv = Gf.Dot(A, Gf.Cross(B, C)) / 6.0
                volume += tv
                moment += ((A + B + C) / 4.0) * tv
    centroid = moment / volume if abs(volume) > 1e-9 else Gf.Vec3d(0, 0, 0)
    return volume, centroid, triangles


def normal_covariance_axis(prim):
    normals = []
    areas = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() != "Mesh":
            continue
        xf = cache.GetLocalToWorldTransform(p)
        mesh = UsdGeom.Mesh(p)
        pts = mesh.GetPointsAttr().Get()
        faces = mesh.GetFaceVertexIndicesAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        offset = 0
        for count in counts:
            ids = faces[offset:offset + count]
            offset += count
            if count < 3:
                continue
            poly = [tuple(xf.Transform(Gf.Vec3d(pts[i]))) for i in ids]
            n = np.zeros(3)
            for k, point in enumerate(poly):
                n += np.cross(point, poly[(k + 1) % len(poly)])
            length = np.linalg.norm(n)
            if length < 1e-9:
                continue
            normals.append(n / length)
            areas.append(length / 2)
    normals = np.asarray(normals)
    areas = np.asarray(areas)
    cov = (normals * areas[:, None]).T @ normals / areas.sum()
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, 0], eigvals


volume, com, triangles = mesh_volume_centroid(root)
all_points = mesh_points_world(root)
bmin, bmax = all_points.min(0), all_points.max(0)

thrusters = []
for c in wrapper.GetChildren():
    if "推进器装配" not in c.GetDisplayName():
        continue
    pts = mesh_points_world(c)
    center = (pts.min(0) + pts.max(0)) / 2.0
    axis, eigvals = normal_covariance_axis(c)
    if abs(axis[2]) > 0.9:
        kind = "vertical"
        if axis[2] < 0:
            axis = -axis
    else:
        kind = "horizontal"
        if axis[0] < 0:
            axis = -axis
    axis = axis / np.linalg.norm(axis)
    rel = (center - np.array([com[0], com[1], com[2]])) / 1000.0
    thrusters.append({
        "prim": c.GetName(),
        "kind": kind,
        "center_mm": [round(float(v), 2) for v in center],
        "center_rel_com_m": [round(float(v), 5) for v in rel],
        "axis_world": [round(float(v), 4) for v in axis],
        "normal_cov_eig": [round(float(v), 4) for v in eigvals],
    })

# --- thrust allocation matrix ---
tam = np.zeros((6, len(thrusters)))
for i, t in enumerate(thrusters):
    u = np.array(t["axis_world"])
    r = np.array(t["center_rel_com_m"])
    tam[0:3, i] = u
    tam[3:6, i] = np.cross(r, u)

data = {
    "source_usd": usd_path,
    "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage),
    "upAxis": UsdGeom.GetStageUpAxis(stage),
    "triangle_count": triangles,
    "bbox_min_m": [round(float(v) / 1000.0, 5) for v in bmin],
    "bbox_max_m": [round(float(v) / 1000.0, 5) for v in bmax],
    "bbox_size_m": [round(float(v - w) / 1000.0, 5) for v, w in zip(bmax, bmin)],
    "signed_volume_L": round(volume / 1e6, 4),
    "com_m": [round(float(com[0]) / 1000.0, 5),
              round(float(com[1]) / 1000.0, 5),
              round(float(com[2]) / 1000.0, 5)],
    "thrusters": thrusters,
    "tam_rows": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
    "tam": [[round(float(tam[r, c]), 5) for c in range(len(thrusters))] for r in range(6)],
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("bbox (m):", data["bbox_min_m"], "->", data["bbox_max_m"])
print("size (m):", data["bbox_size_m"])
print("volume (L):", data["signed_volume_L"], "com (m):", data["com_m"])
print("\n#   kind        center_rel_com_m                     axis")
for i, t in enumerate(thrusters, 1):
    print(f"{i:2d}  {t['kind']:10s}  {str(t['center_rel_com_m']):34s}  {t['axis_world']}")
print("\nTAM (columns = thrusters 1..8):")
print(np.round(tam, 4))
print("\nwrote:", out_path)
