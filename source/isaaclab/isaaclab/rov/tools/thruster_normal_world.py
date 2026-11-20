"""Per-thruster world face-normal covariance -> cylinder/thrust axis."""

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


def normals_world(prim):
    normals = []
    areas = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() != "Mesh":
            continue
        xf = cache.GetLocalToWorldTransform(p)
        mesh = UsdGeom.Mesh(p)
        points = mesh.GetPointsAttr().Get()
        faces = mesh.GetFaceVertexIndicesAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        offset = 0
        for count in counts:
            ids = faces[offset:offset + count]
            offset += count
            if count < 3:
                continue
            polygon = [tuple(xf.Transform(Gf.Vec3d(points[index]))) for index in ids]
            normal = np.zeros(3)
            for k, point in enumerate(polygon):
                normal += np.cross(point, polygon[(k + 1) % len(polygon)])
            length = np.linalg.norm(normal)
            if length < 1e-9:
                continue
            normals.append(normal / length)
            areas.append(length / 2)
    return np.asarray(normals), np.asarray(areas)


for i, thruster in enumerate(thrusters, 1):
    normals, areas = normals_world(thruster)
    covariance = (normals * areas[:, None]).T @ normals / areas.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, 0]
    if axis[0] < 0:
        axis = -axis
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(thruster).ComputeAlignedRange()
    center = (bounds.GetMin() + bounds.GetMax()) / 2
    print(f"#{i} {thruster.GetName():34s} center=({center[0]:7.2f},{center[1]:7.2f},{center[2]:7.2f}) "
          f"faces={len(normals):6d} eig={np.round(eigenvalues,4)} axis={np.round(axis,3)}")
