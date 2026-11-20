"""Find thruster axis from face-normal covariance (cylindrical surfaces)."""

import sys
import numpy as np
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

thr = None
for c in wrapper.GetChildren():
    if "推进器装配" in c.GetDisplayName():
        thr = c
        break

proto = thr.GetPrototype()
meshes = []


def rec(p):
    if p.GetTypeName() == "Mesh":
        meshes.append(p)
    for c in p.GetChildren():
        rec(c)


rec(proto)

normals = []
areas = []
for mp in meshes:
    m = UsdGeom.Mesh(mp)
    pts = np.array(m.GetPointsAttr().Get())
    nrm = m.GetNormalsAttr().Get()
    faces = m.GetFaceVertexIndicesAttr().Get()
    counts = m.GetFaceVertexCountsAttr().Get()
    interp = m.GetNormalsInterpolation()
    if nrm:
        nrm = np.array(nrm)
    idx = 0
    for c in counts:
        ids = faces[idx:idx + c]
        idx += c
        if c < 3:
            continue
        A, B, C = pts[ids[0]], pts[ids[1]], pts[ids[2]]
        n = np.cross(B - A, C - A)
        a = np.linalg.norm(n)
        if a < 1e-12:
            continue
        # choose a normal: prefer authored per-face/vertex normal
        if nrm is not None and interp in ("faceVarying", "uniform", "vertex"):
            try:
                if interp == "faceVarying":
                    fn = nrm[ids].mean(axis=0)
                elif interp == "uniform":
                    fn = nrm[0]
                else:
                    fn = nrm[ids[0]]
                n = fn
                a = np.linalg.norm(n)
                if a < 1e-12:
                    continue
            except Exception:
                pass
        normals.append(n / a)
        areas.append(a / 2.0)

N = np.array(normals)
W = np.array(areas)
cov = (N * W[:, None]).T @ N / W.sum()
w, v = np.linalg.eigh(cov)
print("points-normal count:", len(N))
print("normal covariance eigenvalues:", np.round(w, 4))
print("axis (smallest normal-variance):", np.round(v[:, 0], 3))
print("other axes:", np.round(v[:, 1], 3), np.round(v[:, 2], 3))
