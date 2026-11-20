"""Prepare the GS-ROV CAD USD for RL training.

The raw STEP->USD export cannot be dropped into an RL env as-is:

1. Its centering ``xformOp:translate`` was authored in *metres* while the stage is
   ``metersPerUnit = 0.001`` (1 unit == 1 mm), so the hull sits ~0.96 m above the prim
   origin -- the physics frame and the visuals disagree by ~1 m.
2. ``physics:mass`` is 0 and ``physics:diagonalInertia`` is (0, 0, 0), so the rigid body
   has no usable mass properties.
3. It carries no collider, and its indexed ``normals`` buffer no longer matches the point
   count (Hydra warns "corrupted data in primvar 'normal'").

Rather than patch the referenced/instanced CAD stage in place, this tool rebuilds a flat,
self-contained asset: every mesh is baked into the body frame (centred on the COM), mass /
inertia / COM are authored on the rigid body, a box collider is added, and the meshes can
optionally be decimated. Appearance is flattened to per-part constant colours, which is
fine because the vehicle is never inside its own camera view.

Usage:
    python prepare_gsrov_usd.py                     # 8.1k tris, full detail
    python prepare_gsrov_usd.py --cluster-mm 4      # ~3.6k tris
    python prepare_gsrov_usd.py --quadric 0.6 --cluster-mm 3   # ~1.9k tris
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.normpath(os.path.join(_HERE, "..", "models"))
_DEFAULT_SRC = os.path.join(_MODELS, "GS-ROV_drl_centered.usd")
_DEFAULT_DST = os.path.join(_MODELS, "GS-ROV_drl_train.usd")
_DEFAULT_DATA = os.path.join(_MODELS, "GS-ROV_drl_data.json")

_FALLBACK_COLOR = (0.62, 0.64, 0.66)


# --------------------------------------------------------------------------------------
# mesh reading / decimation
# --------------------------------------------------------------------------------------
def _triangles(mesh: UsdGeom.Mesh) -> tuple[np.ndarray, np.ndarray] | None:
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    points = mesh.GetPointsAttr().Get()
    if not counts or not indices or not points:
        return None
    if all(c == 3 for c in counts):
        return np.asarray(points, dtype=np.float64), np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    tris: list[tuple[int, int, int]] = []
    cursor = 0
    for c in counts:
        base = indices[cursor]
        for k in range(1, c - 1):
            tris.append((base, indices[cursor + k], indices[cursor + k + 1]))
        cursor += c
    return np.asarray(points, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def _quadric(points: np.ndarray, faces: np.ndarray, reduction: float) -> tuple[np.ndarray, np.ndarray]:
    if reduction <= 0.0 or len(faces) < 8:
        return points, faces
    try:
        import fast_simplification as fs
        pts, fcs = fs.simplify(points, faces.astype(np.int32), target_reduction=float(reduction))
    except Exception:
        return points, faces
    if len(fcs) >= len(faces):
        return points, faces
    return np.asarray(pts, dtype=np.float64), np.asarray(fcs, dtype=np.int64)


def _cluster(points: np.ndarray, faces: np.ndarray, grid: float) -> tuple[np.ndarray, np.ndarray]:
    if grid <= 0.0 or len(faces) < 8:
        return points, faces
    import trimesh

    snapped = np.round(points / grid) * grid
    tm = trimesh.Trimesh(vertices=snapped, faces=faces, process=False)
    tm.merge_vertices()
    tm.update_faces(tm.unique_faces())
    tm.update_faces(tm.nondegenerate_faces())
    tm.remove_unreferenced_vertices()
    if len(tm.faces) < 4:
        return points, faces
    return np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int64)


def _part_color(prim: Usd.Prim) -> tuple[float, float, float]:
    """Per-part colour: ``primvars:displayColor`` first, then the bound material, then grey."""
    try:
        display = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
        if display and display.HasValue():
            value = display.Get()
            if value is not None and len(value) > 0:
                first = value[0]
                return (float(first[0]), float(first[1]), float(first[2]))
    except Exception:
        pass
    try:
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not bound:
            return _FALLBACK_COLOR
        for sub in Usd.PrimRange(bound.GetPrim()):
            if sub.GetTypeName() != "Shader":
                continue
            shader = UsdShade.Shader(sub)
            if shader.GetShaderIdAttr().Get() != "UsdPreviewSurface":
                continue
            value = shader.GetInput("diffuseColor").Get()
            if isinstance(value, Gf.Vec3f):
                return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        pass
    return _FALLBACK_COLOR


# --------------------------------------------------------------------------------------
# stage helpers
# --------------------------------------------------------------------------------------
def _apply_mass(root: Usd.Prim, mass: float, inertia: tuple[float, float, float]) -> None:
    UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
    api = UsdPhysics.MassAPI.Apply(root)
    api.CreateMassAttr().Set(float(mass))
    api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(v) for v in inertia]))
    api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _make_material(stage: Usd.Stage, path: str, color: tuple[float, float, float]) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf_value_type("Color3f")).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf_value_type("Float")).Set(0.55)
    shader.CreateInput("metallic", Sdf_value_type("Float")).Set(0.25)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def Sdf_value_type(name: str):
    from pxr import Sdf

    return getattr(Sdf.ValueTypeNames, name)


def _add_box_collider(stage: Usd.Stage, root: Usd.Prim, size_mm: tuple[float, float, float]) -> None:
    path = root.GetPath().AppendChild("Collider")
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size_mm]))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdGeom.Imageable(cube.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the GS-ROV CAD USD as a training-ready asset.")
    parser.add_argument("--src", default=_DEFAULT_SRC)
    parser.add_argument("--dst", default=_DEFAULT_DST)
    parser.add_argument("--data", default=_DEFAULT_DATA)
    parser.add_argument("--mass", type=float, default=13.15)
    parser.add_argument("--inertia", default="0.31,0.46,0.57", help="Ixx,Iyy,Izz in kg m^2")
    parser.add_argument("--quadric", type=float, default=0.0, help="Quadric reduction in [0,1); 0 = off")
    parser.add_argument("--cluster-mm", type=float, default=0.0, help="Vertex-cluster lattice in mm; 0 = off")
    parser.add_argument("--no-collider", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        raise SystemExit(f"source USD not found: {args.src}")
    inertia = tuple(float(v) for v in args.inertia.split(","))
    if len(inertia) != 3:
        raise SystemExit("--inertia needs three comma separated values")

    source = Usd.Stage.Open(args.src)
    src_root = source.GetDefaultPrim()
    if src_root is None:
        raise SystemExit("stage has no default prim")
    print(f"[prep] src        : {args.src}")
    print(f"[prep] defaultPrim: {src_root.GetPath()}  metersPerUnit={UsdGeom.GetStageMetersPerUnit(source)}")

    com_mm = Gf.Vec3d(0.0, 0.0, 0.0)
    if args.data and os.path.isfile(args.data):
        with open(args.data) as handle:
            com_m = json.load(handle).get("com_m")
        if com_m:
            com_mm = Gf.Vec3d(*[float(v) * 1000.0 for v in com_m])
    print(f"[prep] com        : ({com_mm[0]:.2f}, {com_mm[1]:.2f}, {com_mm[2]:.2f}) mm")

    # ---- gather geometry into the body frame ----------------------------------------
    cache = UsdGeom.XformCache()
    parts: list[tuple[str, np.ndarray, np.ndarray, tuple[float, float, float]]] = []
    before_v = before_t = after_v = after_t = 0
    for prim in Usd.PrimRange(src_root):
        if prim.GetTypeName() != "Mesh":
            continue
        tri = _triangles(UsdGeom.Mesh(prim))
        if tri is None:
            continue
        points, faces = tri
        matrix = cache.GetLocalToWorldTransform(prim)
        world = np.array([list(matrix.Transform(Gf.Vec3d(*p))) for p in points], dtype=np.float64)
        world -= np.array([com_mm[0], com_mm[1], com_mm[2]])
        before_v += len(world)
        before_t += len(faces)
        color = _part_color(prim)
        if args.quadric > 0.0:
            world, faces = _quadric(world, faces, args.quadric)
        if args.cluster_mm > 0.0:
            world, faces = _cluster(world, faces, args.cluster_mm)
        after_v += len(world)
        after_t += len(faces)
        parts.append((prim.GetName(), world, faces, color))

    if not parts:
        raise SystemExit("no triangulable meshes found in source")
    print(f"[prep] meshes     : {len(parts)}")
    print(f"[prep] verts      : {before_v} -> {after_v}")
    print(f"[prep] triangles  : {before_t} -> {after_t}")

    all_points = np.concatenate([p[1] for p in parts], axis=0)
    mn, mx = all_points.min(axis=0), all_points.max(axis=0)
    size = mx - mn
    center = (mn + mx) / 2.0
    print(f"[prep] bbox size  : ({size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f}) mm")
    print(f"[prep] bbox center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) mm")

    # ---- build the flat asset --------------------------------------------------------
    if os.path.isfile(args.dst):
        os.remove(args.dst)
    stage = Usd.Stage.CreateNew(args.dst)
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/GSROV")
    stage.SetDefaultPrim(root.GetPrim())
    _apply_mass(root.GetPrim(), args.mass, inertia)

    UsdGeom.Xform.Define(stage, "/GSROV/geometry")
    material_cache: dict[tuple[float, float, float], UsdShade.Material] = {}
    for index, (name, points, faces, color) in enumerate(parts):
        key = tuple(round(c, 4) for c in color)
        if key not in material_cache:
            material_cache[key] = _make_material(stage, f"/GSROV/Looks/Part{len(material_cache)}", color)
        mesh = UsdGeom.Mesh.Define(stage, f"/GSROV/geometry/{name}_{index}")
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).astype(np.int32).tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        lo, hi = points.min(axis=0), points.max(axis=0)
        mesh.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material_cache[key])

    if not args.no_collider:
        _add_box_collider(stage, root.GetPrim(), (size[0], size[1], size[2]))

    stage.GetRootLayer().Save()
    print(f"[prep] mass       : {args.mass} kg")
    print(f"[prep] inertia    : {inertia} kg m^2")
    print(f"[prep] collider   : {'none' if args.no_collider else 'box'}")
    print(f"[prep] written    : {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
