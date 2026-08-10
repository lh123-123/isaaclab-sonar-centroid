# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared cuboid robot + OceanSim FLS platform (from ``nonar2``).

``nonar2`` and ``lh_2`` share the same architecture; we anchor on ``nonar2`` configs
so sonar_ibvs_drl stays in sync with the tracking stack the team already validated.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from isaaclab_tasks.direct.nonar2.nonar2_env_cfg import Nonar2EnvCfg
from isaaclab_tasks.direct.nonar2.oceansim_sonar_sensor import OceanSimSonarSensor
from isaaclab_tasks.direct.nonar2.oceansim_sonar_sensor_cfg import OceanSimSonarSensorCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObjectCfg

_NONAR2 = Nonar2EnvCfg()

# IBVS lateral tracking uses velocity / kinematic control only — no contact physics needed.
_NO_COLLISION = sim_utils.CollisionPropertiesCfg(collision_enabled=False)

# --- Robot / sonar (Nonar2 geometry, collisions off) ---
ROBOT_CFG: RigidObjectCfg = _NONAR2.robot_cfg.replace(
    spawn=_NONAR2.robot_cfg.spawn.replace(collision_props=_NO_COLLISION),
)
SONAR_MOUNT_OFFSET_LOCAL: tuple[float, float, float] = _NONAR2.sonar_mount_offset_local
OCEANSIM_SONAR_CFG: OceanSimSonarSensorCfg = _NONAR2.oceansim_sonar_cfg
WORKSPACE_HEIGHT: float = float(_NONAR2.workspace_height)
SIM_CFG = _NONAR2.sim
VIEWER_CFG = _NONAR2.viewer
# Adjacent env origins must be farther apart than sonar max range so FLS returns
# from neighboring beacons/robots do not leak into each env's observations.
_SONAR_MAX_RANGE_M = float(OCEANSIM_SONAR_CFG.max_range)
SONAR_ENV_SPACING_M = max(12.0, _SONAR_MAX_RANGE_M * 2.0)
SCENE_CFG = _NONAR2.scene.replace(env_spacing=SONAR_ENV_SPACING_M)

# Target: sphere (distinct color), single object for sonar returns.
_nonar2_target_spawn = _NONAR2.target_cfg.spawn
TARGET_SPHERE_RADIUS: float = 0.5
BEACON_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Target",
    spawn=sim_utils.SphereCfg(
        radius=TARGET_SPHERE_RADIUS,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.38, 0.12)),
        physics_material=_nonar2_target_spawn.physics_material,
        rigid_props=_nonar2_target_spawn.rigid_props,
        mass_props=_nonar2_target_spawn.mass_props,
        collision_props=_NO_COLLISION,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(3.0, 0.0, WORKSPACE_HEIGHT),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)
# Legacy alias; second target disabled by default (use_second_beacon=False).
BEACON2_CFG: RigidObjectCfg = BEACON_CFG.replace(prim_path="/World/envs/env_.*/Target2")


def _resolve_rigid_body_prim_path(robot_prim_path: str, stage) -> str:
    """Return the PhysX rigid-body prim (same as ``RigidObject`` root), not the spawn wrapper."""
    from pxr import UsdPhysics

    from isaaclab.sim.utils import get_all_matching_child_prims

    root_prims = get_all_matching_child_prims(
        robot_prim_path,
        predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
    )
    if len(root_prims) == 1:
        return root_prims[0].GetPath().pathString
    if len(root_prims) == 0:
        return robot_prim_path
    # Match RigidObject: first unique body (WarpAUV should only have one).
    return root_prims[0].GetPath().pathString


def _remove_legacy_sonar_mount(stage, robot_prim_path: str, sonar_mount_path: str) -> None:
    """Drop stale ``Robot/SonarMount`` when the sensor is parented on an inner rigid link."""
    legacy_path = f"{robot_prim_path}/SonarMount"
    if legacy_path != sonar_mount_path and stage.GetPrimAtPath(legacy_path).IsValid():
        stage.RemovePrim(legacy_path)


def _set_sonar_mount_xform(
    stage,
    sonar_mount_path: str,
    *,
    offset_local: tuple[float, float, float],
    rpy_deg: tuple[float, float, float] | None = None,
) -> None:
    """Create/update ``Robot/SonarMount`` as a child xform of the robot rigid body."""
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform.Define(stage, sonar_mount_path)
    xformable = UsdGeom.Xformable(xform)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    ox, oy, oz = offset_local
    translate_op.Set((float(ox), float(oy), float(oz)))
    if rpy_deg is not None:
        rotate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rotate_op = op
                break
        if rotate_op is None:
            rotate_op = xformable.AddRotateXYZOp()
        rx, ry, rz = rpy_deg
        rotate_op.Set(Gf.Vec3f(float(rx), float(ry), float(rz)))


def _add_sonar_housing_visual(
    stage,
    sonar_mount_path: str,
    *,
    hull_offset_local: tuple[float, float, float],
) -> None:
    """Blue housing on the hull (offset from SonarMount) — does not block the sensor."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    housing_path = f"{sonar_mount_path}/SonarHousing"
    if stage.GetPrimAtPath(housing_path).IsValid():
        stage.RemovePrim(housing_path)
    housing = UsdGeom.Xform.Define(stage, housing_path)
    hx, hy, hz = hull_offset_local
    UsdGeom.XformCommonAPI(housing).SetTranslate(Gf.Vec3d(float(hx), float(hy), float(hz)))
    cube = UsdGeom.Cube.Define(stage, f"{housing_path}/Mesh")
    UsdGeom.XformCommonAPI(cube).SetScale(Gf.Vec3f(0.14, 0.10, 0.08))
    color = Gf.Vec3f(0.15, 0.55, 0.85)
    mat_path = f"{housing_path}/Material"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)


def mount_oceansim_sonar_on_robots(
    oceansim_sonar: OceanSimSonarSensor,
    *,
    num_envs: int,
    sonar_mount_offset_local: tuple[float, float, float],
    sim,
    sonar_mount_rpy_deg: tuple[float, float, float] | None = None,
    sonar_mount_visual: bool = False,
    sonar_housing_offset_local: tuple[float, float, float] | None = None,
) -> None:
    """Attach OceanSim FLS to ``<rigid_body>/SonarMount`` (body frame, follows PhysX rotation)."""
    for env_idx in range(num_envs):
        robot_prim_path = f"/World/envs/env_{env_idx}/Robot"
        sonar_mount_path = f"{robot_prim_path}/SonarMount"
        try:
            import omni.usd as usd_utils

            stage = usd_utils.get_context().get_stage()
            body_path = _resolve_rigid_body_prim_path(robot_prim_path, stage)
            sonar_mount_path = f"{body_path}/SonarMount"
            if env_idx == 0:
                print(
                    f"[SonarMount] env={env_idx} rigid_body={body_path} mount={sonar_mount_path} "
                    f"offset={sonar_mount_offset_local}",
                    flush=True,
                )
            _remove_legacy_sonar_mount(stage, robot_prim_path, sonar_mount_path)
            _set_sonar_mount_xform(
                stage,
                sonar_mount_path,
                offset_local=sonar_mount_offset_local,
                rpy_deg=sonar_mount_rpy_deg,
            )
            sonar_mount_prim = stage.GetPrimAtPath(sonar_mount_path)
            if sonar_mount_prim:
                for child in sonar_mount_prim.GetChildren():
                    if str(child.GetName()).startswith("SonarSensor"):
                        stage.RemovePrim(str(child.GetPath()))
            if sonar_mount_visual:
                hull_off = sonar_housing_offset_local or (0.0, 0.0, 0.0)
                _add_sonar_housing_visual(stage, sonar_mount_path, hull_offset_local=hull_off)
        except Exception as exc:
            print(f"[SonarIbvsDrl] sonar mount fallback to robot root: {exc}")
            sonar_mount_path = robot_prim_path

        oceansim_sonar.initialize(env_idx, sonar_mount_path)

    for warmup_idx in range(4):
        sim.step()
        if warmup_idx == 0:
            time.sleep(0.02)


__all__ = [
    "BEACON_CFG",
    "BEACON2_CFG",
    "OCEANSIM_SONAR_CFG",
    "OceanSimSonarSensor",
    "ROBOT_CFG",
    "SCENE_CFG",
    "SONAR_ENV_SPACING_M",
    "SIM_CFG",
    "SONAR_MOUNT_OFFSET_LOCAL",
    "TARGET_SPHERE_RADIUS",
    "VIEWER_CFG",
    "WORKSPACE_HEIGHT",
    "mount_oceansim_sonar_on_robots",
]
