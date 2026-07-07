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


def mount_oceansim_sonar_on_robots(
    oceansim_sonar: OceanSimSonarSensor,
    *,
    num_envs: int,
    sonar_mount_offset_local: tuple[float, float, float],
    sim,
) -> None:
    """Attach OceanSim FLS to ``Robot/SonarMount`` (same USD setup as Nonar2)."""
    for env_idx in range(num_envs):
        robot_prim_path = f"/World/envs/env_{env_idx}/Robot"
        sonar_mount_path = f"{robot_prim_path}/SonarMount"
        try:
            from pxr import UsdGeom
            import omni.usd as usd_utils

            stage = usd_utils.get_context().get_stage()
            xform = UsdGeom.Xform.Define(stage, sonar_mount_path)
            ox, oy, oz = sonar_mount_offset_local
            xformable = UsdGeom.Xformable(xform.GetPrim())
            translate_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            translate_op.Set((float(ox), float(oy), float(oz)))
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
