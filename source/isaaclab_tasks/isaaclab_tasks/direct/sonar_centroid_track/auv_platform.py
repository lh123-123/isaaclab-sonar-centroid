# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""WarpAUV platform constants for sonar centroid track (Phase A0+)."""

from __future__ import annotations

import importlib

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

from isaaclab_tasks.direct.nonar2.oceansim_sonar_sensor_cfg import OceanSimSonarSensorCfg
from isaaclab_tasks.direct.sonar_flow_hover.shared_platform import (
    BEACON_CFG,
    OCEANSIM_SONAR_CFG,
    SONAR_ENV_SPACING_M,
    TARGET_SPHERE_RADIUS,
)

_warpauv_assets = importlib.import_module("isaaclab_tasks.direct.isaac-auv-env.assets.warpauv")

# WarpAUV convention: world Z at ``starting_depth`` (see isaac-auv-env README / warpauv_env).
AUV_STARTING_DEPTH_M: float = 8.0

# FLS on ``warpauv_base_link`` (see mount_oceansim_sonar_on_robots); offset in body frame.
AUV_SONAR_MOUNT_OFFSET_LOCAL: tuple[float, float, float] = (0.35, 0.0, 0.05)
# Draw housing on the hull (local offset from SonarMount back toward COM).
AUV_SONAR_HOUSING_OFFSET_LOCAL: tuple[float, float, float] = (-0.27, 0.0, 0.0)
AUV_SONAR_MOUNT_VISUAL: bool = True

AUV_SIM_CFG = SimulationCfg(
    dt=1.0 / 120.0,
    render_interval=2,
    gravity=(0.0, 0.0, -9.81),
)

AUV_SCENE_CFG = InteractiveSceneCfg(
    num_envs=1,
    env_spacing=SONAR_ENV_SPACING_M,
    replicate_physics=True,
)

AUV_VIEWER_CFG = ViewerCfg(
    eye=(5.0, -5.0, AUV_STARTING_DEPTH_M + 3.0),
    lookat=(3.0, 0.0, AUV_STARTING_DEPTH_M),
    origin_type="env",
    env_index=0,
)

WARPAUV_ROBOT_CFG: RigidObjectCfg = _warpauv_assets.WARPAUV_CFG.replace(
    prim_path="/World/envs/env_.*/Robot",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, AUV_STARTING_DEPTH_M),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

_NO_COLLISION = sim_utils.CollisionPropertiesCfg(collision_enabled=False)

AUV_BEACON_CFG: RigidObjectCfg = BEACON_CFG.replace(
    spawn=BEACON_CFG.spawn.replace(collision_props=_NO_COLLISION),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(3.0, 0.0, AUV_STARTING_DEPTH_M),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

__all__ = [
    "AUV_BEACON_CFG",
    "AUV_SCENE_CFG",
    "AUV_SIM_CFG",
    "AUV_SONAR_MOUNT_OFFSET_LOCAL",
    "AUV_SONAR_HOUSING_OFFSET_LOCAL",
    "AUV_SONAR_MOUNT_VISUAL",
    "AUV_STARTING_DEPTH_M",
    "AUV_VIEWER_CFG",
    "OCEANSIM_SONAR_CFG",
    "TARGET_SPHERE_RADIUS",
    "WARPAUV_ROBOT_CFG",
]
