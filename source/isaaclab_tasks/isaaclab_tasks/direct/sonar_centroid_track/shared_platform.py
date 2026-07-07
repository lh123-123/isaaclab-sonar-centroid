# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Re-export validated Nonar2 / OceanSim platform from sonar_flow_hover."""

from isaaclab_tasks.direct.sonar_flow_hover.shared_platform import (
    BEACON_CFG,
    OCEANSIM_SONAR_CFG,
    OceanSimSonarSensor,
    ROBOT_CFG,
    SCENE_CFG,
    SIM_CFG,
    SONAR_ENV_SPACING_M,
    SONAR_MOUNT_OFFSET_LOCAL,
    TARGET_SPHERE_RADIUS,
    VIEWER_CFG,
    WORKSPACE_HEIGHT,
    mount_oceansim_sonar_on_robots,
)

__all__ = [
    "BEACON_CFG",
    "OCEANSIM_SONAR_CFG",
    "OceanSimSonarSensor",
    "ROBOT_CFG",
    "SCENE_CFG",
    "SIM_CFG",
    "SONAR_ENV_SPACING_M",
    "SONAR_MOUNT_OFFSET_LOCAL",
    "TARGET_SPHERE_RADIUS",
    "VIEWER_CFG",
    "WORKSPACE_HEIGHT",
    "mount_oceansim_sonar_on_robots",
]
