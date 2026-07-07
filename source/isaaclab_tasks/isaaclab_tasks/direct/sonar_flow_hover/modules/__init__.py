# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Modular IBVS-DRL building blocks (debug each layer independently)."""

from .actions import apply_target_motion, apply_velocity_action, hold_robot_at_hover, hold_robot_at_station
from .constants import (
    BODY_STATE_DIM,
    LOST_FEATURE_TERMINATE_STEPS,
    MAX_TRACK_POINTS,
    MIN_VALID_FEATURES,
    POLICY_ACTION_DIM,
    POLICY_OBS_DIM,
)
from .debug import collect_metrics, decode_observation, maybe_print_ibvs_debug
from .geometry import (
    beacon_bearing_range,
    beacon_image_rois,
    hover_pos_error_norm,
    robot_lateral_sway_abs,
    robot_lateral_sway_signed,
    station_pos_error_norm,
    sonar_pos_w,
    spawn_yaw,
    sway_tracking_error_abs,
    target_lateral_offset_dy,
    yaw_toward_beacon,
)
from .observation import ObservationBuilder
from .reference_manager import ReferenceManager
from .reset_logic import reset_beacon_and_hover, reset_beacon_and_station, reset_robot_episode, sample_spawn_offset
from .rewards import compute_reward
from .sonar_pipeline import SonarPerceptionPipeline
from .terminations import TerminationChecker

__all__ = [
    "BODY_STATE_DIM",
    "LOST_FEATURE_TERMINATE_STEPS",
    "MAX_TRACK_POINTS",
    "MIN_VALID_FEATURES",
    "POLICY_ACTION_DIM",
    "POLICY_OBS_DIM",
    "ObservationBuilder",
    "ReferenceManager",
    "SonarPerceptionPipeline",
    "TerminationChecker",
    "apply_target_motion",
    "apply_velocity_action",
    "beacon_bearing_range",
    "beacon_image_rois",
    "collect_metrics",
    "compute_reward",
    "decode_observation",
    "hold_robot_at_station",
    "hold_robot_at_hover",
    "hover_pos_error_norm",
    "maybe_print_ibvs_debug",
    "reset_beacon_and_station",
    "reset_beacon_and_hover",
    "reset_robot_episode",
    "robot_lateral_sway_abs",
    "robot_lateral_sway_signed",
    "sample_spawn_offset",
    "sonar_pos_w",
    "spawn_yaw",
    "station_pos_error_norm",
    "sway_tracking_error_abs",
    "target_lateral_offset_dy",
    "yaw_toward_beacon",
]
