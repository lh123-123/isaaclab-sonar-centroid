# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sonar centroid tracking modules (polar-only, fixed-point origin)."""

from .actions import discrete_to_lateral_velocity
from .centroid_bus import CentroidFrame, make_centroid_frame
from .constants import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    NUM_DISCRETE_ACTIONS,
    POLICY_OBS_DIM,
    POLAR_HEIGHT,
    POLAR_WIDTH,
)
from .coordinates import PolarPoint, mirror_azimuth_col, normalize_uv, polar_point, to_fixed_frame
from .debug import decode_observation, format_centroid_frame
from .fixed_point import FixedPointSnapshot, FixedPointStore
from .geometry_perception import GeometryPerception
from .observation import DiscreteObservationBuilder
from .sonar_centroid_tracker import SonarCentroidTracker
from .rewards import compute_reward
from .sonar_gt import SonarGtDetector
from .sonar_pipeline import PolarSonarPipeline

__all__ = [
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_STAY",
    "CentroidFrame",
    "DiscreteObservationBuilder",
    "FixedPointSnapshot",
    "FixedPointStore",
    "GeometryPerception",
    "NUM_DISCRETE_ACTIONS",
    "POLAR_HEIGHT",
    "POLAR_WIDTH",
    "POLICY_OBS_DIM",
    "PolarPoint",
    "SonarCentroidTracker",
    "SonarGtDetector",
    "compute_reward",
    "decode_observation",
    "discrete_to_lateral_velocity",
    "format_centroid_frame",
    "make_centroid_frame",
    "mirror_azimuth_col",
    "normalize_uv",
    "polar_point",
    "PolarSonarPipeline",
    "preprocess_polar_batch",
    "preprocess_polar_single",
    "to_fixed_frame",
]
