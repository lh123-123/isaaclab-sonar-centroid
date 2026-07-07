# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared dimensions and thresholds for sonar centroid tracking."""

from __future__ import annotations

# Polar sonar (OceanSim FLS native binning via nonar2 cfg)
POLAR_HEIGHT = 110
POLAR_WIDTH = 61

# Discrete PPO
NUM_DISCRETE_ACTIONS = 3
ACTION_LEFT = 0
ACTION_STAY = 1
ACTION_RIGHT = 2

POLICY_OBS_DIM = 7
POLICY_OBS_DIM_IMAGE = 4
POLICY_ACTION_DIM = 1

# Perception / tracking
DEFAULT_THRESHOLD_RATIO = 0.35
DEFAULT_FB_ERROR_THRESHOLD = 1.0
DEFAULT_MAX_TRACK = 15
MIN_VALID_FEATURES = 3
MIN_VALID_RATIO = 0.25
FLOW_RETEACH_MIN_VALID = 3

# Phase 1 validation (absolute coords, pixels)
PERC_SONAR_COL_MEAN_MAX = 2.5
PERC_SONAR_COL_P90_MAX = 3.0
PERC_SONAR_ROW_MEAN_MAX = 2.0
PERC_SONAR_DIST_MEAN_MAX = 3.0

# Phase 2+ tracking (fixed-point frame, pixels)
TRACK_ORIGIN_DIST_MAX = 3.0

# Control
DEFAULT_MAX_LATERAL_VEL = 0.30
DEFAULT_LATERAL_AMP_M = 1.5

# Reward weights
REWARD_W_U = 1.0
REWARD_W_V = 0.3
REWARD_W_ACTION_SWITCH = 0.02
REWARD_W_SAFETY = 10.0
