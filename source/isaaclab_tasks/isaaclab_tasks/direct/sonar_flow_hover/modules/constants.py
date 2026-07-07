# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared dimensions and defaults for Sonar IBVS-DRL."""

from __future__ import annotations

MAX_TRACK_POINTS = 20
MAX_EXTRACT_FEATURES = 30
MIN_VALID_FEATURES = 2
BODY_STATE_DIM = 8
POLICY_ACTION_DIM = 4
POLICY_OBS_DIM = MAX_TRACK_POINTS * 2 + MAX_TRACK_POINTS + BODY_STATE_DIM + POLICY_ACTION_DIM

# 1D lateral train: sway-only action
LATERAL_1D_ACTION_DIM = 1
# Legacy full stack (sparse 20-point layout + body + prev action)
LATERAL_1D_OBS_DIM = MAX_TRACK_POINTS * 2 + MAX_TRACK_POINTS + BODY_STATE_DIM + LATERAL_1D_ACTION_DIM
# Compact 1D: teach-track centroid offset (px) + valid + prev sway
LATERAL_1D_COMPACT_OBS_DIM = 4
LATERAL_1D_OBS_NAMES = (
    "centroid_daz_px",
    "centroid_dr_px",
    "valid_ratio",
    "prev_sway",
)

# Observation layout (for slicing during debug)
OBS_FEAT_ERR_SLICE = slice(0, MAX_TRACK_POINTS * 2)
OBS_VALID_SLICE = slice(MAX_TRACK_POINTS * 2, MAX_TRACK_POINTS * 2 + MAX_TRACK_POINTS)
OBS_BODY_SLICE = slice(MAX_TRACK_POINTS * 3, MAX_TRACK_POINTS * 3 + BODY_STATE_DIM)
OBS_ACTION_SLICE = slice(MAX_TRACK_POINTS * 3 + BODY_STATE_DIM, POLICY_OBS_DIM)

LOST_FEATURE_TERMINATE_STEPS = 40
