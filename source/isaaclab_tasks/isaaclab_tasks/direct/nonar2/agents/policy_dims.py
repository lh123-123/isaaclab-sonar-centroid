# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation dimensions for Nonar2 policy modes."""

STRUCTURED_OBS_DIM = 10
BELIEF_STRUCTURED_DIM = 6
SONAR_FEATURE_DIM = 256
BELIEF_SAC_OBS_DIM = BELIEF_STRUCTURED_DIM + SONAR_FEATURE_DIM
# Module B strongest-beam index (SAC Belief6 / debug).
BELIEF6_BEAM_OBS_DIM = 1
# Subsampled raw sonar column intensities: 61 beams / stride 3 → 21 bins.
COLUMN_PROFILE_OBS_DIM = 21
# SAC Belief6 task (Isaac-Nonar2-Belief6-Direct-v0).
BELIEF6_OBS_DIM = BELIEF6_BEAM_OBS_DIM
