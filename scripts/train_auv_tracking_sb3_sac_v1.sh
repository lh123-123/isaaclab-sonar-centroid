#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# AUV tracking vision SAC (Holo v1-style): Actor=sonar image, Critic=belief state
#
#   ./scripts/train_auv_tracking_sb3_sac_v1.sh --num_envs 16 --headless
#
# Task: Isaac-AUV-Tracking-Direct-v1 (not Direct-v0 MLP)

set -euo pipefail
cd "$(dirname "$0")/.."

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-AUV-Tracking-Direct-v1 \
  --agent sb3_cfg_entry_point \
  --algorithm SAC \
  --log_interval 20000 \
  --enable_cameras \
  "$@"
