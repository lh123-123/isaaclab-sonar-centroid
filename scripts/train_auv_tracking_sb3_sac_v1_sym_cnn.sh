#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Direct-v1 SAC, compact CNN, Actor AND Critic eat sonar (not asymmetric).
# Calm water is the env default (ocean_current.enabled=False).
#
#   ./scripts/train_auv_tracking_sb3_sac_v1_sym_cnn.sh --num_envs 8 --headless

set -euo pipefail
cd "$(dirname "$0")/.."

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-AUV-Tracking-Direct-v1 \
  --agent sb3_symmetric_cnn_cfg_entry_point \
  --algorithm SAC \
  --log_interval 20000 \
  --enable_cameras \
  "$@"
