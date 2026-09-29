#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# DrQ-v2 on Direct-v1 (sonar polar + temporal frame_stack=3 by default):
#   ./scripts/train_auv_tracking_drqv2.sh --num_envs 8 --headless
# Smoke:
#   ./scripts/train_auv_tracking_drqv2.sh --num_envs 2 --total_steps 2000 --headless
#
# Obs: take polar channel-0 from Direct-v1's 3x64x64 image, stack K frames → (K,64,64).
# Not RGB×K (the 3 env channels are replicated polar, not color).

set -euo pipefail
cd "$(dirname "$0")/.."

# isaaclab.sh `tabs 4` fails under TERM=dumb
export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/reinforcement_learning/drqv2/train_auv_tracking.py \
  --enable_cameras \
  "$@"
