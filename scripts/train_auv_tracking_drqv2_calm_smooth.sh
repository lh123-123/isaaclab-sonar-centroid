#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# DrQ-v2 Direct-v1: calm water + action smoothness + target cold-start.
#   ./scripts/train_auv_tracking_drqv2_calm_smooth.sh --num_envs 8 --headless

set -euo pipefail
cd "$(dirname "$0")/.."

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./scripts/train_auv_tracking_drqv2.sh \
  --rew_scale_smooth 1.0 \
  "$@"
