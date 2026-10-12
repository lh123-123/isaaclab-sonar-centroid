#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Dump Direct-v4-cam's actor observation (12ch RGB stack + 2ch GT flow) as PNGs.
#
#   ./scripts/debug_auv_v4_cam_obs.sh --headless --num_envs 1 --steps 8 --save_last 3

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/debug_auv_v4_cam_obs.py "$@"
