#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Runtime contract check for the Direct-v5-flow GT optical-flow stack observation:
# reset stack is all-zero, the stack shifts by exactly one 2-channel frame per step, the newest
# frame matches the env's live `_flow_image`, values stay inside [-1, 1], and the stamped target
# disk is present and bounded.
#
#   ./scripts/debug_auv_v5_flow_obs.sh --headless --num_envs 4

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/debug_auv_v5_flow_obs.py "$@"
