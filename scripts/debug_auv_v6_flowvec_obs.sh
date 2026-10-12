#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Runtime contract check for the Direct-v6-flowvec flow-VECTOR observation: reset stack is
# all-zero, the stack shifts by exactly one 5-vector per step, the newest slots match the env's
# live `_flow_vec_now`, everything stays inside the declared Box bounds, the vector agrees with the
# raster it is a picture of, and the documented scale/saturation caveats still hold.
#
#   ./scripts/debug_auv_v6_flowvec_obs.sh --headless --num_envs 4

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/debug_auv_v6_flowvec_obs.py "$@"
