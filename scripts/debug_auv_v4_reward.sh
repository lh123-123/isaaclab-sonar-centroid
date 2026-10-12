#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Reward smoke test for Direct-v4-cam (numbers, not spaces). See debug_auv_v4_reward.py.
#
# Runs a scripted hold / chase / random rollout and checks that the reward decomposition
# recomputed in the probe matches what step() returns, that nothing is NaN/inf, and that the
# legacy clamp(max=1.0) shaping can still be reproduced via rew_penalty_clip=1.0.
#
#   ./scripts/debug_auv_v4_reward.sh --num_envs 4 --headless

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/debug_auv_v4_reward.py "$@"
