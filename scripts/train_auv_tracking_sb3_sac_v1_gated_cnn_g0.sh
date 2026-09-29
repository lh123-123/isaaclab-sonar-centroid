#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# G0 ablation: same gated Critic as G1, but rew_scale_speed=0 (no speed-match reward).
# Stops at 2M transitions. Actor=compact CNN. action_scale unchanged.
#
#   ./scripts/train_auv_tracking_sb3_sac_v1_gated_cnn_g0.sh --num_envs 8 --headless

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-AUV-Tracking-Direct-v1 \
  --agent sb3_gated_cnn_cfg_entry_point \
  --algorithm SAC \
  --log_interval 20000 \
  --max_iterations 2000000 \
  --enable_cameras \
  env.rew_scale_speed=0.0 \
  env.rew_speed_sigma=0.5 \
  "$@"
