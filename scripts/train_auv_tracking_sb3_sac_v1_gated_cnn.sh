#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Direct-v1 SAC: Actor=sonar CNN. Critic=belief backbone + gated image residual.
# Weak speed-match reward (c_v=0.06); action_scale stays 1.5 / 0.5.
#
#   ./scripts/train_auv_tracking_sb3_sac_v1_gated_cnn.sh --num_envs 8 --headless

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
  --enable_cameras \
  env.rew_scale_speed=0.06 \
  env.rew_speed_sigma=0.5 \
  "$@"
