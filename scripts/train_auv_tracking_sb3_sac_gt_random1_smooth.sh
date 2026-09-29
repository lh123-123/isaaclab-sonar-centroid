#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Direct-v0 GT KF + random current 1.0 + action-smooth reward (c_smooth=1.0).
# Holo yaml has c_smooth=1.0 but the term is commented out in world_auv_v0.py;
# this run turns it on: r += exp(-||a - a_prev||).
#
#   ./scripts/train_auv_tracking_sb3_sac_gt_random1_smooth.sh --num_envs 64 --headless

set -euo pipefail
cd "$(dirname "$0")/.."

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-AUV-Tracking-Direct-v0 \
  --agent sb3_cfg_entry_point \
  --algorithm SAC \
  --kf_measurement gt \
  --log_interval 20000 \
  env.ocean_current.enabled=true \
  env.ocean_current.type=random \
  env.ocean_current.strength=1.0 \
  env.ocean_current.clip_speed=5.0 \
  env.ocean_current.affect_target=false \
  env.ocean_current.params.max_velocity=1.0 \
  env.ocean_current.params.spatial_scale=5.0 \
  env.ocean_current.params.seed=42.0 \
  env.rew_scale_smooth=1.0 \
  "$@"
