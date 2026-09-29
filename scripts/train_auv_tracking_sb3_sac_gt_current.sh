#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Direct-v0 belief SAC + geometric GT KF (fast, no OceanSim) + ocean current
# aligned with Holo v0 (`v0_config_fast.yml`: type=random, max_velocity=2.0,
# spatial_scale=5.0, seed=42, clip 5 m/s).
#
#   ./scripts/train_auv_tracking_sb3_sac_gt_current.sh --num_envs 64 --headless
#
# Hydra overrides still work, e.g. env.ocean_current.strength=0.5

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
  env.ocean_current.strength=2.0 \
  env.ocean_current.clip_speed=5.0 \
  env.ocean_current.affect_target=false \
  env.ocean_current.params.spatial_scale=5.0 \
  env.ocean_current.params.seed=42.0 \
  "$@"
