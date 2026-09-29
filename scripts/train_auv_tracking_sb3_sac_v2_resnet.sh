#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# AUV tracking flow SAC (Direct-v2) — ResNet50 image encoder, high-res sonar.
# Same as train_auv_tracking_sb3_sac_v2_hires.sh but with ResNet50 instead of
# compact CNN. All SAC / env hyperparams unchanged.
#
#   ./scripts/train_auv_tracking_sb3_sac_v2_resnet.sh --num_envs 8 --headless

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
  --task Isaac-AUV-Tracking-Direct-v2 \
  --agent sb3_resnet_cfg_entry_point \
  --algorithm SAC \
  --log_interval 20000 \
  --enable_cameras \
  env.oceansim_sonar_cfg.range_res=0.025 \
  env.oceansim_sonar_cfg.angular_res=1.0 \
  env.oceansim_sonar_cfg.height=140 \
  env.oceansim_sonar_cfg.width=121 \
  "$@"
