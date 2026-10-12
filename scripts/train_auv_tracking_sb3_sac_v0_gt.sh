#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# AUV tracking Direct-v0 — belief MLP (symmetric actor/critic), KF measurement from
# geometric GT (range r + bearing α). NO OceanSim sonar, NO camera → no rendering.
#
#   ./scripts/train_auv_tracking_sb3_sac_v0_gt.sh --num_envs 64 --headless
#
# Why a separate script: scripts/train_auv_tracking_sb3_sac.sh hardcodes --enable_cameras
# (correct for the stock sonar v0). For the GT variant, train.py's own logic decides:
#   task contains AUV-Tracking, not Direct-v1/v2, no "-cam", and kf_measurement == "gt"
#   → _needs_cameras stays False → enable_cameras is NOT forced on.
# So we deliberately do NOT pass --enable_cameras here.
#
# Policy/config unchanged from Holo sac_v0_fast (agents/sb3_sac_cfg.yaml):
#   n_timesteps=3e6, buffer_size=20000, lr 1e-4, pi [256,256,256] / qf [256,256], auto alpha.
# Ocean current stays OFF (calm water), same as stock v0.

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
  "$@"
