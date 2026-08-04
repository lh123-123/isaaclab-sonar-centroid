#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# AUV tracking — Stable-Baselines3 SAC (Holo sac_v0_fast aligned)
#
#   ./scripts/train_auv_tracking_sb3_sac.sh --num_envs 64 --headless
#
# SB3 n_timesteps counts transitions across envs (default cfg: 2e6 ≈ Holo).

set -euo pipefail
cd "$(dirname "$0")/.."

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-AUV-Tracking-Direct-v0 \
  --agent sb3_cfg_entry_point \
  --algorithm SAC \
  --log_interval 20000 \
  "$@"
