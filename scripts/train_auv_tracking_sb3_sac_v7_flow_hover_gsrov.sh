#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Direct-v7-flow-hover-gsrov SAC (GS-ROV 8-thruster + cuboid proxy).
# For remote SSH training (activates conda env automatically).
#
#   ./scripts/train_auv_tracking_sb3_sac_v7_flow_hover_gsrov.sh --num_envs 8 --headless

set -euo pipefail
cd "$(dirname "$0")/.."

# Activate conda isaaclab env if not already active
if [[ -z "${CONDA_PREFIX:-}" || ! -f "${CONDA_PREFIX}/bin/python" ]]; then
    if [[ -f ~/miniconda3/bin/conda ]]; then
        eval "$(~/miniconda3/bin/conda shell.bash hook 2>/dev/null)" || true
        conda activate isaaclab
    fi
fi

source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-AUV-Tracking-Direct-v7-flow-hover \
  --agent sb3_cfg_entry_point \
  --algorithm SAC \
  --log_interval 20000 \
  --flow_method farneback \
  "$@"
