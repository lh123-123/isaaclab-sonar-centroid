#!/bin/bash
# SonarFlowHover step-1 smoke eval: sonar detection + flow debug (no trained policy).
#
# Usage:
#   ./scripts/eval_sonar_flow_hover.sh
#   ./scripts/eval_sonar_flow_hover.sh --headless --max_steps 300
#
# GUI + sonar viewport (optional):
#   ./scripts/eval_sonar_flow_hover.sh --enable_cameras --sonar_ui

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/ensure_single_isaac_process.sh
source "${SCRIPT_DIR}/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process || exit 1

./isaaclab.sh -p scripts/eval_sonar_flow_hover.py \
    --task Isaac-SonarFlowHover-Eval-Direct-v0 \
    --num_envs 1 \
    --max_steps 200 \
    --enable_cameras \
    "$@"
