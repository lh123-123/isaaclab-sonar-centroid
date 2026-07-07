#!/bin/bash
# Eval template homing (requires taught snapshot from teach_sonar_flow_station.sh).
#
# Usage:
#   ./scripts/teach_sonar_flow_station.sh
#   ./scripts/eval_sonar_flow_template.sh --headless --max_steps 200

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/ensure_single_isaac_process.sh
source "${SCRIPT_DIR}/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process || exit 1

./isaaclab.sh -p scripts/eval_sonar_flow_hover.py \
    --task Isaac-SonarFlowHover-Template-Eval-Direct-v0 \
    --num_envs 1 \
    --max_steps 200 \
    --headless \
    --enable_cameras \
    "$@"
