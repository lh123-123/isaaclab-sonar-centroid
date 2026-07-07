#!/bin/bash
# Interactive GUI debug for SonarFlowHover (OceanSim sonar viewport + flow logs).
#
# Usage:
#   ./scripts/debug_sonar_flow_hover.sh
#   ./scripts/debug_sonar_flow_hover.sh --no-freeze_robot
#   ./scripts/debug_sonar_flow_hover.sh --auto_sweep --sweep_interval_s 4
#   ./scripts/debug_sonar_flow_hover.sh --offset_dx 0.25 --offset_dy 0
#   ./scripts/debug_sonar_flow_hover.sh --save_sonar

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser
export TERM="${TERM:-xterm-256color}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/ensure_single_isaac_process.sh
source "${SCRIPT_DIR}/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process || exit 1

./isaaclab.sh -p scripts/debug_sonar_flow_hover.py \
    --task Isaac-SonarIbvsDrl-Debug-Direct-v0 \
    --num_envs 1 \
    --enable_cameras \
    --sonar_ui \
    "$@"
