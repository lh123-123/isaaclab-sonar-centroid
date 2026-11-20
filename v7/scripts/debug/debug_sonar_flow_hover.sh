#!/bin/bash
# Interactive GUI debug for SonarFlowHover (OceanSim sonar viewport + flow logs).
#
# Usage:
#   ./v7/scripts/debug/debug_sonar_flow_hover.sh
#   ./v7/scripts/debug/debug_sonar_flow_hover.sh --no-freeze_robot
#   ./v7/scripts/debug/debug_sonar_flow_hover.sh --auto_sweep --sweep_interval_s 4
#   ./v7/scripts/debug/debug_sonar_flow_hover.sh --offset_dx 0.25 --offset_dy 0
#   ./v7/scripts/debug/debug_sonar_flow_hover.sh --save_sonar

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser
export TERM="${TERM:-xterm-256color}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
# shellcheck source=lib/ensure_single_isaac_process.sh
source "${REPO_ROOT}/scripts/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process || exit 1

./isaaclab.sh -p v7/scripts/debug/debug_sonar_flow_hover.py \
    --task Isaac-SonarIbvsDrl-Debug-Direct-v0 \
    --num_envs 1 \
    --enable_cameras \
    --sonar_ui \
    "$@"
