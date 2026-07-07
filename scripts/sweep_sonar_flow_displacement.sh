#!/bin/bash
# Small-displacement sonar optical-flow sweep (headless, logs table).
#
# Usage:
#   ./scripts/sweep_sonar_flow_displacement.sh
#   ./scripts/sweep_sonar_flow_displacement.sh --settle_steps 40

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/ensure_single_isaac_process.sh
source "${SCRIPT_DIR}/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process || exit 1

./isaaclab.sh -p scripts/sweep_sonar_flow_displacement.py \
    --task Isaac-SonarFlowHover-Sweep-Direct-v0 \
    --num_envs 1 \
    --headless \
    --enable_cameras \
    "$@"
