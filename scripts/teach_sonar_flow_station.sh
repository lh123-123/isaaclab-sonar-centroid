#!/bin/bash
# Teach: capture sonar + flow reference at the correct station pose.
#
# Usage:
#   ./scripts/teach_sonar_flow_station.sh
#   ./scripts/teach_sonar_flow_station.sh --output logs/station_ref/my_ref.npz
#   ./scripts/teach_sonar_flow_station.sh --headless --settle_steps 30
#   ./scripts/teach_sonar_flow_station.sh --offset_dx 0.1 --offset_dy 0

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser
export TERM="${TERM:-xterm-256color}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/ensure_single_isaac_process.sh
source "${SCRIPT_DIR}/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process || exit 1

./isaaclab.sh -p scripts/teach_sonar_flow_station.py \
    --task Isaac-SonarIbvsDrl-Debug-Direct-v0 \
    --num_envs 1 \
    --enable_cameras \
    --sonar_ui \
    "$@"
