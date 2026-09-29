#!/bin/bash
# Headless play for Direct-v0 belief SAC. Not training.
#
#   ./scripts/eval_auv_tracking_sb3_sac_v0.sh --checkpoint PATH --episodes 24 --headless

set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/play_auv_tracking_sb3_sac_v0.py \
  "$@"
