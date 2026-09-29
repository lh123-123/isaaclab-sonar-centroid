#!/bin/bash
# Dump Direct-v2 network image as fan PPI. Optional SAC checkpoint.
#
#   ./scripts/debug_auv_tracking_flow_v2.sh --headless --steps 16 --save_every 2
#   ./scripts/debug_auv_tracking_flow_v2.sh --headless --checkpoint logs/sb3/.../model_500000_steps.zip

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/debug_auv_tracking_flow_v2.py --enable_cameras "$@"
