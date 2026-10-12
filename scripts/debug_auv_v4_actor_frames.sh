#!/bin/bash
# Save the 4 RGB frames in the v4 actor observation.
#
#   ./scripts/debug_auv_v4_actor_frames.sh --steps 8

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/debug_auv_v4_actor_frames.py "$@"
