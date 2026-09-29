#!/bin/bash
# Paper-eval play: N completed episodes, episode mean±std JSON. Not training.
#
#   ./scripts/eval_auv_tracking_vision_v1.sh --checkpoint PATH --episodes 20 --headless

set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/play_auv_tracking_vision_v1.py \
  --rew_scale_smooth 0 \
  "$@"
