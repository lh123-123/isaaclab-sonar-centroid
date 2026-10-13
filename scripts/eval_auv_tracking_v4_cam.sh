#!/usr/bin/env bash
# Eval Direct-v4-cam SAC checkpoint (tracking geometry + return).
#
#   ./scripts/eval_auv_tracking_v4_cam.sh --checkpoint PATH --episodes 20 --headless
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
# Detached launches (setsid nohup) inherit TERM=dumb; tput then fails nonzero and
# `set -euo pipefail` aborts this script before isaaclab.sh prints anything.
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/eval_auv_tracking_v4_cam.py \
  --enable_cameras \
  "$@"
