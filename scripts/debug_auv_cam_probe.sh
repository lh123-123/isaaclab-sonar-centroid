#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Probe which pose the RTX renderer uses for the Direct-v3-cam camera / target.
#
#   ./scripts/debug_auv_cam_probe.sh --headless

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/debug_auv_cam_probe.py "$@"
