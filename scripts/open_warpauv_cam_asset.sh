#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Open the editable *copy* of the WarpAUV asset (data/warpauv_cam/warpauv.usd) in the Isaac Sim GUI
# so prims can be added by hand.
#
#   ./scripts/open_warpauv_cam_asset.sh              # GUI
#   ./scripts/open_warpauv_cam_asset.sh --headless   # print the prim tree, then exit
#
# The copy is byte-identical to data/warpauv/warpauv.usd and self-contained, so edits saved into it
# do not affect v0/v1/v2. File > Save (Ctrl+S) writes into the asset layer.

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/open_warpauv_cam_asset.py "$@"
