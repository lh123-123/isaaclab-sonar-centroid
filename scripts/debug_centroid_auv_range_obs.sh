#!/usr/bin/env bash
# Verify peak-beam sonar range vs GT (image row shown as baseline only).
set -euo pipefail
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"
./scripts/kill_isaac_lab_processes.sh 2>/dev/null || true
sleep 2

./isaaclab.sh -p scripts/debug_centroid_auv_range_obs.py \
  --task Isaac-SonarCentroidTrack-AUV-Residual-Eval-Direct-v0 \
  --range_accept_m 0.5 \
  --print_every 70 \
  --headless \
  "$@"
