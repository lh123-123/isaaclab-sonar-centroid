#!/usr/bin/env bash
# Phase A0: WarpAUV + OceanSim sonar smoke test (headless).
set -euo pipefail
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"
./scripts/kill_isaac_lab_processes.sh 2>/dev/null || true
sleep 2

STEPS="${STEPS:-120}"
./isaaclab.sh -p scripts/debug_centroid_track_auv_sonar.py \
  --task Isaac-SonarCentroidTrack-AUV-Debug-Direct-v0 \
  --steps "${STEPS}" \
  --headless
