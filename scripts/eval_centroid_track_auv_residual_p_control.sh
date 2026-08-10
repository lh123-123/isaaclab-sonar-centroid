#!/usr/bin/env bash
# Sanity: position_lead base + rule-based image residual (verify Δ scale before PPO).
set -euo pipefail
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"
./scripts/kill_isaac_lab_processes.sh 2>/dev/null || true
sleep 2

./isaaclab.sh -p scripts/eval_centroid_track_auv_residual_p_control.py \
  --task Isaac-SonarCentroidTrack-AUV-Residual-Eval-Direct-v0 \
  --headless \
  "$@"
