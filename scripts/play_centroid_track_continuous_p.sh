#!/usr/bin/env bash
# Continuous image-P demo on sine target (Box action env, sonar UI).
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/kill_isaac_lab_processes.sh 2>/dev/null || true
sleep 2

KP="${KP:-1.0}"
KFF="${KFF:-1.0}"
STEP_SLEEP="${STEP_SLEEP:-0.02}"

TERM=xterm ./isaaclab.sh -p scripts/play_centroid_track_continuous_p.py \
  --task Isaac-SonarCentroidTrack-Continuous-Eval-Direct-v0 \
  --enable_cameras --sonar_ui \
  --kp "${KP}" --kff "${KFF}" --step_sleep "${STEP_SLEEP}" \
  "$@"
