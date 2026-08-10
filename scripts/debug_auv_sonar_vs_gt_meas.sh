#!/usr/bin/env bash
# GT (r,α) vs OceanSim beam-peak sonar measurement on WarpAUV (debug only).
#
#   ./scripts/debug_auv_sonar_vs_gt_meas.sh
#   MOVING=1 ./scripts/debug_auv_sonar_vs_gt_meas.sh
#   MOVING=1 BELIEF_GATE=1 ./scripts/debug_auv_sonar_vs_gt_meas.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"

# shellcheck source=lib/ensure_single_isaac_process.sh
source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

STEPS="${STEPS:-}"
EXTRA_ARGS=()
if [[ "${MOVING:-0}" == "1" ]]; then
  if [[ -z "${STEPS}" ]]; then STEPS=240; fi
  EXTRA_ARGS+=(--moving --log_every 10)
else
  if [[ -z "${STEPS}" ]]; then STEPS=60; fi
fi
if [[ "${BELIEF_GATE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--belief_gate)
  if [[ -z "${STEPS}" || "${STEPS}" == "60" ]] && [[ "${MOVING:-0}" == "1" ]]; then
    STEPS=240
  fi
fi
# Optional: DETECTOR=beam|centroid (belief_gate defaults to centroid in the py script)
if [[ -n "${DETECTOR:-}" ]]; then
  EXTRA_ARGS+=(--detector "${DETECTOR}")
fi

HEADLESS_ARGS=(--headless)
if [[ "${GUI:-0}" == "1" ]]; then
  HEADLESS_ARGS=()
fi

./isaaclab.sh -p scripts/debug_auv_sonar_vs_gt_meas.py \
  --task Isaac-SonarCentroidTrack-AUV-Debug-Direct-v0 \
  --num_envs 1 \
  --steps "${STEPS}" \
  "${HEADLESS_ARGS[@]}" \
  --enable_cameras \
  "${EXTRA_ARGS[@]}" \
  "$@"
