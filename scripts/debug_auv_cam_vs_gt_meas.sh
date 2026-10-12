#!/usr/bin/env bash
# GT (r,α) vs camera luma-centroid under morning v4-cam policy (debug only).
#
#   ./scripts/debug_auv_cam_vs_gt_meas.sh
#   BELIEF_GATE=1 ./scripts/debug_auv_cam_vs_gt_meas.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck source=lib/ensure_single_isaac_process.sh
source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
export PYTHONUNBUFFERED=1

CKPT="${CKPT:-logs/sb3/Isaac-AUV-Tracking-Direct-v4-cam/2026-09-21_10-36-23/model_650000_steps.zip}"
STEPS="${STEPS:-400}"
EXTRA_ARGS=()
if [[ "${BELIEF_GATE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--belief_gate)
fi
if [[ "${S1:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--s1)
fi
# Default open-loop (sonar-style). Set USE_POLICY=1 to drive with morning checkpoint.
if [[ "${USE_POLICY:-0}" != "1" ]]; then
  EXTRA_ARGS+=(--no_policy)
fi
if [[ -n "${VIZ_DIR:-}" ]]; then
  EXTRA_ARGS+=(--viz_dir "${VIZ_DIR}")
fi

HEADLESS_ARGS=(--headless)
if [[ "${GUI:-0}" == "1" ]]; then
  HEADLESS_ARGS=()
fi

./isaaclab.sh -p scripts/debug_auv_cam_vs_gt_meas.py \
  --task Isaac-AUV-Tracking-Direct-v4-cam \
  --num_envs 1 \
  --steps "${STEPS}" \
  --checkpoint "${CKPT}" \
  --enable_cameras \
  "${HEADLESS_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
