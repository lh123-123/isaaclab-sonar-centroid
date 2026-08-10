#!/usr/bin/env bash
# GUI: AUV compensatory PPO play + 6D obs delta logging.
set -euo pipefail
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"
./scripts/kill_isaac_lab_processes.sh 2>/dev/null || true
sleep 2

CKPT="${CKPT:-logs/skrl/sonar_centroid_track/2026-07-13_17-53-29_ppo_torch_auv_residual_ppo_sine_beamrecalib_gtreward/checkpoints/best_agent.pt}"

./isaaclab.sh -p scripts/eval_centroid_track_ppo.py \
  --task Isaac-SonarCentroidTrack-AUV-Residual-Eval-Direct-v0 \
  --checkpoint "${CKPT}" \
  --episodes 1 \
  --sonar_ui \
  --obs_debug_interval "${OBS_DEBUG_INTERVAL:-20}" \
  --step_sleep "${STEP_SLEEP:-0.02}" \
  "$@"
