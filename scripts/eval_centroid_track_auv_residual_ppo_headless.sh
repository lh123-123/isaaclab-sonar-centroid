#!/usr/bin/env bash
# Headless: PPO residual metrics — GT distance + compensation vs PD baseline.
set -euo pipefail
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"
./scripts/kill_isaac_lab_processes.sh 2>/dev/null || true
sleep 2

# Deploy baseline: 2D residual + surge scale 0.50 (|GT−hold| ≈0.30m, ±0.5m ≈77%).
CKPT="${CKPT:-logs/skrl/sonar_centroid_track/2026-07-13_17-53-29_ppo_torch_auv_residual_ppo_sine_beamrecalib_gtreward/checkpoints/best_agent.pt}"
PRINT_EVERY="${PRINT_EVERY:-50}"

./isaaclab.sh -p scripts/eval_centroid_track_auv_residual_ppo_metrics.py \
  --headless --enable_cameras \
  --checkpoint "${CKPT}" \
  --baseline_pd \
  --print_every "${PRINT_EVERY}" \
  "$@"
