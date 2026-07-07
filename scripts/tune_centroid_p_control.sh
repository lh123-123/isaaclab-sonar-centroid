#!/usr/bin/env bash
# One Isaac session, sweep P gains, print |d_col| table.
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/kill_isaac_lab_processes.sh 2>/dev/null || true
sleep 3
TERM=xterm ./isaaclab.sh -p scripts/eval_centroid_track_p_control.py --headless --tune \
  --target_vy "${VY:-0.28}" \
  --output_dir "${OUT_ROOT:-logs/centroid_track/phase2_tune}"
