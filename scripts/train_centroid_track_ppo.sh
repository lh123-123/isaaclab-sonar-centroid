#!/usr/bin/env bash
# Continuous PPO for sonar centroid tracking (Box vy, image-only obs, triangle target).
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/kill_isaac_lab_processes.sh 2>/dev/null || true
sleep 2

NUM_ENVS="${NUM_ENVS:-8}"
MAX_ITERS="${MAX_ITERS:-2000}"
SEED="${SEED:-42}"

TERM=xterm ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-SonarCentroidTrack-Continuous-PPO-Direct-v0 \
  --headless --enable_cameras \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  --seed "${SEED}"
