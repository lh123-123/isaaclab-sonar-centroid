#!/usr/bin/env bash
# Phase A2: AUV residual PPO under sim→real A+B base:
#   A) beam hold_meas surge  B) station depth/IMU obs noise
# Defaults: 2 envs, 800 iters (~51k steps).
set -euo pipefail
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"
./scripts/kill_isaac_lab_processes.sh 2>/dev/null || true
sleep 2

NUM_ENVS="${NUM_ENVS:-2}"
MAX_ITERS="${MAX_ITERS:-800}"
SEED="${SEED:-42}"

echo "[train] AUV residual PPO A+B  num_envs=${NUM_ENVS}  max_iters=${MAX_ITERS}  seed=${SEED}"

./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-SonarCentroidTrack-AUV-Residual-PPO-Direct-v0 \
  --headless --enable_cameras \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  --seed "${SEED}" \
  "$@"
