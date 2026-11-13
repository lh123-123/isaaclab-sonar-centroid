#!/bin/bash
# Leave-one-out for the v7 hover cues. One Isaac process at a time.
# The full system is the 2026-11-13 run; these four start from scratch.
#
#   ./scripts/train_auv_v7_ablation_queue.sh
#
# Each run is 200k steps (the full run had already plateaued by ~116k).
# Compare logged disp_end_m, not the episode reward: removing a term
# lowers the reward ceiling.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p /home/lh/auv_v7_ablate
QUEUE_LOG=/home/lh/auv_v7_ablate/queue.log

run() {
  local name="$1"
  shift
  echo "=== ${name} start $(date -Is) ===" | tee -a "${QUEUE_LOG}"
  set +e
  ./scripts/train_auv_tracking_sb3_sac_v7_flow_hover_gsrov.sh \
    --num_envs 8 --headless --max_iterations 200000 \
    "$@" > "/home/lh/auv_v7_ablate/${name}.log" 2>&1
  local code=$?
  set -e
  echo "=== ${name} exit ${code} $(date -Is) ===" | tee -a "${QUEUE_LOG}"
  if [[ "${code}" -ne 0 ]]; then
    exit "${code}"
  fi
}

run no_visodom env.ablate_visodom=true
run no_template env.ablate_template=true
run no_flow env.ablate_flow=true
run no_drift env.ablate_drift=true
echo "=== queue done $(date -Is) ===" | tee -a "${QUEUE_LOG}"
