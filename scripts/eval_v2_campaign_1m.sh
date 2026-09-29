#!/bin/bash
# After v2-full seed 123 hits 1M: stop the train queue and eval finished 1M models.
# Protocol: 8 env × 24 episodes, headless, hires 140×121 (play_v2 hardcodes this).
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT123="logs/sb3/Isaac-AUV-Tracking-Direct-v2/2026-09-15_10-40-20/model_1000000_steps.zip"
OUT="${OUT:-/tmp/paper_eval}"
LOG="${LOG:-/tmp/v2_eval_batch.log}"
mkdir -p "${OUT}"

log() { echo "[eval-batch] $*" | tee -a "${LOG}"; }

log "waiting for seed-123 1M at ${CKPT123} ($(date))"
while [[ ! -f "${CKPT123}" ]]; do
  sleep 20
done
log "saw 1M zip $(date); waiting 25s for flush then stopping Isaac"
sleep 25

# Do not resume the paused train queue.
kill -KILL 636945 2>/dev/null || true
./scripts/kill_isaac_lab_processes.sh >> "${LOG}" 2>&1 || true
sleep 8

eval_one() {
  local task="$1" tag="$2" ckpt="$3"
  if [[ ! -f "${ckpt}" ]]; then
    log "SKIP ${tag}: missing ${ckpt}"
    return 0
  fi
  log "START ${tag}  $(date)"
  ./scripts/eval_auv_tracking_flow_v2.sh \
    --task "${task}" \
    --checkpoint "${ckpt}" \
    --num_envs 8 \
    --episodes 24 \
    --headless \
    --tag "${tag}" \
    --summary "${OUT}/${tag}.json" \
    --dump "${OUT}/${tag}.npz"
  log "DONE ${tag} exit=$?  $(date)"
  sleep 5
}

# This-campaign 1M, same hires protocol. Prefer explicit 1M zip (not 1.2M overshoot).
eval_one Isaac-AUV-Tracking-Direct-v2 full-s42 \
  logs/sb3/Isaac-AUV-Tracking-Direct-v2/2026-09-14_12-28-05/model_1000000_steps.zip
eval_one Isaac-AUV-Tracking-Direct-v2 full-s7 \
  logs/sb3/Isaac-AUV-Tracking-Direct-v2/2026-09-14_20-54-11/model_1000000_steps.zip
eval_one Isaac-AUV-Tracking-Direct-v2 full-s123 \
  "${CKPT123}"
eval_one Isaac-AUV-Tracking-Direct-v2-image image-s7 \
  logs/sb3/Isaac-AUV-Tracking-Direct-v2-image/2026-09-15_03-56-49/model_1000000_steps.zip

python3 scripts/align_auv_eval_metrics.py \
  "${OUT}"/full-s42.json \
  "${OUT}"/full-s7.json \
  "${OUT}"/full-s123.json \
  "${OUT}"/image-s7.json \
  --out "${OUT}/aligned.json" \
  | tee -a "${LOG}"

log "=== EVAL_BATCH_DONE $(date) ==="
echo EVAL_BATCH_DONE >> "${LOG}"
