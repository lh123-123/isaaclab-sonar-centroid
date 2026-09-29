#!/bin/bash
# Sequential Direct-v2 1M queue. One Isaac at a time.
# Seeds: 7, 42, 123, 2024 (seed 66 deferred). After the in-flight job, v2-full
# seed 123 runs next; remaining jobs follow. Skip if that seed already has ≥1M steps.
#
#   ./scripts/run_v2_five_seed_queue.sh
#
# Does NOT evaluate. Eval all seeds together after the queue finishes.

set -uo pipefail
cd "$(dirname "$0")/.."

QUEUE_LOG="${QUEUE_LOG:-/tmp/v2_five_seed_queue.log}"
export V2_MAX_STEPS="${V2_MAX_STEPS:-1000000}"
export TERM="${TERM:-xterm-256color}"
export PYTHONUNBUFFERED=1

HIRES=(
  env.oceansim_sonar_cfg.range_res=0.025
  env.oceansim_sonar_cfg.angular_res=1.0
  env.oceansim_sonar_cfg.height=140
  env.oceansim_sonar_cfg.width=121
)

SEEDS=(7 42 123 2024)
TASKS=(
  Isaac-AUV-Tracking-Direct-v2
  Isaac-AUV-Tracking-Direct-v2-image
  Isaac-AUV-Tracking-Direct-v2-state
)

MANIFEST="${MANIFEST:-logs/sb3/v2_seed_model_manifest.tsv}"
CKPT_HELPER="./scripts/lib/auv_seed_ckpt.py"

log() { echo "[queue] $*" | tee -a "${QUEUE_LOG}"; }

record_map() {
  local task="$1" seed="$2" ckpt="$3"
  mkdir -p "$(dirname "${MANIFEST}")"
  if [[ ! -f "${MANIFEST}" ]]; then
    printf "time\ttask\tseed\tcheckpoint\n" > "${MANIFEST}"
  fi
  printf "%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "${task}" "${seed}" "${ckpt}" >> "${MANIFEST}"
  log "MAP task=${task} seed=${seed} ckpt=${ckpt}"
}

isaac_busy() {
  pgrep -f 'bin/python.*scripts/reinforcement_learning/sb3/train.py' >/dev/null 2>&1
}

wait_idle() {
  if isaac_busy; then
    log "waiting for current Isaac train to finish..."
  fi
  while isaac_busy; do
    sleep 60
  done
  # let watchdog flush / kit unlock
  sleep 15
}

# Print latest model_*_steps.zip for this task+seed, or nothing.
# Uses params/agent.yaml seed — never the newest zip in the task folder.
latest_ckpt_for_seed() {
  local task="$1" seed="$2"
  python3 "${CKPT_HELPER}" latest --task "${task}" --seed "${seed}"
}

run_job() {
  local task="$1" seed="$2"
  local ckpt
  ckpt="$(latest_ckpt_for_seed "${task}" "${seed}")"
  local step=0
  if [[ -n "${ckpt}" && "${ckpt}" == *model_*_steps.zip ]]; then
    step="${ckpt##*/}"
    step="${step#model_}"
    step="${step%%_steps.zip}"
  fi
  if [[ "${step}" -ge 1000000 ]]; then
    python3 "${CKPT_HELPER}" verify --checkpoint "${ckpt}" --seed "${seed}" >/dev/null
    python3 "${CKPT_HELPER}" lock --task "${task}" --seed "${seed}" --run-dir "$(dirname "${ckpt}")" >/dev/null
    log "SKIP ${task} seed=${seed} (already ${step}: ${ckpt})"
    record_map "${task}" "${seed}" "${ckpt}"
    return 0
  fi
  wait_idle
  log "START ${task} seed=${seed} 1M at $(date)"
  if [[ -n "${ckpt}" ]]; then
    python3 "${CKPT_HELPER}" verify --checkpoint "${ckpt}" --seed "${seed}" || {
      log "ABORT ${task} seed=${seed}: checkpoint seed mismatch ${ckpt}"
      return 1
    }
    log "resume ${task} seed=${seed} from ${ckpt}"
    V2_MAX_STEPS="${V2_MAX_STEPS}" ./scripts/auto_train_auv_tracking_ablation.sh \
      --task "${task}" \
      --checkpoint "${ckpt}" \
      --num_envs 16 --headless --seed "${seed}" \
      "${HIRES[@]}"
  else
    log "fresh ${task} seed=${seed} (no prior zip for this seed)"
    V2_MAX_STEPS="${V2_MAX_STEPS}" ./scripts/auto_train_auv_tracking_ablation.sh \
      --task "${task}" \
      --fresh \
      --num_envs 16 --headless --seed "${seed}" \
      "${HIRES[@]}"
  fi
  log "DONE ${task} seed=${seed} exit=$? at $(date)"
  ckpt="$(latest_ckpt_for_seed "${task}" "${seed}")"
  record_map "${task}" "${seed}" "${ckpt:-NONE}"
  if [[ -n "${ckpt}" ]]; then
    python3 "${CKPT_HELPER}" lock --task "${task}" --seed "${seed}" --run-dir "$(dirname "${ckpt}")" >/dev/null
  fi
}

echo "[queue] === 1M queue started $(date) ===" | tee -a "${QUEUE_LOG}"
log "seeds=${SEEDS[*]}  seed 66 deferred; next after idle = v2-full seed 123; do not eval until queue finishes"

# Current image-7 (or any in-flight Isaac) must finish first.
wait_idle

# User: after the current run, train v2-full seed 123 next.
run_job "Isaac-AUV-Tracking-Direct-v2" "123"

for seed in "${SEEDS[@]}"; do
  for task in "${TASKS[@]}"; do
    run_job "${task}" "${seed}"
  done
done

log "=== QUEUE FINISHED (seed 66 not included) $(date) ==="
echo QUEUE_ALL_DONE >> "${QUEUE_LOG}"
