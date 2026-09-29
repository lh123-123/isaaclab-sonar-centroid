#!/bin/bash
# Generic auto-restart watchdog for Direct-v2 ablation training.
# Usage:
#   ./scripts/auto_train_auv_tracking_ablation.sh --task Isaac-AUV-Tracking-Direct-v2-state --num_envs 16 --headless
#   ./scripts/auto_train_auv_tracking_ablation.sh --task Isaac-AUV-Tracking-Direct-v2-image --num_envs 16 --headless
#
# Monitors training; on hang/crash, kills and resumes from latest checkpoint.

set -uo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

# --- config ---
TASK=""
AGENT="sb3_cfg_entry_point"
LOG_DIR=""
TRAIN_LOG="/tmp/v2_ablation_train.log"
MAX_STEPS="${V2_MAX_STEPS:-1000000}"
MAX_RESTARTS=20
STALL_CHECK_INTERVAL=90
STALL_THRESHOLD=10
RESTART_DELAY=5
CURRENT_RUN_DIR=""

# --- args ---
TRAIN_ARGS=()
CHECKPOINT_ARG=""
FRESH=0
SEED=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2;;
    --agent) AGENT="$2"; shift 2;;
    --checkpoint) CHECKPOINT_ARG="$2"; shift 2;;
    --max_restarts) MAX_RESTARTS="$2"; shift 2;;
    --seed) SEED="$2"; TRAIN_ARGS+=("$1" "$2"); shift 2;;
    --fresh) FRESH=1; shift;;
    *) TRAIN_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "${TASK}" ]]; then
  echo "ERROR: --task is required"
  exit 1
fi

LOG_DIR="logs/sb3/${TASK}"
SEED_TAG="${SEED:-noseed}"
WATCHDOG_LOG="/tmp/v2_ablation_watchdog_${TASK//[^a-zA-Z0-9]/_}_s${SEED_TAG}.log"
CKPT_HELPER="./scripts/lib/auv_seed_ckpt.py"

restart_count=0
last_timesteps=0
stall_count=0

get_latest_checkpoint() {
  # Only zips whose params/agent.yaml seed matches. Never "newest in the whole task dir".
  local extra=()
  if [[ -z "${SEED}" ]]; then
    echo "[watchdog] REFUSING unfiltered checkpoint search (no --seed)" >&2
    echo ""
    return
  fi
  if [[ -n "${CURRENT_RUN_DIR}" && -d "${CURRENT_RUN_DIR}" ]]; then
    extra+=(--run-dir "${CURRENT_RUN_DIR}")
  fi
  python3 "${CKPT_HELPER}" latest --task "${TASK}" --seed "${SEED}" "${extra[@]}"
}

verify_checkpoint_seed() {
  local ckpt="$1"
  [[ -z "${ckpt}" ]] && return 0
  if [[ -z "${SEED}" ]]; then
    echo "[watchdog] REFUSING to load ${ckpt} without --seed" | tee -a "${WATCHDOG_LOG}"
    return 1
  fi
  if ! python3 "${CKPT_HELPER}" verify --checkpoint "${ckpt}" --seed "${SEED}"; then
    echo "[watchdog] REFUSING checkpoint (seed mismatch): ${ckpt} vs seed=${SEED}" | tee -a "${WATCHDOG_LOG}"
    return 1
  fi
  return 0
}

get_current_run_dir() {
  local sb3
  sb3=$(grep -aE "Logging to .*/${TASK}/" "${TRAIN_LOG}" 2>/dev/null \
    | tail -1 | sed -E 's/.*Logging to *//; s|/SAC_[0-9]+[[:space:]]*$||; s/[[:space:]]*$//')
  echo "${sb3}"
}

pick_restart_checkpoint() {
  local ckpt
  ckpt=$(get_latest_checkpoint)
  if [[ -n "${ckpt}" ]]; then echo "${ckpt}"; return; fi
  if [[ -n "${CHECKPOINT_ARG}" ]]; then echo "${CHECKPOINT_ARG}"; return; fi
  echo ""
}

get_latest_timesteps() {
  grep -aE "total_timesteps" "${TRAIN_LOG}" 2>/dev/null \
    | tail -1 | grep -oE "[0-9]+" | tail -1
}

is_proc_alive() {
  # Require both task and seed so v2 / v2-image / another seed cannot attach.
  if [[ -n "${SEED}" ]]; then
    pgrep -f "bin/python.*train.py.*--task ${TASK} .*--seed ${SEED}" >/dev/null 2>&1
  else
    pgrep -f "bin/python.*train.py.*--task ${TASK} " >/dev/null 2>&1
  fi
}

kill_training() {
  echo "[watchdog] killing training process..." | tee -a "${WATCHDOG_LOG}"
  ./scripts/kill_isaac_lab_processes.sh 2>&1 | tee -a "${WATCHDOG_LOG}"
  sleep 3
}

start_training() {
  local ckpt="$1"
  local args=("--task" "${TASK}" "--agent" "${AGENT}"
              "--algorithm" "SAC" "--log_interval" "20000" "--enable_cameras"
              "--max_iterations" "${MAX_STEPS}")
  args+=("${TRAIN_ARGS[@]}")
  if [[ -n "${ckpt}" ]]; then
    if ! verify_checkpoint_seed "${ckpt}"; then
      echo "[watchdog] abort start: would mix seeds" | tee -a "${WATCHDOG_LOG}"
      return 1
    fi
    args+=("--checkpoint" "${ckpt}")
    echo "[watchdog] restart #${restart_count}: seed=${SEED} resuming from ${ckpt}" | tee -a "${WATCHDOG_LOG}"
  else
    echo "[watchdog] start #${restart_count}: fresh training seed=${SEED}" | tee -a "${WATCHDOG_LOG}"
  fi
  [[ -f "${TRAIN_LOG}" ]] && mv "${TRAIN_LOG}" "${TRAIN_LOG}.prev"
  ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py "${args[@]}" \
    > "${TRAIN_LOG}" 2>&1 &
  echo "[watchdog] training PID=$! started at $(date)" | tee -a "${WATCHDOG_LOG}"
  sleep 20
  CURRENT_RUN_DIR=$(get_current_run_dir || true)
  if [[ -n "${CURRENT_RUN_DIR}" ]]; then
    echo "[watchdog] current run dir: ${CURRENT_RUN_DIR}" | tee -a "${WATCHDOG_LOG}"
  fi
  return 0
}

# --- main ---
echo "[watchdog] === ablation training: ${TASK} seed=${SEED:-?} started $(date) ===" | tee -a "${WATCHDOG_LOG}"
echo "[watchdog] target=${MAX_STEPS} steps, max_restarts=${MAX_RESTARTS}" | tee -a "${WATCHDOG_LOG}"
if [[ -n "${CHECKPOINT_ARG}" ]]; then
  verify_checkpoint_seed "${CHECKPOINT_ARG}" || exit 1
fi

if [[ "${FRESH}" -eq 1 ]]; then
  echo "[watchdog] --fresh: starting from scratch" | tee -a "${WATCHDOG_LOG}"
  ensure_single_isaac_process
  start_training ""
else
  # Attach to existing process if running
  if is_proc_alive; then
    echo "[watchdog] attaching to existing training process" | tee -a "${WATCHDOG_LOG}"
    CURRENT_RUN_DIR=$(get_current_run_dir)
  else
    ensure_single_isaac_process
    ckpt="${CHECKPOINT_ARG}"
    if [[ -z "${ckpt}" ]]; then
      ckpt=$(get_latest_checkpoint)
      [[ -n "${ckpt}" ]] && echo "[watchdog] found existing checkpoint: ${ckpt}" | tee -a "${WATCHDOG_LOG}"
    fi
    start_training "${ckpt}"
  fi
fi

last_timesteps=0
stall_count=0
sleep 30

while true; do
  sleep "${STALL_CHECK_INTERVAL}"
  ts=$(get_latest_timesteps)
  ts=${ts:-0}

  if [[ ${ts} -ge ${MAX_STEPS} ]]; then
    echo "[watchdog] DONE: ${ts} >= ${MAX_STEPS} at $(date)" | tee -a "${WATCHDOG_LOG}"
    break
  fi

  if ! is_proc_alive; then
    echo "[watchdog] process died at ${ts} steps. Restarting..." | tee -a "${WATCHDOG_LOG}"
    restart_count=$((restart_count + 1))
    if [[ ${restart_count} -ge ${MAX_RESTARTS} ]]; then
      echo "[watchdog] max_restarts reached. Giving up." | tee -a "${WATCHDOG_LOG}"
      break
    fi
    ckpt=$(pick_restart_checkpoint)
    start_training "${ckpt}" || break
    last_timesteps=0
    stall_count=0
    sleep 30
    continue
  fi

  if [[ ${ts} -eq ${last_timesteps} ]]; then
    stall_count=$((stall_count + 1))
    echo "[watchdog] stall check ${stall_count}/${STALL_THRESHOLD}: timesteps=${ts}" | tee -a "${WATCHDOG_LOG}"
    if [[ ${stall_count} -ge ${STALL_THRESHOLD} ]]; then
      echo "[watchdog] HANG at ${ts} steps. Killing+restarting..." | tee -a "${WATCHDOG_LOG}"
      kill_training
      restart_count=$((restart_count + 1))
      if [[ ${restart_count} -ge ${MAX_RESTARTS} ]]; then
        echo "[watchdog] max_restarts reached. Giving up." | tee -a "${WATCHDOG_LOG}"
        break
      fi
      ckpt=$(pick_restart_checkpoint)
      start_training "${ckpt}" || break
      last_timesteps=0
      stall_count=0
      sleep 30
    fi
  else
    stall_count=0
    last_timesteps=${ts}
    CURRENT_RUN_DIR=$(get_current_run_dir)
    pct=$(echo "scale=1; ${ts}*100/${MAX_STEPS}" | bc)
    echo "[watchdog] OK: timesteps=${ts}/${MAX_STEPS} (${pct}%) seed=${SEED} run=${CURRENT_RUN_DIR} restarts=${restart_count}" | tee -a "${WATCHDOG_LOG}"
  fi
done

echo "[watchdog] === finished at $(date), total_restarts=${restart_count} ===" | tee -a "${WATCHDOG_LOG}"
