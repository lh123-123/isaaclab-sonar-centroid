#!/bin/bash
# Auto-restart watchdog for Direct-v2 training.
# Monitors training; on hang/crash, kills and resumes from latest checkpoint.
# Runs until total_timesteps reaches n_timesteps (default 3M) or max_restarts.
#
#   ./scripts/auto_train_auv_tracking_flow_v2.sh --num_envs 16 --headless
#   V2_MAX_STEPS=3000000 ./scripts/auto_train_auv_tracking_flow_v2.sh --fresh --num_envs 16 --headless
#
# No need to babysit — it restarts itself.

set -uo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

# --- config ---
TASK="Isaac-AUV-Tracking-Direct-v2"
LOG_DIR="logs/sb3/${TASK}"
TRAIN_LOG="/tmp/v2_autotrain.log"
MAX_STEPS="${V2_MAX_STEPS:-3000000}"
MAX_RESTARTS=20
STALL_CHECK_INTERVAL=90      # seconds between stall checks
STALL_THRESHOLD=10           # consecutive stalled checks = hung (Isaac Sim+cameras needs ~10min to start)
RESTART_DELAY=5              # seconds to wait before restarting
CURRENT_RUN_DIR=""           # tracked from train.py log output

# --- args passthrough ---
TRAIN_ARGS=()
CHECKPOINT_ARG=""
FRESH=0
ATTACH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT_ARG="$2"; shift 2;;
    --max_restarts)
      MAX_RESTARTS="$2"; shift 2;;
    --fresh)
      FRESH=1; shift;;
    --attach)
      ATTACH=1; shift;;
    *)
      TRAIN_ARGS+=("$1"); shift;;
  esac
done

restart_count=0
last_timesteps=0
stall_count=0

get_latest_checkpoint() {
  # Prefer checkpoint from the CURRENT run dir (so we don't jump to a different sweep config).
  # Fall back to global latest only if no current run dir tracked yet.
  local latest
  if [[ -n "${CURRENT_RUN_DIR}" && -d "${CURRENT_RUN_DIR}" ]]; then
    latest=$(find "${CURRENT_RUN_DIR}" -name "model_*_steps.zip" -printf '%T@ %p\n' 2>/dev/null \
      | sort -rn | head -1 | cut -d' ' -f2-)
    [[ -n "${latest}" ]] && { echo "${latest}"; return; }
  fi
  latest=$(find "${LOG_DIR}" -name "model_*_steps.zip" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-)
  echo "${latest}"
}

get_current_run_dir() {
  # train.py prints log_root then "Exact experiment name ...: <timestamp>".
  # Checkpoints live in log_root/<timestamp>/, not the parent task folder.
  local root name sb3
  root=$(grep -a "Logging experiment in directory:" "${TRAIN_LOG}" 2>/dev/null \
    | tail -1 | sed -E 's/.*directory: *//; s/[[:space:]]*$//')
  name=$(grep -a "Exact experiment name requested from command line:" "${TRAIN_LOG}" 2>/dev/null \
    | tail -1 | sed -E 's/.*command line: *//; s/[[:space:]]*$//')
  if [[ -n "${root}" && -n "${name}" ]]; then
    echo "${root}/${name}"
    return
  fi
  sb3=$(grep -aE "Logging to .*/Isaac-AUV-Tracking-Direct-v2/" "${TRAIN_LOG}" 2>/dev/null \
    | tail -1 | sed -E 's/.*Logging to *//; s|/SAC_[0-9]+[[:space:]]*$||; s/[[:space:]]*$//')
  echo "${sb3}"
}

pick_restart_checkpoint() {
  # 1) newest checkpoint in CURRENT_RUN_DIR (same sweep config, safe)
  # 2) explicit CHECKPOINT_ARG passed at launch (the best-config seed)
  # 3) global latest (last resort)
  local ckpt
  ckpt=$(get_latest_checkpoint)
  if [[ -n "${ckpt}" ]]; then echo "${ckpt}"; return; fi
  if [[ -n "${CHECKPOINT_ARG}" ]]; then echo "${CHECKPOINT_ARG}"; return; fi
  echo ""
}

get_latest_timesteps() {
  # Extract latest total_timesteps from training log
  grep -aE "total_timesteps" "${TRAIN_LOG}" 2>/dev/null \
    | tail -1 | grep -oE "[0-9]+" | tail -1
}

is_proc_alive() {
  pgrep -f "bin/python.*train.py.*${TASK}" >/dev/null 2>&1
}

kill_training() {
  echo "[watchdog] killing training process..." | tee -a "${TRAIN_LOG}.watchdog"
  ./scripts/kill_isaac_lab_processes.sh 2>&1 | tee -a "${TRAIN_LOG}.watchdog"
  sleep 3
}

start_training() {
  local ckpt="$1"
  local args=("--task" "${TASK}" "--agent" "sb3_cfg_entry_point"
              "--algorithm" "SAC" "--log_interval" "20000" "--enable_cameras"
              "--max_iterations" "${MAX_STEPS}")
  args+=("${TRAIN_ARGS[@]}")
  if [[ -n "${ckpt}" ]]; then
    args+=("--checkpoint" "${ckpt}")
    echo "[watchdog] restart #${restart_count}: resuming from ${ckpt}" | tee -a "${TRAIN_LOG}.watchdog"
  else
    echo "[watchdog] start #${restart_count}: fresh training (no checkpoint)" | tee -a "${TRAIN_LOG}.watchdog"
  fi
  # Clear log for new run (keep old as .prev)
  [[ -f "${TRAIN_LOG}" ]] && mv "${TRAIN_LOG}" "${TRAIN_LOG}.prev"
  ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py "${args[@]}" \
    > "${TRAIN_LOG}" 2>&1 &
  echo "[watchdog] training PID=$! started at $(date)" | tee -a "${TRAIN_LOG}.watchdog"
  # Wait briefly then capture the run dir this training created
  sleep 20
  CURRENT_RUN_DIR=$(get_current_run_dir)
  [[ -n "${CURRENT_RUN_DIR}" ]] && echo "[watchdog] current run dir: ${CURRENT_RUN_DIR}" | tee -a "${TRAIN_LOG}.watchdog"
}

# --- main loop ---
echo "[watchdog] === auto-restart training started $(date) ===" | tee -a "${TRAIN_LOG}.watchdog"
echo "[watchdog] target=${MAX_STEPS} steps, max_restarts=${MAX_RESTARTS}" | tee -a "${TRAIN_LOG}.watchdog"

# Initial start
if [[ "${ATTACH}" -eq 1 ]]; then
  if ! is_proc_alive; then
    echo "[watchdog] --attach: no training process found. Giving up." | tee -a "${TRAIN_LOG}.watchdog"
    exit 1
  fi
  CURRENT_RUN_DIR=$(get_current_run_dir)
  echo "[watchdog] --attach: monitoring existing process, run dir=${CURRENT_RUN_DIR:-unknown}" | tee -a "${TRAIN_LOG}.watchdog"
else
  ensure_single_isaac_process
  ckpt="${CHECKPOINT_ARG}"
  # Resume prior run unless --fresh (new obs / new experiment).
  if [[ -z "${ckpt}" && "${FRESH}" -eq 0 ]]; then
    ckpt=$(get_latest_checkpoint)
    [[ -n "${ckpt}" ]] && echo "[watchdog] found existing checkpoint: ${ckpt}" | tee -a "${TRAIN_LOG}.watchdog"
  elif [[ "${FRESH}" -eq 1 ]]; then
    echo "[watchdog] --fresh: ignoring prior checkpoints, starting from scratch" | tee -a "${TRAIN_LOG}.watchdog"
    ckpt=""
  fi
  start_training "${ckpt}"
fi
last_timesteps=0
stall_count=0
sleep 30  # let it boot

while true; do
  sleep "${STALL_CHECK_INTERVAL}"

  ts=$(get_latest_timesteps)
  ts=${ts:-0}

  # Check completion
  if [[ ${ts} -ge ${MAX_STEPS} ]]; then
    echo "[watchdog] training reached ${ts} >= ${MAX_STEPS} steps. DONE at $(date)" | tee -a "${TRAIN_LOG}.watchdog"
    break
  fi

  # Check if process died
  if ! is_proc_alive; then
    echo "[watchdog] process died at ${ts} steps. Restarting..." | tee -a "${TRAIN_LOG}.watchdog"
    restart_count=$((restart_count + 1))
    if [[ ${restart_count} -ge ${MAX_RESTARTS} ]]; then
      echo "[watchdog] max_restarts (${MAX_RESTARTS}) reached. Giving up." | tee -a "${TRAIN_LOG}.watchdog"
      break
    fi
    ckpt=$(pick_restart_checkpoint)
    start_training "${ckpt}"
    last_timesteps=0
    stall_count=0
    sleep 30
    continue
  fi

  # Check for stall (steps not advancing)
  if [[ ${ts} -eq ${last_timesteps} ]]; then
    stall_count=$((stall_count + 1))
    echo "[watchdog] stall check ${stall_count}/${STALL_THRESHOLD}: timesteps=${ts} (no progress)" | tee -a "${TRAIN_LOG}.watchdog"
    if [[ ${stall_count} -ge ${STALL_THRESHOLD} ]]; then
      echo "[watchdog] HANG DETECTED at ${ts} steps after ${STALL_THRESHOLD} stalled checks. Killing+restarting..." | tee -a "${TRAIN_LOG}.watchdog"
      kill_training
      restart_count=$((restart_count + 1))
      if [[ ${restart_count} -ge ${MAX_RESTARTS} ]]; then
        echo "[watchdog] max_restarts (${MAX_RESTARTS}) reached. Giving up." | tee -a "${TRAIN_LOG}.watchdog"
        break
      fi
      ckpt=$(pick_restart_checkpoint)
      start_training "${ckpt}"
      last_timesteps=0
      stall_count=0
      sleep 30
    fi
  else
    stall_count=0
    last_timesteps=${ts}
    # Periodic progress report
    echo "[watchdog] OK: timesteps=${ts}/${MAX_STEPS} ($(echo "scale=1; ${ts}*100/${MAX_STEPS}" | bc)%) restarts=${restart_count}" | tee -a "${TRAIN_LOG}.watchdog"
  fi
done

echo "[watchdog] === finished at $(date), total_restarts=${restart_count} ===" | tee -a "${TRAIN_LOG}.watchdog"
