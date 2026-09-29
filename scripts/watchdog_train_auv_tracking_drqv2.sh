#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Watchdog for Direct-v1 DrQ-v2: if timesteps stall for STALL_SEC, kill and resume
# from the newest checkpoint_*.pt written during this watchdog session.
#
#   ./scripts/watchdog_train_auv_tracking_drqv2.sh --num_envs 8 --headless
#
# Optional env:
#   STALL_SEC=600          # no progress → restart (default 10 min)
#   STARTUP_GRACE_SEC=900  # wait this long for first progress before stalling
#   POLL_SEC=30

set -euo pipefail
cd "$(dirname "$0")/.."

# isaaclab.sh runs `tabs 4` under set -e; TERM=dumb (nohup/CI) makes tabs fail.
export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

STALL_SEC="${STALL_SEC:-600}"
STARTUP_GRACE_SEC="${STARTUP_GRACE_SEC:-900}"
POLL_SEC="${POLL_SEC:-30}"
TASK_LOG_ROOT="logs/drqv2/Isaac-AUV-Tracking-Direct-v1"
RUN_LOG="${RUN_LOG:-/tmp/drqv2_watchdog.log}"
NUM_ENVS=8

FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_envs)
      NUM_ENVS="$2"
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --num_envs=*)
      NUM_ENVS="${1#*=}"
      FORWARD_ARGS+=("$1")
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

has_num_envs=0
for a in "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"; do
  [[ "$a" == "--num_envs" || "$a" == --num_envs=* ]] && has_num_envs=1
done
if [[ "$has_num_envs" -eq 0 ]]; then
  FORWARD_ARGS=(--num_envs "$NUM_ENVS" "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}")
fi

progress_probe() {
  SESSION_START_EPOCH="${SESSION_START_EPOCH}" \
  TASK_LOG_ROOT="${TASK_LOG_ROOT}" \
  RUN_LOG="${RUN_LOG}" \
  TRAIN_PID="${TRAIN_PID:-}" \
  TRAIN_PY_PID="${TRAIN_PY_PID:-}" \
  bash scripts/lib/auv_drqv2_train_progress.sh
}

train_finished() {
  python3 - <<PY
from pathlib import Path
root = Path("${TASK_LOG_ROOT}")
session_start = int("${SESSION_START_EPOCH}")
runs = sorted([p for p in root.glob("20*") if p.is_dir()], reverse=True)
for r in runs[:8]:
    mz = r / "checkpoint_final.pt"
    try:
        if mz.exists() and mz.stat().st_mtime >= session_start - 5:
            print(str(mz.resolve()))
            break
    except OSError:
        pass
PY
}

# Only resume from checkpoints written during this watchdog session.
SESSION_START_EPOCH="${SESSION_START_EPOCH:-$(date +%s)}"
SESSION_MARKER="${TASK_LOG_ROOT}/.watchdog_session_${SESSION_START_EPOCH}"
mkdir -p "${TASK_LOG_ROOT}"
echo "${SESSION_START_EPOCH}" >"${SESSION_MARKER}"

latest_ckpt_this_session() {
  python3 - <<PY
from pathlib import Path
root = Path("${TASK_LOG_ROOT}")
session_start = int("${SESSION_START_EPOCH}")
best = None
best_step = -1
for p in root.glob("*/checkpoint_*.pt"):
    try:
        if p.stat().st_mtime < session_start - 5:
            continue
    except OSError:
        continue
    stem = p.stem
    if stem == "checkpoint_final":
        continue
    try:
        step = int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        continue
    if step > best_step:
        best_step = step
        best = p
if best is not None:
    print(best.resolve())
PY
}

echo "[watchdog-drqv2] STALL_SEC=${STALL_SEC} STARTUP_GRACE_SEC=${STARTUP_GRACE_SEC} POLL_SEC=${POLL_SEC}"
echo "[watchdog-drqv2] log → ${RUN_LOG}"
echo "[watchdog-drqv2] args → ${FORWARD_ARGS[*]}"
echo "[watchdog-drqv2] session_start=${SESSION_START_EPOCH} (only resume ckpts from this session)"

attempt=0
while true; do
  attempt=$((attempt + 1))
  CKPT="$(latest_ckpt_this_session || true)"
  EXTRA=()
  if [[ -n "${CKPT}" ]]; then
    echo "[watchdog-drqv2] attempt=${attempt} resume from ${CKPT}"
    EXTRA=(--checkpoint "${CKPT}")
  else
    echo "[watchdog-drqv2] attempt=${attempt} fresh start (no session checkpoint)"
  fi

  echo "" >>"${RUN_LOG}"
  echo "===== watchdog-drqv2 attempt ${attempt} $(date -Is) =====" >>"${RUN_LOG}"
  set +e
  ./scripts/train_auv_tracking_drqv2.sh "${FORWARD_ARGS[@]}" "${EXTRA[@]}" \
    >>"${RUN_LOG}" 2>&1 &
  TRAIN_PID=$!
  set -e

  last_step=-1
  last_cpu=-1
  last_tb=-1
  last_change="$(date +%s)"
  started="$(date +%s)"
  saw_progress=0

  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep "${POLL_SEC}"
    # shellcheck disable=SC2034
    read -r step cpu tb <<<"$(progress_probe)"
    now="$(date +%s)"
    moved=0
    if [[ "${step}" -gt "${last_step}" && "${step}" -gt 0 ]]; then
      echo "[watchdog-drqv2] progress step ${last_step} → ${step}"
      last_step="${step}"
      moved=1
    fi
    if [[ "${cpu}" -gt "${last_cpu}" && "${cpu}" -gt 0 ]]; then
      last_cpu="${cpu}"
      moved=1
    fi
    if [[ "${tb}" -gt "${last_tb}" ]]; then
      echo "[watchdog-drqv2] progress tb_bytes ${last_tb} → ${tb}"
      last_tb="${tb}"
      moved=1
    fi
    if [[ "${moved}" -eq 1 ]]; then
      last_change="${now}"
      if [[ "${step}" -gt 0 || "${tb}" -gt 0 ]]; then
        saw_progress=1
      fi
    fi
    if [[ "${saw_progress}" -eq 0 ]]; then
      if (( now - started > STARTUP_GRACE_SEC )); then
        echo "[watchdog-drqv2] no progress within startup grace ${STARTUP_GRACE_SEC}s — kill & retry"
        ./scripts/kill_isaac_lab_processes.sh || true
        wait "${TRAIN_PID}" 2>/dev/null || true
        break
      fi
      continue
    fi
    if (( now - last_change > STALL_SEC )); then
      echo "[watchdog-drqv2] stalled ${STALL_SEC}s (step≈${step} cpu=${cpu} tb=${tb}) — kill & resume"
      ./scripts/kill_isaac_lab_processes.sh || true
      wait "${TRAIN_PID}" 2>/dev/null || true
      break
    fi
  done

  set +e
  wait "${TRAIN_PID}"
  rc=$?
  set -e
  echo "[watchdog-drqv2] train exit_code=${rc}"

  FINAL="$(train_finished || true)"
  if [[ -n "${FINAL}" && "${rc}" -eq 0 ]]; then
    echo "[watchdog-drqv2] training finished: ${FINAL}"
    exit 0
  fi
  if [[ "${rc}" -eq 0 ]]; then
    echo "[watchdog-drqv2] train exited 0 — stop watchdog"
    exit 0
  fi

  echo "[watchdog-drqv2] restarting in 5s..."
  sleep 5
  ./scripts/kill_isaac_lab_processes.sh || true
  sleep 2
done
