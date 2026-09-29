#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Attach a stall watchdog to an already-running train.py, then hand off
# to watchdog_train_auv_tracking_sb3_sac_v1.sh on stall/crash (not on success).
#
# Heartbeat = timesteps (log/ckpt) OR train CPU ticks OR TensorBoard bytes.
# Terminal log freeze alone will NOT trigger a restart while the process is busy.
#
#   TRAIN_PY_PID=... SESSION_START_EPOCH=... \
#     ./scripts/attach_watchdog_auv_tracking_sb3_sac_v1.sh --num_envs 8 --headless

set -euo pipefail
cd "$(dirname "$0")/.."

export TERM="${TERM:-xterm-256color}"
[[ "${TERM}" == "dumb" ]] && export TERM=xterm-256color

STALL_SEC="${STALL_SEC:-600}"
STARTUP_GRACE_SEC="${STARTUP_GRACE_SEC:-900}"
POLL_SEC="${POLL_SEC:-30}"
TASK_LOG_ROOT="logs/sb3/Isaac-AUV-Tracking-Direct-v1"
RUN_LOG="${RUN_LOG:-/tmp/auv_vision_v1_resnet_watchdog.log}"
OUTER="${OUTER:-/tmp/auv_vision_v1_watchdog_outer.log}"

if [[ -z "${TRAIN_PY_PID:-}" ]]; then
  echo "TRAIN_PY_PID required" >&2
  exit 2
fi
if [[ -z "${SESSION_START_EPOCH:-}" ]]; then
  echo "SESSION_START_EPOCH required" >&2
  exit 2
fi

echo "[watchdog-attach] monitoring pid=${TRAIN_PY_PID} session=${SESSION_START_EPOCH}" | tee -a "${OUTER}"
echo "[watchdog-attach] heartbeat=step|cpu|tb (terminal-only freeze is OK)" | tee -a "${OUTER}"

progress_probe() {
  SESSION_START_EPOCH="${SESSION_START_EPOCH}" \
  TASK_LOG_ROOT="${TASK_LOG_ROOT}" \
  RUN_LOG="${RUN_LOG}" \
  TRAIN_PY_PID="${TRAIN_PY_PID}" \
  bash scripts/lib/auv_v1_train_progress.sh
}

session_final_model() {
  python3 - <<PY
from pathlib import Path
root = Path("${TASK_LOG_ROOT}")
session_start = int("${SESSION_START_EPOCH}")
for r in sorted([p for p in root.glob("20*") if p.is_dir()], reverse=True)[:5]:
    mz = r / "model.zip"
    try:
        if mz.exists() and mz.stat().st_mtime >= session_start - 5:
            print(mz.resolve())
            break
    except OSError:
        pass
PY
}

last_step=-1
last_cpu=-1
last_tb=-1
last_change="$(date +%s)"
started="$(date +%s)"
saw_progress=0
stalled=0

while kill -0 "${TRAIN_PY_PID}" 2>/dev/null; do
  sleep "${POLL_SEC}"
  read -r step cpu tb <<<"$(progress_probe)"
  now="$(date +%s)"
  moved=0
  if [[ "${step}" -gt "${last_step}" && "${step}" -gt 0 ]]; then
    echo "[watchdog-attach] progress step ${last_step} → ${step}" | tee -a "${OUTER}"
    last_step="${step}"
    moved=1
  fi
  if [[ "${cpu}" -gt "${last_cpu}" && "${cpu}" -gt 0 ]]; then
    last_cpu="${cpu}"
    moved=1
  fi
  if [[ "${tb}" -gt "${last_tb}" ]]; then
    echo "[watchdog-attach] progress tb_bytes ${last_tb} → ${tb}" | tee -a "${OUTER}"
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
      echo "[watchdog-attach] no progress in startup grace — kill & handoff" | tee -a "${OUTER}"
      stalled=1
      ./scripts/kill_isaac_lab_processes.sh || true
      break
    fi
    continue
  fi
  if (( now - last_change > STALL_SEC )); then
    echo "[watchdog-attach] stalled ${STALL_SEC}s (step≈${step} cpu=${cpu} tb=${tb}) — kill & handoff" | tee -a "${OUTER}"
    stalled=1
    ./scripts/kill_isaac_lab_processes.sh || true
    break
  fi
done

set +e
wait "${TRAIN_PY_PID}" 2>/dev/null
rc=$?
set -e
echo "[watchdog-attach] train pid gone exit≈${rc} stalled=${stalled}" | tee -a "${OUTER}"

FINAL="$(session_final_model || true)"
if [[ -n "${FINAL}" && "${stalled}" -eq 0 ]]; then
  echo "[watchdog-attach] training finished: ${FINAL}" | tee -a "${OUTER}"
  exit 0
fi

echo "[watchdog-attach] handing off to full watchdog (SESSION_START_EPOCH=${SESSION_START_EPOCH})" | tee -a "${OUTER}"
export SESSION_START_EPOCH
export TERM
exec ./scripts/watchdog_train_auv_tracking_sb3_sac_v1.sh "$@"
