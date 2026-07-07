#!/bin/bash
# Stop Isaac Lab / Isaac Sim sessions started from this repo.
#
# Usage:
#   ./scripts/kill_isaac_lab_processes.sh

set -euo pipefail

PATTERN='/isaaclab/bin/python.*(eval_sonar_flow|debug_sonar_flow|reinforcement_learning|train_|play\.py)|kit_.*Isaac-Sim'

echo "[kill] Searching for Isaac Lab / Sim processes..."
mapfile -t PIDS < <(pgrep -f "$PATTERN" 2>/dev/null || true)

if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo "[kill] No matching processes."
    exit 0
fi

echo "[kill] Terminating PIDs: ${PIDS[*]}"
kill -TERM "${PIDS[@]}" 2>/dev/null || true
sleep 2
mapfile -t ALIVE < <(pgrep -f "$PATTERN" 2>/dev/null || true)
if [[ ${#ALIVE[@]} -gt 0 ]]; then
    echo "[kill] Force kill: ${ALIVE[*]}"
    kill -9 "${ALIVE[@]}" 2>/dev/null || true
fi

echo "[kill] Done."
