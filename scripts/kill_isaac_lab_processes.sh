#!/bin/bash
# Stop Isaac Lab / Isaac Sim sessions started from this repo.
#
# Usage:
#   ./scripts/kill_isaac_lab_processes.sh

set -euo pipefail

# Match Isaac python jobs by script name; exclude this killer and its parent shell.
PATTERN='isaaclab/bin/python.*(eval_|debug_|reinforcement_learning|train_|play\.py)|kit_.*Isaac-Sim'
SELF=$$
PARENT=$PPID

echo "[kill] Searching for Isaac Lab / Sim processes..."
mapfile -t PIDS < <(pgrep -f "$PATTERN" 2>/dev/null || true)

FILTERED=()
for pid in "${PIDS[@]:-}"; do
    [[ -z "$pid" ]] && continue
    [[ "$pid" == "$SELF" || "$pid" == "$PARENT" ]] && continue
    # Skip if the cmdline is this kill script itself.
    if tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q 'kill_isaac_lab_processes'; then
        continue
    fi
    FILTERED+=("$pid")
done

if [[ ${#FILTERED[@]} -eq 0 ]]; then
    echo "[kill] No matching processes."
    exit 0
fi

echo "[kill] Terminating PIDs: ${FILTERED[*]}"
kill -TERM "${FILTERED[@]}" 2>/dev/null || true
sleep 2
mapfile -t ALIVE < <(pgrep -f "$PATTERN" 2>/dev/null || true)
STILL=()
for pid in "${ALIVE[@]:-}"; do
    [[ -z "$pid" ]] && continue
    [[ "$pid" == "$SELF" || "$pid" == "$PARENT" ]] && continue
    STILL+=("$pid")
done
if [[ ${#STILL[@]} -gt 0 ]]; then
    echo "[kill] Force kill: ${STILL[*]}"
    kill -9 "${STILL[@]}" 2>/dev/null || true
fi

echo "[kill] Done."
