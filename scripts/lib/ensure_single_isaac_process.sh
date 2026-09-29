#!/usr/bin/env bash
# Guard: only one Isaac Lab / Isaac Sim session at a time (GPU + kit lock).
#
# Source from launch scripts:
#   source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
#   ensure_single_isaac_process
#
# Also used by .cursor/hooks/block-second-isaac.sh (Agent beforeShellExecution).

list_running_isaac_processes() {
    local running="" python_running kit_running watchdog_running

    # Running Isaac Lab Python entrypoints (not the bash wrapper scripts).
    if pgrep -af "/isaaclab/bin/python" >/dev/null 2>&1; then
        python_running="$(pgrep -af "/isaaclab/bin/python" 2>/dev/null \
            | grep -E "scripts/(eval|debug)_sonar_flow|reinforcement_learning|train_|play\.py" \
            | grep -v grep || true)"
        if [[ -n "$python_running" ]]; then
            running="$python_running"
        fi
    fi

    # Omniverse Kit (Isaac Sim) already up.
    kit_running="$(pgrep -af "kit_.*Isaac-Sim|omni\.kit.*isaaclab" 2>/dev/null | grep -v grep || true)"
    if [[ -n "$kit_running" ]]; then
        running="${running}${running:+$'\n'}${kit_running}"
    fi

    # Watchdog wrapping a train job (python may not be up yet).
    watchdog_running="$(pgrep -af "watchdog_train_auv" 2>/dev/null | grep -v grep || true)"
    if [[ -n "$watchdog_running" ]]; then
        running="${running}${running:+$'\n'}${watchdog_running}"
    fi

    if [[ -n "$running" ]]; then
        printf '%s\n' "$running"
        return 0
    fi
    return 1
}

ensure_single_isaac_process() {
    local running=""
    running="$(list_running_isaac_processes || true)"

    if [[ -n "$running" ]]; then
        echo "[ERROR] 约定：终端同时只跑一个 Isaac Lab / Isaac Sim 进程。" >&2
        echo "[ERROR] 检测到已有进程：" >&2
        echo "$running" >&2
        echo "[ERROR] 请先结束上述进程，或运行: ./scripts/kill_isaac_lab_processes.sh" >&2
        return 1
    fi
    return 0
}
