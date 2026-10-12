#!/usr/bin/env bash
# Guard: only one Isaac Lab / Isaac Sim session at a time (GPU + kit lock).
#
# Source from launch scripts:
#   source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
#   ensure_single_isaac_process
#
# Also used by .cursor/hooks/block-second-isaac.sh (Agent beforeShellExecution).

# PIDs of this shell and all its ancestors.
#
# A watchdog that launched us sits in this chain and must NOT be mistaken for a competing session.
# The guard is sourced by the train script *inside* the watchdog, so flagging our own supervisor
# deadlocks the pair -- the watchdog can never start the job it exists to supervise. Before this
# exclusion, `watchdog_train_auv_tracking_sb3_sac_v4.sh` aborted with
#   [ERROR] 检测到已有进程：<the watchdog's own pid>
# which silently broke every watchdog_* script (v0_sonar, v0_gt_smooth, v1, v4).
_ppid_of() {
    # /proc/<pid>/stat is "pid (comm) state ppid ...". comm may itself contain spaces and
    # parentheses, so strip through the LAST ")" before splitting into fields; the parent pid is
    # then field 2 of what remains.
    sed 's/^[^)]*) //' "/proc/${1}/stat" 2>/dev/null | awk '{print $2}'
}

_self_and_ancestor_pids() {
    local pid="$$" out="" next
    while [[ -n "${pid}" && "${pid}" -gt 1 ]]; do
        out="${out}${out:+ }${pid}"
        next="$(_ppid_of "${pid}" || true)"
        [[ -z "${next}" || "${next}" == "${pid}" ]] && break
        pid="${next}"
    done
    printf '%s' "${out}"
}

# stdin: "pid cmdline" lines. stdout: the same lines minus those whose pid is in $1.
_drop_ancestors() {
    local ancestors="$1" line pid
    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        pid="${line%% *}"
        [[ " ${ancestors} " == *" ${pid} "* ]] && continue
        printf '%s\n' "${line}"
    done
}

list_running_isaac_processes() {
    local running="" python_running kit_running watchdog_running
    local ancestors
    ancestors="$(_self_and_ancestor_pids)"

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

    # Watchdog wrapping a train job (python may not be up yet). Ancestors are skipped: one of them
    # is our own supervisor, which is not a conflict.
    watchdog_running="$(pgrep -af "watchdog_train_auv" 2>/dev/null | grep -v grep | _drop_ancestors "${ancestors}" || true)"
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
