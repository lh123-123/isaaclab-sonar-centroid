#!/usr/bin/python3
"""beforeShellExecution: deny a second Isaac Lab / Isaac Sim launch."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

LAUNCH_RE = re.compile(
    r"isaaclab\.sh"
    r"|/isaaclab/bin/python"
    r"|watchdog_train"
    r"|scripts/(?:train_|eval_|debug_|teach_sonar|sweep_sonar)"
    r"|train_auv_tracking"
    r"|eval_auv_tracking"
    r"|(?:^|[\s;|&])(?:python3?)\s+\S*(?:train|play)\.py"
    r"|Isaac-Sim"
    r"|isaac-sim",
    re.IGNORECASE,
)
INSPECT_HEAD_RE = re.compile(
    r"^(?:[A-Za-z_][\w.]*=\S*\s+)*(?:pgrep|pkill|nvidia-smi)\b"
)
KILL_SCRIPT_RE = re.compile(r"kill_isaac_lab_processes(?:\.sh)?")


def emit(permission: str, user_message: str = "", agent_message: str = "") -> None:
    out: dict[str, str] = {"permission": permission}
    if user_message:
        out["user_message"] = user_message
    if agent_message:
        out["agent_message"] = agent_message
    print(json.dumps(out, ensure_ascii=False))


def segments(command: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*(?:&&|\|\||;)\s*", command) if part.strip()]


def is_launch_segment(segment: str) -> bool:
    if INSPECT_HEAD_RE.match(segment):
        return False
    if KILL_SCRIPT_RE.search(segment):
        remainder = KILL_SCRIPT_RE.sub(" ", segment)
        return LAUNCH_RE.search(remainder) is not None
    return LAUNCH_RE.search(segment) is not None


def command_starts_isaac(command: str) -> bool:
    return any(is_launch_segment(part) for part in segments(command))


def list_running_isaac_processes() -> str:
    script = f'''
source "{ROOT}/scripts/lib/ensure_single_isaac_process.sh"
list_running_isaac_processes || true
'''
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return (result.stdout or "").strip()


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        emit("allow")
        return 0

    command = str(payload.get("command") or "")
    if not command_starts_isaac(command):
        emit("allow")
        return 0

    running = list_running_isaac_processes()
    if not running:
        emit("allow")
        return 0

    user_msg = (
        "已有 Isaac Lab / Isaac Sim 在跑，不能再开第二个。"
        "先运行 ./scripts/kill_isaac_lab_processes.sh 并确认干净，或等当前任务结束。\n"
        f"检测到：\n{running}"
    )
    agent_msg = (
        "Hook blocked a second Isaac launch. Occupied processes:\n"
        f"{running}\n"
        "Do not start another isaaclab.sh / train.py / play / watchdog. "
        "If the user asked to kill first, run ./scripts/kill_isaac_lab_processes.sh, "
        "confirm clean, then launch exactly one job."
    )
    emit("deny", user_msg, agent_msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
