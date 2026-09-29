#!/bin/bash
# Shared progress / heartbeat probe for AUV v1 train watchdogs.
# Prints: <step> <cpu_ticks> <tb_bytes>
# - step: max of progress-bar parse + session checkpoints
# - cpu_ticks: utime+stime of train.py (or TRAIN_PID tree)
# - tb_bytes: total size of session TensorBoard event files
#
# Env: TASK_LOG_ROOT, RUN_LOG, SESSION_START_EPOCH, TRAIN_PY_PID (optional), TRAIN_PID (optional),
#      TOTAL_STEPS (optional; progress-bar budget, default 5000000 = Direct-v1)

python3 - <<'PY'
import os, re
from pathlib import Path

session_start = int(os.environ["SESSION_START_EPOCH"])
root = Path(os.environ.get("TASK_LOG_ROOT", "logs/sb3/Isaac-AUV-Tracking-Direct-v1"))
log = Path(os.environ.get("RUN_LOG", "/tmp/auv_vision_v1_resnet_watchdog.log"))

step = 0
if log.exists():
    try:
        data = log.read_bytes()[-200000:].decode("utf-8", errors="replace")
        total_steps = os.environ.get("TOTAL_STEPS", "5,000,000")
        found = re.findall(r"([0-9,]+)/" + re.escape(total_steps), data)
        if found:
            step = max(int(x.replace(",", "")) for x in found)
    except OSError:
        pass

tb_bytes = 0
for p in root.glob("*/model_*_steps.zip"):
    try:
        if p.stat().st_mtime < session_start - 5:
            continue
        step = max(step, int(p.stem.split("_")[1]))
    except (OSError, IndexError, ValueError):
        pass

for p in root.glob("*/SAC_*/events.out.tfevents*"):
    try:
        st = p.stat()
        if st.st_mtime < session_start - 5:
            continue
        tb_bytes += st.st_size
    except OSError:
        pass

def cpu_ticks(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
            parts = f.read().split()
        # fields 14,15 are utime, stime (1-indexed) → indices 13,14
        return int(parts[13]) + int(parts[14])
    except (OSError, IndexError, ValueError):
        return 0

def children(pid: int) -> list[int]:
    out = []
    try:
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                with open(d / "stat", "r", encoding="utf-8") as f:
                    # comm can contain spaces inside (); split carefully
                    s = f.read()
                rparen = s.rfind(")")
                rest = s[rparen + 2 :].split()
                ppid = int(rest[1])  # after (comm): state ppid ...
                if ppid == pid:
                    out.append(int(d.name))
            except (OSError, IndexError, ValueError):
                continue
    except OSError:
        pass
    return out

def tree_cpu(root_pid: int) -> int:
    total = 0
    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += cpu_ticks(pid)
        stack.extend(children(pid))
    return total

cpu = 0
train_py = os.environ.get("TRAIN_PY_PID", "").strip()
train_sh = os.environ.get("TRAIN_PID", "").strip()
if train_py.isdigit():
    cpu = tree_cpu(int(train_py))
elif train_sh.isdigit():
    cpu = tree_cpu(int(train_sh))

print(f"{step} {cpu} {tb_bytes}")
PY
