#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# 训练过程中每 100k 暂停 → Test1 eval → 输出是否继续 → 可选自动续训
#
# 用法:
#   ./scripts/watch_test1_milestones.sh [run_dir] [interval] [max_step]
#
# 例（另开终端，与 train 并行跑）:
#   ./scripts/watch_test1_milestones.sh \
#     logs/skrl/residual_rl_test1_ppo_v3.2_wrench/2026-06-09_19-28-04_ppo_torch \
#     100000 500000
#
# 环境变量:
#   AUTO_RESUME=1  评估后若 verdict=CONTINUE/CAUTION 则自动 --checkpoint 续训

set -euo pipefail

RUN_DIR="${1:-}"
INTERVAL="${2:-100000}"
MAX_STEP="${3:-500000}"
AUTO_RESUME="${AUTO_RESUME:-0}"
POLL_SEC="${POLL_SEC:-60}"

if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="$(ls -td logs/skrl/residual_rl_test1_ppo_v3.2_wrench/*/ 2>/dev/null | head -1)"
fi
RUN_DIR="${RUN_DIR%/}"
CKPT_DIR="$RUN_DIR/checkpoints"
TRAIN_LOG="${TRAIN_LOG:-logs/skrl/v3.2_train.log}"

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "[watch] checkpoint dir not found: $CKPT_DIR"
    exit 1
fi

echo "[watch] run_dir=$RUN_DIR interval=${INTERVAL} max=${MAX_STEP} auto_resume=$AUTO_RESUME"

stop_train() {
    if pgrep -f "train.py.*Test1-Train" >/dev/null; then
        echo "[watch] stopping training..."
        pkill -f "train.py.*Test1-Train" || true
        sleep 8
    fi
}

start_train() {
    local ckpt="$1"
    echo "[watch] resuming training from $ckpt"
    nohup ./scripts/train_residual_rl_test1.sh --num_envs 128 --headless \
        --checkpoint "$ckpt" >> "$TRAIN_LOG" 2>&1 &
    echo "[watch] train PID=$!"
}

DONE_FILE="logs/residual_rl_eval/milestones_done_${RUN_DIR##*/}.txt"
mkdir -p logs/residual_rl_eval
touch "$DONE_FILE"

for (( step=INTERVAL; step<=MAX_STEP; step+=INTERVAL )); do
    ckpt="$CKPT_DIR/agent_${step}.pt"
    if grep -qx "$step" "$DONE_FILE" 2>/dev/null; then
        continue
    fi

    echo "[watch] waiting for $(basename "$ckpt") ..."
    while [[ ! -f "$ckpt" ]]; do
        if ! pgrep -f "train.py.*Test1-Train" >/dev/null; then
            prog="$(grep -oP '\d+/500000' "$TRAIN_LOG" 2>/dev/null | tail -1 || true)"
            cur="${prog%%/*}"
            if [[ -n "${cur:-}" && "$cur" -ge "$step" ]] 2>/dev/null; then
                echo "[watch] train stopped but $ckpt missing (progress=$prog)"
                break
            fi
            echo "[watch] train not running, progress=${prog:-unknown}; waiting for manual restart..."
        fi
        sleep "$POLL_SEC"
    done

    if [[ ! -f "$ckpt" ]]; then
        echo "[watch] skip step $step — checkpoint not found"
        continue
    fi

    stop_train
    ./scripts/eval_test1_milestone.sh "$ckpt" "$step" "$RUN_DIR" | tee -a "logs/residual_rl_eval/milestone_${RUN_DIR##*/}_${step}.log"
    echo "$step" >> "$DONE_FILE"

    verdict="$(tail -20 "logs/residual_rl_eval/milestones_report.jsonl" | python3 -c "import sys,json; print(json.loads(sys.stdin.readlines()[-1])['verdict'])")"
    echo "[watch] verdict@$step = $verdict"

    if [[ "$verdict" == "STOP" || "$verdict" == "TARGET_MET" ]]; then
        echo "[watch] done — no auto resume ($verdict)"
        exit 0
    fi

    if [[ "$step" -ge "$MAX_STEP" ]]; then
        echo "[watch] reached max_step $MAX_STEP"
        exit 0
    fi

    if [[ "$AUTO_RESUME" == "1" ]]; then
        start_train "$ckpt"
    else
        echo "[watch] 续训命令:"
        echo "  ./scripts/train_residual_rl_test1.sh --num_envs 128 --headless --checkpoint $ckpt"
    fi
done
