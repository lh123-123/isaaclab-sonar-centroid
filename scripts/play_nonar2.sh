#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Nonar2 Belief6 + SAC 回放
#
# 使用:
#   ./scripts/play_nonar2.sh                     # GUI + sonar_ui
#   ./scripts/play_nonar2.sh --headless          # headless + 评估
#   ./scripts/play_nonar2.sh --checkpoint <path>
#
# 默认加载 v22 best_agent.pt（训练出新模型后请更新此路径）。
# 环境变量 NONAR2_PLAY_CHECKPOINT 可覆盖。

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

# 默认 checkpoint（v25 stable，训练 v26 后请更新）
# 可用环境变量 NONAR2_PLAY_CHECKPOINT 临时覆盖
DEFAULT_CHECKPOINT="${NONAR2_PLAY_CHECKPOINT:-logs/skrl/nonar2_belief6_sac/2026-06-01_21-42-48_sac_torch_minimal_v25_stable/checkpoints/best_agent.pt}"
PLAY_ARGS=("$@")

echo "[Nonar2 play] Using checkpoint: $DEFAULT_CHECKPOINT" >&2
echo "[Nonar2 play] Add --headless to run headless" >&2

# 自动添加 eval_debug
if [[ " ${PLAY_ARGS[*]} " != *" --eval_debug "* ]]; then
    PLAY_ARGS=(--eval_debug "${PLAY_ARGS[@]}")
fi

# 自动添加 checkpoint
if [[ " ${PLAY_ARGS[*]} " != *" --checkpoint "* ]]; then
    PLAY_ARGS=(--checkpoint "$DEFAULT_CHECKPOINT" "${PLAY_ARGS[@]}")
fi

# 判断 headless
IS_HEADLESS=false
for arg in "${PLAY_ARGS[@]}"; do
    if [ "$arg" = "--headless" ]; then
        IS_HEADLESS=true
        break
    fi
done

if $IS_HEADLESS; then
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
        --task Isaac-Nonar2-Belief6-Direct-v0 \
        --algorithm SAC \
        --enable_cameras \
        "${PLAY_ARGS[@]}"
else
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
        --task Isaac-Nonar2-Belief6-Direct-v0 \
        --algorithm SAC \
        --enable_cameras \
        --sonar_ui \
        "${PLAY_ARGS[@]}"
fi
