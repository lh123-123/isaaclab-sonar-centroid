#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ResidualRL 回放 (SAC + PD 残差)
#
# 使用:
#   ./scripts/play_residual_rl.sh                          # GUI 回放
#   ./scripts/play_residual_rl.sh --headless --eval_debug  # headless + 指标
#   ./scripts/play_residual_rl.sh --checkpoint <path>
#
# 环境变量 RESIDUAL_RL_PLAY_CHECKPOINT 可覆盖默认 checkpoint。

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

DEFAULT_CHECKPOINT="${RESIDUAL_RL_PLAY_CHECKPOINT:-}"
PLAY_ARGS=("$@")

# 自动添加 eval_debug（便于单 episode 指标输出）
if [[ " ${PLAY_ARGS[*]} " != *" --eval_debug "* ]]; then
    PLAY_ARGS=(--eval_debug "${PLAY_ARGS[@]}")
fi

# 若设置了默认 checkpoint 且 CLI 未指定，则注入
if [[ -n "$DEFAULT_CHECKPOINT" && " ${PLAY_ARGS[*]} " != *" --checkpoint "* ]]; then
    PLAY_ARGS=(--checkpoint "$DEFAULT_CHECKPOINT" "${PLAY_ARGS[@]}")
fi

./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
    --task Isaac-ResidualRL-Direct-v0 \
    --algorithm SAC \
    "${PLAY_ARGS[@]}"
