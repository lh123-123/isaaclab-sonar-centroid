#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# 从 minimal_v1 checkpoint 续训（skrl 会新建带时间戳的 log 目录，权重从 checkpoint 加载）
#
# 默认：minimal_v2_facing 的 agent_18000.pt 再训 10000 步
#   ./scripts/train_nonar2_sac_resume.sh --num_envs 1 --headless
#
# 自定义 checkpoint / 步数:
#   ./scripts/train_nonar2_sac_resume.sh --checkpoint path/to/agent_16000.pt --max_iterations 15000

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

# 续训请用 --checkpoint 指定；未指定时用上一轮 minimal_v1 最佳权重
DEFAULT_CKPT="logs/skrl/nonar2_belief6_sac/2026-05-26_17-06-25_sac_torch_minimal_v1/checkpoints/agent_18000.pt"

./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Nonar2-Belief6-Direct-v0 \
    --algorithm SAC \
    --enable_cameras \
    --checkpoint "$DEFAULT_CKPT" \
    --max_iterations 10000 \
    "$@"
