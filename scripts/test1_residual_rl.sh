#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Test 1 — 定点到达: (0,0,8) → (5,0,8)，对比 PD-only vs SAC+PD
#
# 使用:
#   ./scripts/test1_residual_rl.sh --headless
#   ./scripts/test1_residual_rl.sh --headless --episodes 20 --seed 42 --print_interval 60 --step_trace \
#     --checkpoint logs/skrl/residual_rl_test1_ppo_v2/<run>/checkpoints/agent_500000.pt
#
# 训练 Test1 任务 (SAC):
#   ./scripts/train_residual_rl_test1.sh --num_envs 512 --headless
#
# 输出: logs/residual_rl_eval/test1_<timestamp>.csv / .json

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/residual_rl/custom_workflows/test1_point_reach.py \
    "$@"
