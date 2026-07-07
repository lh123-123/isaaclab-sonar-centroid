#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# 在 Test1 定点到达任务上训练 PPO 残差策略（SAC Q 值易爆炸，Test1 用 PPO）
#
# RTX 5060 8GB 推荐:
#   ./scripts/train_residual_rl_test1.sh --num_envs 128 --headless
#
# TensorBoard:
#   tensorboard --logdir logs/skrl/residual_rl_test1_ppo_v3.2_wrench/
# 训练完成后跑 Test1 评估:
#   ./scripts/test1_residual_rl.sh --headless --checkpoint logs/skrl/residual_rl_test1_ppo_v3.2_wrench/<run>/checkpoints/agent_500000.pt

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-ResidualRL-Test1-Train-v0 \
    --algorithm PPO \
    "$@"
