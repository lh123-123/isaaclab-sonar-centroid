#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Residual RL (WarpAUV + imperfect PD + SAC residual)
#
# 训练 (阶段1, 无扰动):
#   ./scripts/train_residual_rl_sac.sh --num_envs 512 --headless
#
# 课程阶段:
#   ./scripts/train_residual_rl_sac.sh --task Isaac-ResidualRL-Stage2-Direct-v0 --num_envs 512 --headless
#   ./scripts/train_residual_rl_sac.sh --task Isaac-ResidualRL-Stage3-Direct-v0 --num_envs 512 --headless
#   ./scripts/train_residual_rl_sac.sh --task Isaac-ResidualRL-Stage4-Direct-v0 --num_envs 512 --headless
#
# TensorBoard:
#   tensorboard --logdir logs/skrl/residual_rl_sac/

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-ResidualRL-Direct-v0 \
    --algorithm SAC \
    "$@"
