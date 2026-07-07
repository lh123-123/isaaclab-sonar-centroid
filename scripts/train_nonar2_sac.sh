#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Nonar2 Belief6 + SAC（minimal_v40_gt_obs_v20params）
#   · GT obs + v20 proven params: gamma=0.80, lr=8e-6, shaper=0.05
#   · 纯偏差惩罚: -5×|det_az_deg|/30, miss=-5（丢失同最大偏差）
#   · 56k steps
#
# 训练:
#   ./scripts/train_nonar2_sac.sh --num_envs 1 --headless
#
# Module B 训练期 debug（与 play --sonar_debug 相同输出，PNG 写入 run/sonar_debug/）:
#   ./scripts/train_nonar2_sac.sh --num_envs 1 --headless --sonar_debug --max_iterations 500
#   ./scripts/train_nonar2_sac.sh --num_envs 1 --headless --sonar_debug --sonar_debug_save \
#       --sonar_debug_interval 20 --max_iterations 500
#
# 回放（训练结束后）:
#   ./scripts/play_nonar2.sh --num_envs 1 --sonar_ui
#
# TensorBoard:
#   tensorboard --logdir logs/skrl/nonar2_belief6_sac/
#   成功: episode mean 稳定 >140，Q mean < ~1；勿用 critic 爆炸后的权重
#
# 续训见: ./scripts/train_nonar2_sac_resume.sh

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Nonar2-Belief6-Direct-v0 \
    --algorithm SAC \
    --enable_cameras \
    "$@"
