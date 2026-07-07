#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Nonar2 前视声纳 1-DoF 跟踪 — Phase0 结构化观测 + RSL-RL PPO（OceanSim）
#
# 回放策略请使用对称脚本（相同 ISAACSIM_EXTENSIONS_FOLDER）:
#   ./scripts/play_nonar2.sh --task Isaac-Nonar2-Direct-v0 --algorithm PPO ...
#   或 RSL-RL: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#       --task Isaac-Nonar2-Direct-v0 --enable_cameras "$@"
#
# 使用:
#   ./scripts/train_nonar2.sh [选项]
#
# 常用追加选项（经 "$@" 传入 train.py）:
#   --num_envs 4
#   --device cuda:0
#   --max_iterations 10000
#   --headless
#   --no-sonar_ui          # headless 时关闭 OceanSim 视口声纳
#   --video
#
# SAC + 信念 CNN 训练请用: ./scripts/train_nonar2_sac.sh
#
# 若 OceanSim 不在默认路径，请先修改下方 ISAACSIM_EXTENSIONS_FOLDER。

# OceanSim 所在 extsUser（与 scripts/train_lh_2.sh 保持一致，按本机安装路径修改）
export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

# GUI 调试默认启用声纳视口；headless 请在命令行追加 --no-sonar_ui
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Nonar2-Direct-v0 \
    --enable_cameras \
    --sonar_ui \
    "$@"
