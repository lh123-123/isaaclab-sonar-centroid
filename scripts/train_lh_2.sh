#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# LH-2 水下目标追踪 — 便捷启动脚本（配置 OceanSim 扩展路径并调用 RSL-RL 训练）
#
# 回放策略请使用对称脚本（相同 ISAACSIM_EXTENSIONS_FOLDER）:
#   ./scripts/play_lh_2.sh --num_envs 1 [--checkpoint path/to/model_xxx.pt]
#
# 使用:
#   ./scripts/train_lh_2.sh [选项]
#   建议在已激活的 isaaclab conda 环境中执行；若用仓库包装器:
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lh-2-Direct-v0 --enable_cameras "$@"
#
# 常用追加选项（经 "$@" 传入 train.py）:
#   --num_envs 4
#   --device cuda:0
#   --max_iterations 10000
#   --video
#   --headless
#
# 若 OceanSim 不在默认路径，请先修改下方 ISAACSIM_EXTENSIONS_FOLDER。

# OceanSim 所在 extsUser（按本机安装路径修改）
export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

# Lh2 + OceanSim 默认启用声纳 UI 视口管线（可在命令行追加 --no-sonar_ui 覆盖）。
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lh-2-Direct-v0 \
    --enable_cameras \
    --sonar_ui \
    "$@"
