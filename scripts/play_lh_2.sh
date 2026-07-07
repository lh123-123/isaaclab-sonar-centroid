#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# LH-2 水下目标追踪 — 与 train_lh_2.sh 使用相同的 OceanSim 扩展路径，调用官方 RSL-RL play。
#
# 使用:
#   ./scripts/play_lh_2.sh [选项]
#  或（与 README 一致）:
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Lh-2-Direct-v0 --enable_cameras "$@"
#
# 常用追加选项（经 "$@" 传入 play.py）:
#   --num_envs 1
#   --device cuda:0
#   --checkpoint /path/to/model_500.pt
#   --real-time
#
# 注意: 请保持下方路径与 scripts/train_lh_2.sh 中一致。

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Lh-2-Direct-v0 \
    --enable_cameras \
    --sonar_ui \
    "$@"
