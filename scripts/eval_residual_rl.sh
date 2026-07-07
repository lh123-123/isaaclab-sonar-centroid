#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ResidualRL 批量评估: PD-only vs SAC 残差
#
# 使用:
#   ./scripts/eval_residual_rl.sh --headless --episodes 20
#   ./scripts/eval_residual_rl.sh --headless --stage 3 --checkpoint logs/skrl/.../best_agent.pt
#   ./scripts/eval_residual_rl.sh --headless --episodes 50 --stage 4 --seed 42
#
# 结果 CSV 默认写入 logs/residual_rl_eval/

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/residual_rl/custom_workflows/evaluate.py \
    "$@"
