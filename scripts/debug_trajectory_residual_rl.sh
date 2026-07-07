#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Test1 单 episode 轨迹 debug（默认 PD-only，零残差）
#
#   ./scripts/debug_trajectory_residual_rl.sh --headless
#   ./scripts/debug_trajectory_residual_rl.sh --headless --interval 5 --steps 800
#   ./scripts/debug_trajectory_residual_rl.sh --checkpoint logs/skrl/residual_rl_test1_sac/<run>/checkpoints/best_agent.pt

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/residual_rl/custom_workflows/debug_trajectory.py \
    "$@"
