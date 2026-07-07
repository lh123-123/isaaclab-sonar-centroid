#!/bin/bash
# Dev SAC training: stable hyperparams, 50k steps (~1.5h @ 10 step/s).
#
# Usage:
#   ./scripts/train_sonar_lateral1d_dev.sh
#   ./scripts/train_sonar_lateral1d_dev.sh --num_envs 4 --headless
#
# TensorBoard:
#   tensorboard --logdir logs/skrl/sonar_ibvs_drl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_sonar_lateral1d_sac.sh" \
    --num_envs 4 \
    --headless \
    --max_iterations 50000 \
    "$@"
