#!/bin/bash
# SAC training for 1D lateral sonar IBVS-DRL (4-d centroid-error obs).
#
# Usage:
#   ./scripts/train_sonar_lateral1d_sac.sh --num_envs 4 --headless
#   ./scripts/train_sonar_lateral1d_dev.sh              # 50k stable dev run
#
# TensorBoard:
#   tensorboard --logdir logs/skrl/sonar_ibvs_drl

export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser
export TERM="${TERM:-xterm-256color}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/ensure_single_isaac_process.sh
source "${SCRIPT_DIR}/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process || exit 1

./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-SonarIbvsDrl-Lateral1D-Direct-v0 \
    --algorithm SAC \
    --enable_cameras \
    "$@"
