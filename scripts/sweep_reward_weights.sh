#!/bin/bash
# Reward weight sweep for Direct-v2.
# Trains each config for SWEEP_STEPS, evals each, saves results.
#
#   ./scripts/sweep_reward_weights.sh
#
# No babysit needed — runs sequentially, auto-kills on hang.

set -uo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

TASK="Isaac-AUV-Tracking-Direct-v2"
LOG_DIR="logs/sb3/${TASK}"
SWEEP_STEPS=500000
EVAL_EPISODES=24
NUM_ENVS=16
STALL_TIMEOUT=18000  # 300 min max per training run (500K steps at 42 it/s ~200 min + init)

# Configs: "label|rew_speed|rew_ref|rew_ref_sigma"
CONFIGS=(
  "speed010_ref010|0.1|0.1|0.5"
  "speed020_ref010|0.2|0.1|0.5"
  "speed010_ref020|0.1|0.2|0.5"
  "speed020_ref020|0.2|0.2|0.5"
  "speed010_ref000|0.1|0.0|0.5"
  "speed000_ref010|0.0|0.1|0.5"
)

mkdir -p /tmp/sweep_results

echo "[sweep] === reward weight sweep started $(date) ==="
echo "[sweep] ${#CONFIGS[@]} configs, ${SWEEP_STEPS} steps each"

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r label rew_speed rew_ref rew_sigma <<< "$cfg"
  echo "[sweep] ========================================"
  echo "[sweep] config: ${label} (speed=${rew_speed} ref=${rew_ref} sigma=${rew_sigma})"

  ensure_single_isaac_process

  # Train
  echo "[sweep] training ${label}..."
  timeout ${STALL_TIMEOUT} ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
    --task ${TASK} \
    --agent sb3_cfg_entry_point \
    --algorithm SAC \
    --log_interval 20000 \
    --enable_cameras \
    --num_envs ${NUM_ENVS} \
    --headless \
    --max_iterations ${SWEEP_STEPS} \
    --rew_speed ${rew_speed} \
    --rew_ref ${rew_ref} \
    --rew_ref_sigma ${rew_sigma} \
    > /tmp/sweep_train_${label}.log 2>&1
  train_rc=$?

  if [[ ${train_rc} -ne 0 ]]; then
    echo "[sweep] WARNING: training ${label} exited with rc=${train_rc}"
  fi

  # Find latest checkpoint
  CKPT=$(find ${LOG_DIR} -name "model_*_steps.zip" -newest -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
  if [[ -z "${CKPT}" ]]; then
    # Try finding any checkpoint from the most recent run
    CKPT=$(find ${LOG_DIR} -name "model_*_steps.zip" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
  fi

  if [[ -z "${CKPT}" ]]; then
    echo "[sweep] ERROR: no checkpoint found for ${label}, skipping eval"
    continue
  fi

  echo "[sweep] evaluating ${label} from ${CKPT}..."
  ./scripts/kill_isaac_lab_processes.sh 2>/dev/null
  sleep 3

  timeout 600 ./isaaclab.sh -p scripts/play_auv_tracking_flow_v2.py \
    --checkpoint ${CKPT} \
    --num_envs 1 \
    --episodes 12 \
    --headless \
    --tag ${label} \
    --dump /tmp/sweep_eval_${label}.npz \
    > /tmp/sweep_eval_${label}.log 2>&1
  eval_rc=$?

  if [[ ${eval_rc} -ne 0 ]]; then
    echo "[sweep] WARNING: eval ${label} exited with rc=${eval_rc}"
  fi

  ./scripts/kill_isaac_lab_processes.sh 2>/dev/null
  sleep 3

  echo "[sweep] done ${label}"
done

echo "[sweep] === sweep complete $(date) ==="
echo "[sweep] results in /tmp/sweep_eval_*.npz and /tmp/sweep_eval_*.log"
