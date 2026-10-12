#!/usr/bin/env bash
# Monitor v4-cam 0929 retrain; when training finishes, auto-run 20-episode eval.
set -uo pipefail
cd /home/lh/IsaacLab
LOG_ROOT="logs/sb3/Isaac-AUV-Tracking-Direct-v4-cam"
EVAL_LOG="/tmp/auv_v4_schemeA_eval.log"

echo "[monitor] waiting for training to finish..." | tee "$EVAL_LOG"
while pgrep -f "watchdog_train_auv_tracking_sb3_sac_v4" > /dev/null; do
  sleep 120
done
# grace: make sure the train process really exited (not a poll gap)
sleep 60
if pgrep -f "watchdog_train_auv_tracking_sb3_sac_v4" > /dev/null; then
  echo "[monitor] watchdog still alive, continuing to wait" | tee -a "$EVAL_LOG"
  while pgrep -f "watchdog_train_auv_tracking_sb3_sac_v4" > /dev/null; do sleep 120; done
  sleep 60
fi

# Newest run dir (scheme-A run has a fresh timestamp); prefer best_model.zip there.
LOG_DIR=$(ls -td "$LOG_ROOT"/*/SAC_1 2>/dev/null | head -1)
echo "[monitor] training done at $(date), logdir=$LOG_DIR" | tee -a "$EVAL_LOG"
CKPT=""
for cand in "$LOG_DIR/best_model.zip" $(ls -t "$LOG_DIR"/model*.zip "$LOG_DIR"/checkpoints/*.zip 2>/dev/null); do
  if [ -f "$cand" ]; then CKPT="$cand"; break; fi
done
echo "[monitor] training done at $(date), ckpt=$CKPT" | tee -a "$EVAL_LOG"
if [ -z "${CKPT:-}" ]; then
  echo "[monitor] ERROR: no checkpoint found, aborting eval" | tee -a "$EVAL_LOG"
  exit 1
fi

./scripts/eval_auv_tracking_v4_cam.sh --checkpoint "$CKPT" --episodes 20 --headless >> "$EVAL_LOG" 2>&1
echo "[monitor] eval done at $(date)" | tee -a "$EVAL_LOG"
