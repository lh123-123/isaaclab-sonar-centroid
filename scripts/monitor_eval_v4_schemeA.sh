#!/usr/bin/env bash
# Monitor v4-cam scheme-A run; when training finishes, auto-run 20-episode eval with best_model.
set -uo pipefail
cd /home/lh/IsaacLab
LOG_DIR="logs/sb3/Isaac-AUV-Tracking-Direct-v4-cam/2026-09-28_23-15-31/SAC_1"
EVAL_LOG="/tmp/auv_v4_schemeA_eval.log"

echo "[monitor] waiting for training to finish..." | tee "$EVAL_LOG"
while pgrep -f "watchdog_train_auv_tracking_sb3_sac_v4" > /dev/null; do
  sleep 120
done
sleep 60
if pgrep -f "watchdog_train_auv_tracking_sb3_sac_v4" > /dev/null; then
  echo "[monitor] watchdog still alive, continuing to wait" | tee -a "$EVAL_LOG"
  while pgrep -f "watchdog_train_auv_tracking_sb3_sac_v4" > /dev/null; do sleep 120; done
  sleep 60
fi

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
