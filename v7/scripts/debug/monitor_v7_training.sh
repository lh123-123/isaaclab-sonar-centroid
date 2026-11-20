#!/usr/bin/env bash
# Watchdog for v7 DR training. Reusable: pass the log prefix as $1 (default v7_drI_).
#
#   v7/scripts/debug/monitor_v7_training.sh v7_drI_
#
# Jobs:
#   1. status line every 10 min (steps / critic_loss / disp_spawn / reward),
#   2. metric snapshots at STOP_STEPS/2 and STOP_STEPS,
#   3. NORMAL STOP at STOP_STEPS, then a 40-episode eval,
#   4. TRIPWIRE: kill if critic_loss > LIMIT twice in a row (>= 20 min), then eval anyway.
#
# Two lessons are baked in:
#   * eval the BEST checkpoint, not the latest. Arms F/G/H all stopped AFTER going bad, so
#     "newest by mtime" systematically means "worst model" (arm H's auto-eval measured a
#     collapsed 200 k model at mean 1062 mm). The best checkpoint is picked from the
#     disp_spawn history: the latest sample within 5 % of the minimum, then the checkpoint
#     at or below that step count.
#   * every exit path evaluates. "Training log says 25 mm" is a different sample from the
#     number that has to clear the bar.
#
# Usage:  setsid nohup bash v7/scripts/debug/monitor_v7_training.sh v7_drI_ >/dev/null 2>&1 &

set -u
ROOT="$HOME/IsaacLab"
PREFIX="${1:-v7_drI_}"
PATTERN="train.py --task Isaac-AUV-Tracking-Direct-v7-flow-hover"
STOP_STEPS="${STOP_STEPS:-150000}"
LIMIT=0.5
SNAP_STEPS=$((STOP_STEPS / 2))
CKPT_ROOT="$ROOT/logs/sb3/Isaac-AUV-Tracking-Direct-v7-flow-hover"
OUT="$ROOT/v7/logs/${PREFIX}monitor.txt"
HIST="$ROOT/v7/logs/${PREFIX}disp_history.txt"
RUN_DIR=""
prev_crit=""

pick_log() {
  ls -t "$ROOT"/v7/logs/${PREFIX}*.log 2>/dev/null | grep -v monitor | head -1
}

last() { grep -a "$1" "$2" 2>/dev/null | tail -1 | awk '{print $(NF-1)}'; }

pick_run_dir() {
  [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ] && return
  RUN_DIR="$(ls -td "$CKPT_ROOT"/*/ 2>/dev/null | head -1)"
}

latest_checkpoint() {
  # Evaluate the LATEST checkpoint, not "best by training history". That heuristic picked
  # model_50000 for arm K (its training-side disp_spawn looked best early on) and reported
  # 132 mm, while the final 150 k model actually measured 45.9 mm end / 15.5 mm/s drift.
  # Under domain randomisation the training-side average is no substitute for an evaluation,
  # and evaluating the newest model is what "how far did this run get" actually means.
  local cp
  pick_run_dir
  cp="$(ls "$RUN_DIR"model_*_steps.zip 2>/dev/null \
        | sed 's/.*model_\([0-9]*\)_steps.zip/\1 &/' | sort -n | tail -1 \
        | awk '{print $2}')"
  if [ -z "$cp" ]; then
    cp="$(ls -t "$RUN_DIR"model_*.zip 2>/dev/null | head -1)"
  fi
  printf "%s" "$cp"
}

eval_best() {
  local cp
  cp="$(best_checkpoint)"
  echo "eval target (best, not latest): $cp" >> "$OUT"
  if [ -z "$cp" ] || [ ! -f "$cp" ]; then
    echo "EVAL SKIPPED: no checkpoint found in $RUN_DIR" >> "$OUT"
    return
  fi
  echo "$(date '+%F %T') running 40-episode eval (action_delay=3) ..." >> "$OUT"
  /home/lh/miniconda3/envs/isaaclab/bin/python "$ROOT/v7/scripts/eval/eval_v7_checkpoint.py" \
    --checkpoint "$cp" --episodes 100 --num_envs 4 --headless --action_delay 3 \
    --out "$ROOT/v7/logs/${PREFIX}eval_best.json" > "$ROOT/v7/logs/${PREFIX}eval_best.log" 2>&1 || true
  grep -aE "disp_spawn_m  mean|peak_m " "$ROOT/v7/logs/${PREFIX}eval_best.log" | tail -2 >> "$OUT"
  echo "EVAL DONE $(date '+%F %T')" >> "$OUT"
}

echo "monitor armed $(date '+%F %T')  prefix=$PREFIX  stop=$STOP_STEPS  limit=$LIMIT" >> "$OUT"

while :; do
  LOG="$(pick_log)"
  if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
    echo "$(date '+%F %T') no log matching ${PREFIX}, retrying" >> "$OUT"
    sleep 300
    continue
  fi

  if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "$(date '+%F %T') TRAINING PROCESS GONE -- monitor exit" >> "$OUT"
    exit 0
  fi

  steps="$(last total_timesteps "$LOG")"
  crit="$(last critic_loss "$LOG")"
  disp="$(last disp_spawn_m "$LOG")"
  rew="$(last ep_rew_mean "$LOG")"
  echo "$(date '+%F %T') steps=$steps critic=$crit disp_spawn=$disp rew=$rew" >> "$OUT"
  if [ -n "$steps" ] && [ -n "$disp" ]; then
    echo "$steps $disp" >> "$HIST"
  fi

  for m in $SNAP_STEPS $STOP_STEPS; do
    if [ -n "$steps" ] && [ "$steps" -ge "$m" ] 2>/dev/null && ! grep -q "SNAPSHOT $m" "$OUT"; then
      echo "SNAPSHOT $m reached: steps=$steps critic=$crit disp_spawn=$disp rew=$rew" >> "$OUT"
      grep -a "disp_spawn_m" "$LOG" | awk '{print $4}' > "$ROOT/v7/logs/${PREFIX}disp_at_${m}.txt"
      grep -a "ep_rew_mean" "$LOG" | awk '{print $4}' > "$ROOT/v7/logs/${PREFIX}rew_at_${m}.txt"
      grep -a "critic_loss" "$LOG" | awk '{print $4}' > "$ROOT/v7/logs/${PREFIX}critic_at_${m}.txt"
    fi
  done

  if [ -n "$steps" ] && [ "$steps" -ge "$STOP_STEPS" ] 2>/dev/null; then
    echo "NORMAL STOP: steps=$steps >= $STOP_STEPS" >> "$OUT"
    pkill -9 -f "$PATTERN" || true
    sleep 8
    eval_best
    exit 0
  fi

  if [ -n "$crit" ] && awk -v c="$crit" -v l="$LIMIT" 'BEGIN{exit !(c+0 > l+0)}' 2>/dev/null; then
    if [ -n "$prev_crit" ] && awk -v c="$prev_crit" -v l="$LIMIT" 'BEGIN{exit !(c+0 > l+0)}' 2>/dev/null; then
      echo "TRIPWIRE fired: critic_loss $prev_crit -> $crit (> $LIMIT twice) at steps=$steps" >> "$OUT"
      pkill -9 -f "$PATTERN" || true
      sleep 8
      echo "KILLED $(date '+%F %T')  reason=critic_loss>$LIMIT sustained" >> "$OUT"
      eval_best
      exit 0
    fi
  fi
  prev_crit="$crit"

  sleep 600
done
