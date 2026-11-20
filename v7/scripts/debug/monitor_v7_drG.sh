#!/usr/bin/env bash
# Watchdog for the v7 DR training run (arm H, restarted 2026-11-18 ~21:30).
#
# Four jobs:
#   1. append a status line every 10 min (steps / critic_loss / disp_spawn / reward),
#   2. snapshot the full metric series at the 100k / 200k milestones,
#   3. STOP at STOP_STEPS (user bar reached: 5 cm) and certify with a 40-episode eval,
#   4. TRIPWIRE: kill the run if critic_loss stays above 0.5 for two consecutive samples
#      (>= 20 min), then certify anyway. A single spike is noise; two in a row is the
#      collapse signature that already cost arm F ~10 h and arm G its best checkpoint.
#
# Both exits run the same eval: "training log says 37 mm" is a different sample than the
# number that has to clear the bar, so every stop must produce a measured number.
#
# Usage:  setsid nohup v7/scripts/debug/monitor_v7_drG.sh >/dev/null 2>&1 &

set -u
ROOT="$HOME/IsaacLab"
PATTERN="train.py --task Isaac-AUV-Tracking-Direct-v7-flow-hover"
MILESTONES="100000 200000"
LIMIT=0.5
STOP_STEPS=200000
LOG_GLOB="v7_drH_"
OUT="$ROOT/v7/logs/v7_drH_monitor.txt"
prev_crit=""

pick_log() {
  ls -t "$ROOT"/v7/logs/${LOG_GLOB}*.log 2>/dev/null | grep -v monitor | head -1
}

last() { grep -a "$1" "$2" 2>/dev/null | tail -1 | awk '{print $(NF-1)}'; }

eval_latest() {
  local cp
  cp="$(ls -t "$ROOT"/logs/sb3/Isaac-AUV-Tracking-Direct-v7-flow-hover/*/model_*.zip 2>/dev/null | head -1)"
  echo "eval checkpoint: $cp" >> "$OUT"
  if [ -z "$cp" ]; then
    echo "EVAL SKIPPED: no checkpoint found" >> "$OUT"
    return
  fi
  echo "$(date '+%F %T') running 40-episode eval (action_delay=3) ..." >> "$OUT"
  /home/lh/miniconda3/envs/isaaclab/bin/python "$ROOT/v7/scripts/eval/eval_v7_checkpoint.py" \
    --checkpoint "$cp" --episodes 40 --num_envs 4 --headless --action_delay 3 \
    --out "$ROOT/v7/logs/eval_stop.json" > "$ROOT/v7/logs/v7_drH_eval_stop.log" 2>&1 || true
  grep -aE "disp_spawn_m  mean|peak_m " "$ROOT/v7/logs/v7_drH_eval_stop.log" | tail -2 >> "$OUT"
  echo "EVAL DONE $(date '+%F %T')" >> "$OUT"
}

echo "monitor armed $(date '+%F %T')  limit=$LIMIT  stop=$STOP_STEPS  milestones=$MILESTONES" >> "$OUT"

while :; do
  LOG="$(pick_log)"
  if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
    echo "$(date '+%F %T') no log matching ${LOG_GLOB}, retrying" >> "$OUT"
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

  for m in $MILESTONES; do
    if [ -n "$steps" ] && [ "$steps" -ge "$m" ] 2>/dev/null && ! grep -q "MILESTONE $m" "$OUT"; then
      echo "MILESTONE $m reached: steps=$steps critic=$crit disp_spawn=$disp rew=$rew" >> "$OUT"
      grep -a "disp_spawn_m" "$LOG" | awk '{print $4}' > "$ROOT/v7/logs/${LOG_GLOB}disp_at_${m}.txt"
      grep -a "ep_rew_mean" "$LOG" | awk '{print $4}' > "$ROOT/v7/logs/${LOG_GLOB}rew_at_${m}.txt"
      grep -a "critic_loss" "$LOG" | awk '{print $4}' > "$ROOT/v7/logs/${LOG_GLOB}critic_at_${m}.txt"
    fi
  done

  # 正常收尾：到点就停，并自动跑正式 40 回合评估 —— 判据要 certify，不能靠训练日志
  if [ -n "$steps" ] && [ "$steps" -ge "$STOP_STEPS" ] 2>/dev/null; then
    echo "NORMAL STOP: steps=$steps >= $STOP_STEPS  (user bar: 5 cm)" >> "$OUT"
    pkill -9 -f "$PATTERN" || true
    sleep 8
    eval_latest
    exit 0
  fi

  if [ -n "$crit" ] && awk -v c="$crit" -v l="$LIMIT" 'BEGIN{exit !(c+0 > l+0)}' 2>/dev/null; then
    if [ -n "$prev_crit" ] && awk -v c="$prev_crit" -v l="$LIMIT" 'BEGIN{exit !(c+0 > l+0)}' 2>/dev/null; then
      echo "TRIPWIRE fired: critic_loss $prev_crit -> $crit (> $LIMIT twice) at steps=$steps" >> "$OUT"
      pkill -9 -f "$PATTERN" || true
      sleep 8
      echo "KILLED $(date '+%F %T')  reason=critic_loss>$LIMIT sustained" >> "$OUT"
      eval_latest
      exit 0
    fi
  fi
  prev_crit="$crit"

  sleep 600
done
