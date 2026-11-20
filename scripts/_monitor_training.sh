#!/bin/bash
# Monitor training progress every 30 minutes and append to log
LOG=~/train_monitor.log
EXPER_DIR=~/IsaacLab/logs/sb3/Isaac-AUV-Tracking-Direct-v7-flow-hover/2026-11-08_20-29-10
MAIN_LOG=~/train_v7_2m.log

while true; do
    sleep 1800  # 30 minutes
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    timesteps=$(grep "total_timesteps" $MAIN_LOG 2>/dev/null | tail -1 | grep -oE '[0-9]+$' || echo "?")
    ep_rew=$(grep "ep_rew_mean" $MAIN_LOG 2>/dev/null | tail -1 | grep -oE '[0-9.]+$' || echo "?")
    fps=$(grep "fps" $MAIN_LOG 2>/dev/null | tail -1 | grep -oE '[0-9]+$' || echo "?")
    elapsed=$(grep "time_elapsed" $MAIN_LOG 2>/dev/null | tail -1 | grep -oE '[0-9]+$' || echo "?")
    gpumem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo "?")
    ckpt=$(ls $EXPER_DIR/model_*.zip 2>/dev/null | wc -l)
    echo "$ts | timesteps=$timesteps/2000000 | ep_rew=$ep_rew | fps=$fps | elapsed=${elapsed}s | gpu=$gpumem | ckpts=$ckpt" >> $LOG
done
