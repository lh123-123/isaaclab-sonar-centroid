#!/usr/bin/env bash
# AUV Phase A1 baseline: decoupled PD + position+lead (GT oracle) + station hold.
# Lateral: ff70_l065 |d_col|≈1.59 px (kp_range=0). Distance: ff70_r12_h25 kp_range=1.2 hold=2.5.
set -euo pipefail
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"
./scripts/kill_isaac_lab_processes.sh 2>/dev/null || true
sleep 2

./isaaclab.sh -p scripts/eval_centroid_track_p_control_auv.py \
  --task Isaac-SonarCentroidTrack-AUV-PControl-Direct-v0 \
  --auv_control decoupled \
  --control position_lead \
  --target_motion sine \
  --sway_amp 1.3 \
  --target_vy 0.28 \
  --kp_sway 6.5 \
  --lead_s 0.65 \
  --kp_vy 0.6 \
  --kp_range 1.2 \
  --kp_surge_hold 2.5 \
  --max_surge 1.0 \
  --auv_pos_kp 14 \
  --auv_pos_kd 3.5 \
  --auv_vy_ff 70 \
  --auv_max_wrench 200 \
  --auv_max_force 300 \
  --output_dir logs/centroid_track/auv_decoupled_poslead \
  --headless \
  "$@"
