#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# AUV tracking camera SAC (Direct-v4-cam) -- a copy of Direct-v0 with only:
#   * the target swapped from the r=0.35 m sphere to a second WarpAUV (motion model unchanged)
#   * the sonar swapped for the forward camera + K=4 RGB frame stack, 12 channels (KF from the
#     appearance/depth path, `kf_measurement: cam` -- the header said "grayscale / geometric GT"
#     until 2026-09-28; see SOURCE_PIN.md in the frozen checkpoint folder)
#
#   ./scripts/train_auv_tracking_sb3_sac_v4.sh --num_envs 8 --headless
#
# DELIBERATELY SIMPLE on dynamics, with ONE observation variable (camera vs belief) plus the
# stable-tracking costs Direct-v4-cam turns on (all absent from v0):
#   * ocean currents OFF. Do NOT add `env.ocean_current.enabled=true` here.
#   * target speed ALIGNED to v0: constant 0.5 m/s cruise / 0.6 ceiling, no accel limit
#   * NOT touched: cold-start 0-3 s + easy leash 3-5 s + free RRT.
#   * Reward (v4 only): the 800k snapshot's Holo-bounded shaping -- uncertainty 0.2 * -log det
#     Sigma, plus range 0.2 (sigma 0.6), bearing 0.3, action-delta 0.02, lost 0.50, all
#     `clamp(max=1.0)`. Restored 2026-09-29; see the reward note in auv_tracking_v4_cam_env.py.
#     The 2026-09-28/29 revision (uncertainty 0.02, unbounded costs 0.10/0.05, plus a linear
#     bearing tail and a yaw-saturation term) is NOT active -- its 16-env run peaked at ep_rew
#     -174 @200k and degraded to -506 @780k. To re-enable it:
#       env.rew_scale_uncertainty_mean=0.02 env.rew_penalty_clip=0.0 \
#       env.rew_penalty_range=0.10 env.rew_range_sigma=0.45 env.rew_penalty_bearing=0.05 \
#       env.rew_penalty_bearing_linear=0.02 env.rew_penalty_yaw_sat=0.02
#     KF measurement is the appearance/depth path ("cam"), not GT. World-velocity matching stays
#     disabled: on a surge+yaw vehicle it produced parallel, off-centre following.
#
# RAM note: SB3 keeps 2 copies of every observation (obs + next_obs) and ONE observation is
# 81936 B (uint8 RGB stack 49152 + float32 flow_image 32768 + belief 16), so the buffer costs
# buffer_size * 2 * 81936 B. `buffer_size` is the TOTAL transition count, NOT per env, so
# num_envs does not affect it: the 2026-09-28 baseline run's `agent.buffer_size=50000` is
# 7.63 GiB. Run `scripts/debug_auv_cam_spaces.sh` to print the live figure.
#
# The v0 defaults are kept for the knobs Direct-v3-cam turned on. To match that run, add:
#   --kf_measurement gt \
#   env.dyn_time_constant=0.2 env.thruster_first_order_lag=true \
#   env.rew_scale_speed=0.2 \
#   env.target_init_dist_min=1.0 env.target_init_dist_max=4.0

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/lib/ensure_single_isaac_process.sh"
ensure_single_isaac_process

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-AUV-Tracking-Direct-v4-cam \
  --agent sb3_cfg_entry_point \
  --algorithm SAC \
  --log_interval 20000 \
  --enable_cameras \
  "$@"
