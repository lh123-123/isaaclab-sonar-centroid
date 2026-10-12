#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# AUV tracking GT-optical-flow SAC (Direct-v5-flow) -- Direct-v4-cam with ONE change:
#   * the actor's observation is a K=4 stack of GT optical-flow frames ((frame_stack * 2, 64, 64)
#     float32, 8 channels) instead of the uint8 RGB stack (12 channels) plus a separate 2-channel
#     "flow_image" key (14 channels total). The shared MidCNNExtractor needed no code change: it
#     reads observations["image"].shape[0] and skips its rescale guard for float32 in [-1, 1].
#
# Everything else is Direct-v4-cam verbatim: the second-WarpAUV target with its unchanged
# BatchHoloTargetMotion, the camera and the camera-based KF measurement (kf_measurement: cam),
# the 800k reward shape, the reset schedule and the control chain.
#
#   ./scripts/train_auv_tracking_sb3_sac_v5_flow.sh --num_envs 8 --headless
#
# DELIBERATELY SIMPLE on dynamics, with ONE observation variable (flow stack vs belief) plus the
# stable-tracking costs Direct-v4-cam turns on (all absent from v0):
#   * ocean currents OFF. Do NOT add `env.ocean_current.enabled=true` here.
#   * target speed ALIGNED to v0: constant 0.5 m/s cruise / 0.6 ceiling, no accel limit
#   * NOT touched: cold-start 0-3 s + easy leash 3-5 s + free RRT.
#   * Reward (2026-10-09 first-principles revision, see the env cfg): alive 1.0
#     + bearing 1.1 * exp(-(a/15deg)^2) + size-ratio 1.0 * exp(-(log s / 0.5)^2)
#     - action-delta 0.03. The 800k-era shaping (uncertainty/range/bearing/lost with
#     clip=1.0) is gone; termination (OOV streak / envelope / timeout) replaces the
#     lost penalty. No one-time OOV terminal penalty.
#   * KF measurement is the appearance/depth path ("cam"), not GT. World-velocity matching stays
#     disabled: on a surge+yaw vehicle it produced parallel, off-centre following.
#
# The camera is NOT part of the actor's input any more, but it is still rendered and still read
# every step by `_update_belief_from_camera` (straight off the sensor), so `--enable_cameras` is
# still required and the KF is unaffected by the observation change.
#
# RAM note: SB3 keeps 2 copies of every observation (obs + next_obs) and ONE observation is
# 81944 B (float32 5ch flow+mask stack 81920 + state 24), so the buffer costs
# buffer_size * 2 * 81944 B. `buffer_size` is the TOTAL transition count, NOT per env.
# 2026-10-12: buffer_size=106667 with gradient_steps=5 (the measured anti-dilution setting,
# 16.3 GiB) -- see sb3_sac_v5_flow_cfg.yaml for the derivation.
#
# The v0 defaults are kept for the knobs Direct-v3-cam turned on. To match that run, add:
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
  --task Isaac-AUV-Tracking-Direct-v5-flow \
  --agent sb3_cfg_entry_point \
  --algorithm SAC \
  --log_interval 20000 \
  --enable_cameras \
  "$@"
