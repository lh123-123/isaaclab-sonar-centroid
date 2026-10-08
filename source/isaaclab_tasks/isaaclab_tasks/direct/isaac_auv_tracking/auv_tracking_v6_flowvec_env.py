# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AUV tracking Direct-v6-flowvec: Direct-v5-flow with the flow *raster* replaced by the six
analytic numbers that raster is a picture of.

Lineage
-------
``auv_tracking_env.py`` (Direct-v0, the "整合项目" belief-SAC task) is kept untouched as the
baseline. ``auv_tracking_v4_cam_env.py`` (Direct-v4-cam) is the home of the
``v4_cam_sb3_sac_800k`` checkpoint and stays frozen. ``auv_tracking_v5_flow_env.py``
(Direct-v5-flow) replaced the actor's RGB stack with a GT optical-flow raster. **This** module is
a copy of Direct-v5-flow and changes exactly one thing: the actor's input representation.

Sensing: the measurement is geometric GT (``kf_measurement="gt"``), matching the GT flow below,
so this task currently runs on analytic ground truth end to end. The camera path and the
``"cam"`` measurement are still present and unchanged; see ``AUVTrackingV6FlowVecEnvCfg`` for
why GT is the right source for the experiment and what has to change before switching back.

Why this exists (the measurement that motivated it)
---------------------------------------------------
The v5 flow raster is, by construction, a *picture of five scalars*. ``_update_gt_flow_image``
computes, per env per step:

    u, v   target's pinhole-projected pixel position   (the disk's centre)
    rad    disk radius = (0.5 * length / dist * fx)    (the disk's size)
    du, dv pixel displacement since the previous frame (the value painted inside the disk)
    vis    geometric visibility: range <= sensor_range and |bearing| <= fov_half

then paints every pixel of that disk with the *same* du/dv, and zeroes the other ~4000.
Measured on a live env (see ``scripts/debug_auv_v5_flow_obs.sh``): the lit disk covers 84-100 of
the 4096 pixels per channel, i.e. ~1% of the frame, and each channel therefore has a two-value
histogram whose entropy is ~0.083 bit/px against a stored 32 bit/px -- roughly 340x redundancy.
The whole 8-channel stack carries on the order of 40 bytes of content.

That is not a defect in itself: five numbers is the sufficient statistic for a tracking decision.
But feeding them as a raster means a convolutional network has to first *undo* the rasterisation
(find the blob, estimate its centroid and area) before it can begin work, and the GAP branch
additionally sees the values attenuated by the ~1% lit fraction. This module removes that
indirection: it hands the network the five numbers.

What changes here, and only this:
3. **Actor observation** -- instead of ``(frame_stack * 2, 64, 64)`` float32 the actor sees
   ``(frame_stack * 6,)`` float32, the six numbers above stacked over ``frame_stack`` frames
   (24 dims at the default K=4). The dict key is still called ``image`` -- a pre-structured
   vector is not an image, but keeping the key means ``CustomSACPolicy``, the SAC actor plumbing
   and the reward path are all unchanged, and the same yaml structure still applies. The one
   real policy-side change is the extractor: ``agents/flow_vec_encoder.FlowVectorExtractor``
   replaces ``MidCNNExtractor`` (a Linear+LayerNorm+Tanh, no convolution).

Per-frame vector layout, in order (all already on a comparable scale):
    [ u / (W-1),  v / (H-1),  rad / 12,  du,  dv,  validity ]
``u``/``v``/``rad`` land in ``[0, 1]`` and ``du``/``dv`` in ``[-1, 1]``, the raster's own
normalisation (displacement over one step divided by W and H respectively). ``validity`` is the
geometric visibility of the target (range + horizontal FOV, in {0, 1}); it exists because ``u``,
``v`` and ``rad`` are all clamped, so "left the frustum" is otherwise indistinguishable from
"pinned to the image border". The same five geometric numbers are still written into
``self._flow_image`` as well, unchanged, so ``scripts/debug_auv_v5_flow_obs.sh`` and any raster
debugging keep working.

Absolute vs differential -- this is why the reset semantics differ from v5's, so it is worth being
explicit. ``u``, ``v`` and ``rad`` are *instantaneous absolute* measurements: they say where the
target is and how big it looks RIGHT NOW, and need no history. Only ``du`` and ``dv`` are
*differential*: a difference against the previous frame, meaningless without one. v5's raster held
only the differential pair, which is why v5 zeroed its whole stack on reset. Here the stack must
REPEAT the current absolute triple and zero only the differential pair:

    reset stack = [ u/W, v/H, rad/12, 0, 0 ] * frame_stack

Zeroing the whole vector would be actively wrong -- it would report the target at pixel (0, 0)
with apparent size 0 on the first step of every episode, a lie the network would have to unlearn.

Consequence worth knowing: only ``2K`` of the ``5K`` dims carry history. The ``3K`` absolute dims
are one measurement repeated K times, i.e. redundant by construction. A leaner variant would feed
the absolute triple ONCE and stack only the differential pair -- ``3 + 2K`` dims, 11 rather than
20 at K=4. That is deliberately NOT done here: the point of this module is to change the
*representation* while holding everything else, stack structure included, fixed against v5.

Caveat 1 -- the ``rad`` slot is a real distance signal here, but it saturates at BOTH ends, so its
useful window should be stated rather than assumed. With this env's camera (``_CAM_HFOV_DEG = 90``
gives ``fx = 32 px``; the 45 deg in the older v4-era comments is stale) and
``length = 0.8 * target_scale = 0.64 m``:

    rad = (0.5 * 0.64 / dist * 32).clamp(1, 12) = (10.24 / dist).clamp(1, 12)

so ``rad`` is pinned at 12 for every ``dist <= 0.853 m`` and pinned at 1 for every
``dist >= 10.24 m``. Inside ``sensor_range = 4 m`` the slot therefore spans ``[0.213, 1.0]``: a
monotone, informative distance cue. (An earlier revision of this docstring claimed saturation at
2.06 m, computed from an assumed 45 deg HFOV. That was wrong by 2.4x in the radius and is
corrected here; ``scripts/debug_auv_v6_flowvec_obs.sh`` asserts the live values instead of
trusting this comment.)

Caveat 2 -- scale of ``du``/``dv``. Measured on a live env, ``|du|`` and ``|dv|`` run from ~3e-4
at the first step (the reset difference is deliberately zero, so this is pure drift) up to ~1e-2
over the first ten steps as the soft-start ends, while ``u``/``v``/``rad`` are O(1). The vector is
fed faithful by default (no gain), matching what the raster offers. Because a vector *can* be
rescaled while a raster cannot, ``cfg.flow_vec_dv_gain`` exists to amplify the ``du``/``dv``
slots; it defaults to 1.0 (off) so the comparison starts clean, and it is the first knob to try if
the actor fails to learn. NOTE the publisher clips to the Box bounds, so a gain > 1 saturates the
observation and the Box bounds must be widened alongside it.


Everything else is Direct-v5-flow verbatim, and therefore Direct-v4-cam and Direct-v0: the
second-WarpAUV target with its unchanged ``BatchHoloTargetMotion``, the camera and the
camera-based KF measurement (``kf_measurement="cam"``, which reads the render straight off the
sensor and so is unaffected by the actor's representation), the reset schedule and the control
chain. The test below says "before changing the representation, try the
representation the task already understands", and that is all this module does.

Reward is v0's information term plus a bearing cost and an out-of-view reset; range and
action-delta are NOT priced (see ``AUVTrackingV6FlowVecEnvCfg`` for the measured weights and
``_compute_step_reward`` / ``_get_dones`` for the mechanics).

Target cruise is 0.5 m/s (max 0.6). Surge maps ``a_v`` onto ``[-1.0, 1.0]`` m/s.
Yaw scale stays 0.5 rad/s. No bearing cone. Geometry pricing is range 0.0 / bearing 0.7
(unclipped, angle clamped at fov_half) / action-delta 0.0 / per-step lost 0.0, plus a
one-time 5.0 on the step that completes ``oov_limit_steps`` = 30 consecutive steps with no
KF measurement, which also terminates the episode. Target a2t at reset is fixed at 2 m
and bearing 0. Motion after that is still soft-start → leash → RRT.

The camera plumbing (mount pose, USD pose sync, RGB ring buffer, GT flow image) is ported
from ``auv_tracking_cam_env.py`` (Direct-v3-cam); that module documents the
``replicate_physics`` frozen-child-prim pitfall.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_tasks.direct.sonar_centroid_track.auv_platform import AUV_SONAR_MOUNT_OFFSET_LOCAL

from .assets.warpauv import WARPAUV_CFG
from .assets.warpauv_cam import WARPAUV_CAM_CFG
from .auv_tracking_env import AUVTrackingEnv, AUVTrackingEnvCfg
from .cam_meas import detect_s1_rgb, focal_px, teacher_lock_rgb
from .sonar_meas import body_polar_from_xy
from .teach_match import (
    _match_detail,
    cut_teach_template,
    detect_points,
    detect_teach,
    orange_core,
    orange_extent,
    teach_from_numpy,
)

# The camera reuses the sonar mount pose so v0 / v3-cam comparisons stay apples-to-apples.
CAM_MOUNT_OFFSET_LOCAL: tuple[float, float, float] = AUV_SONAR_MOUNT_OFFSET_LOCAL

# 90 deg HFOV -- HoloOcean's RGBCamera ``FovAngle`` default, which its scenarios inherit by not
# setting the key (the asset as shipped is 45 deg). On the asset's 20.955 mm aperture that means
# f = (h/2) / tan(HFOV/2) = 10.4775 mm.
# NOTE: changing this constant alone does NOT change the rendered view. The intrinsics are
# authored in the asset (warpauv_base_link/LookFollowCamera0), so the derived focal length has to
# be written onto the prims -- see ``_apply_camera_intrinsics``. ``sensor_fov_deg`` must track this
# value, since the GT ``observed`` gate and the rendered view have to agree on what is visible.
_CAM_HFOV_DEG = 90.0
_CAM_APERTURE_MM = 20.955
_CAM_FOCAL_MM = float(_CAM_APERTURE_MM * 0.5 / np.tan(np.radians(_CAM_HFOV_DEG) * 0.5))




TARGET_AUV_CFG: RigidObjectCfg = WARPAUV_CFG.replace(
    prim_path="/World/envs/env_.*/Target",
    spawn=WARPAUV_CFG.spawn.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # The pose is rewritten every step by _sync_target_entity, so gravity would only
            # add drift between the write and the physics step.
            disable_gravity=True,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        # NOTE: the target keeps the asset's collisions -- deliberately, matching Holo v0, where
        # the target AUV is a normal collidable body. Do not re-add
        # `collision_props=CollisionPropertiesCfg(collision_enabled=False)`: it would be a silent
        # no-op, because the AUV's collision geometry lives in
        # `/warpauv/warpauv_base_link/collisions`, an *instance* prim that
        # `isaaclab.sim.utils.apply_nested` skips outright (`if child_prim.IsInstance(): continue`).
    ),
)
"""Second WarpAUV used as the tracking target (Holo v0 parity: same vehicle as the agent)."""


@configclass
class AUVTrackingV6FlowVecEnvCfg(AUVTrackingEnvCfg):
    """v0 dynamics / reward / motion, camera-only observation, AUV target."""

    # --- (2) sensing: no sonar. Geometric GT measurement (the reliable-data experiment). ---
    #
    # "gt" (not "cam"), as requested: the flow side of this task is already analytic GT
    # (`_update_gt_flow_raster` is pure geometry), so the measurement side is kept analytic as
    # well for now. Two concrete reasons, both about separating causes:
    #   * the "cam" path gates acceptance on a teacher-template NCC lock (`_cam_locked`), depth
    #     read-back and a range fit. That gate is a *perception* signal, so a lost target and a
    #     failed template lock are indistinguishable -- measured for this task: only 27% of
    #     env-steps carried a measurement, most of them lock failures, not geometry.
    #   * the reward's out-of-view reset counts consecutive steps with no measurement, and the
    #     validity channel of the actor's flow vector is read off the same flag. With "cam" those
    #     two would be reporting perception quality rather than "is the target in the frustum".
    # Under "gt" the flag is exactly `r <= sensor_range and |bearing| <= fov_half`, which is what
    # `_publish_flow_vec`'s validity slot and the 30-step counter both mean.
    # Switch back to "cam" once the GT value is replaced by something the vehicle can actually
    # perceive; nothing downstream of `_observed` needs to change for that.
    kf_measurement: str = "gt"
    sensor_range: float = 4.0
    # Must match the camera HFOV below. If it were wider, the GT `observed` gate would report a
    # valid measurement for targets the camera cannot see, so the uncertainty reward would
    # contradict the observation.
    sensor_fov_deg: float = _CAM_HFOV_DEG

    # The agent rides the editable asset *copy* (data/warpauv_cam), which carries the
    # LookFollowCamera0 prim. The target stays on the pristine asset: it needs no camera, and
    # sharing the copy would silently add a camera prim to every target as well.
    robot_cfg: RigidObjectCfg = WARPAUV_CAM_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- (1) target: a second WarpAUV. Motion model is untouched (v0 target_* fields). ---
    target_cfg: RigidObjectCfg = TARGET_AUV_CFG
    # Every episode starts with the target 2 m dead ahead, which is the teach pose.
    # sample_uniform returns the bound itself when min == max. Agent XY and yaw stay random.
    target_init_dist_min: float = 2.0
    target_init_dist_max: float = 2.0
    target_init_ang_min: float = 0.0
    target_init_ang_max: float = 0.0
    # v0 / Holo cruise, same as the 2026-09-22 10:22 run. Spread 1 keeps it constant.
    # Max is 1.2× cruise. Cold start is unchanged: 0–3 s ramp 0→0.5 with leash,
    # 3–5 s full cruise with leash, then free RRT. No bearing cone (base default 180°).
    # Holo v1 parity: action_scale_w 0.5 -> yaw a_w * 0.5.
    # Surge is +-1.0 m/s, i.e. the 800k training snapshot (`params/env.yaml` in
    # models/auv_tracking/v4_cam_sb3_sac_800k). The 2026-09-28/29 revision widened it to
    # v1's +-1.5; reverted here so the config matches the frozen checkpoint.
    action_v_min: float = -1.0
    action_v_max: float = 1.0
    target_speed: float = 0.5
    target_max_speed: float = 0.6
    target_cruise_spread: float = 1.0
    target_peak_factor: float = 0.0
    target_max_accel: float = 0.0

    # --- stable-tracking reward (v4 only; v0 and other tasks are unchanged) ---
    # Keep every optional positive base shaping term off. In particular, matching the target's
    # world-frame velocity vector conflicts with camera centring on this surge+yaw vehicle: the
    # easiest way to match vectors is to travel parallel to the target with it held off-centre.
    # A positive exp(-|Δaction|) bonus similarly rewards a persistent turn rather than correcting
    # the bearing. Deterministic 1M-checkpoint evaluation showed exactly that failure:
    # median |bearing| 16.3° -> 25.8° despite a higher nominal episode return.
    rew_scale_smooth: float = 0.0
    rew_scale_speed: float = 0.0
    rew_speed_sigma: float = 0.5
    rew_scale_lag: float = 0.0
    rew_scale_range: float = 0.0
    rew_range_star: float = 2.0
    rew_scale_bearing: float = 0.0
    rew_bearing_sigma_frac: float = 0.45
    # --- reward: restored to the 800k training snapshot (2026-09-29) ---
    #
    # This block is the Holo-bounded shaping the frozen checkpoint
    # (`models/auv_tracking/v4_cam_sb3_sac_800k`) was trained under, verbatim:
    #     uncertainty 0.2, range 0.2 / sigma 0.6, bearing 0.3, clip 1.0
    #
    # History, so the pendulum is not re-swung blind. The 2026-09-28 "rebalance" argued
    # that `-log det(Sigma)` saturates at the EKF floor (3.39/step at weight 0.2) within
    # ~3 steps of the target entering the 90 deg / 4 m cone, independent of range and
    # bearing, so the agent could park at mean_gt_r 2.14 m / median |bearing| 13.6 deg and
    # still bank the information term. It therefore dropped uncertainty to 0.02, made the
    # geometry costs primary, and removed the `clamp(max=1.0)` that flattened them outside
    # their band. The 2026-09-29 follow-up added a linear bearing tail and a yaw-saturation
    # penalty on top.
    #
    # That line of experiments did not produce a usable policy: the 16-env rebalance run
    # (logs/sb3/.../2026-09-28_18-52-44) peaked at ep_rew -174 @200k and degraded
    # monotonically to -506 @780k with ep_len falling 193 -> 118. This revert restores the
    # configuration whose 800k checkpoint actually tracks (lost 0.3 %, med|alpha| 13.6 deg
    # over 20 deterministic episodes).
    #
    # (Superseded by the revision below, which restores the v0 base and prices only the bearing.)
    # --- reward: v0 base + a bearing term + a 30-step out-of-view reset (2026-09-29 revision) ---
    #
    # Requested change: keep the v0 reward, add an ANGLE (bearing) term, ignore range, and end the
    # episode with a penalty after 30 consecutive steps without a KF measurement.
    #
    # Measured justification (Direct-v6-flowvec, 4 env, 260 random-action steps; instrumented
    # per-step breakdown in this file's `_compute_step_reward`):
    #   * the v0 base term `0.2 * -log det Sigma` sits at **+3.4/step** while the filter is
    #     confident and goes to about -4.4 when it is not. That is the only *positive* signal, so
    #     any shaping term has to be priced against 3.4, not against 1.0.
    #   * `fov_half = 45 deg` and `bearing_sigma = 0.45 * 45 = 20.25 deg`.
    #   * the previous settings (`bearing 0.3`, `clip 1.0`) made the whole bearing term worth at
    #     most 0.3, i.e. 9% of the base, and it hits that ceiling at 20.25 deg -- so beyond 20 deg
    #     the bearing gradient was EXACTLY zero. That is the mechanism behind the documented
    #     "parks off-centre (median |bearing| 13.6-16.3 deg) and still banks the information term".
    #
    # New weights, chosen so the term is priced against the measured base:
    #   |bearing|   penalty        as % of base(+3.4)
    #     8 deg     0.11                 3%
    #    13.6 deg   0.32                 9%
    #    20.25 deg  0.70                21%
    #    30 deg     1.54                45%
    #    45 deg     3.40               100%   <- at the field-of-view edge the base is fully spent
    # `rew_penalty_clip = 0.0` is required: with the legacy clip of 1.0 the cost saturates and the
    # gradient dies outside the sigma band. Boundedness is instead provided by clamping the ANGLE
    # at `fov_half` before squaring (see `_compute_step_reward`), so the worst case stays at
    # -3.4/step and no single step can produce a catastrophic spike (unclamped, a target at 180 deg
    # behind the vehicle would cost 1.4 * (180/20.25)^2 = 110 in one step).
    rew_scale_uncertainty_mean: float = 0.2
    rew_penalty_clip: float = 0.0  # 0 = unclipped; the bearing angle is clamped instead
    # Range is deliberately NOT priced in this revision ("距离先不看"): the v0 information term
    # already rewards holding the filter confident, and range shaping is what the 800k snapshot
    # used to reach r = 2 m. Kept as a field so it can be re-enabled without touching the code.
    rew_penalty_range: float = 0.0
    rew_range_sigma: float = 0.6
    rew_penalty_bearing: float = 0.7
    # Replaced by the out-of-view reset below: paying 0.5/step forever is strictly worse than
    # paying once and restarting, and it double-counts the same failure.
    rew_penalty_lost: float = 0.0
    # v0 has no action-delta cost; keep it off so the reward is "v0 + bearing" as requested.
    rew_penalty_action_delta: float = 0.0
    # Consecutive steps without a KF measurement before the episode is terminated. `_observed`
    # (the KF accept gate) is the same flag v0 used for "measurement valid", so this cannot drift
    # from what the belief actually did. Reset starts with the target 2 m dead ahead, and the
    # measured streak is 0 on the very first step of every episode, so no post-reset grace period
    # is needed. 30 sits well above the scattered losses a working policy shows (the 700k v6
    # checkpoint lost the target on 14.3% of steps, which at that rate is mostly 1-2 step runs)
    # and well below the run lengths a blind policy produces (measured 84-126 consecutive steps
    # under random actions), so it fires on genuine loss without punishing routine drop-outs.
    oov_limit_steps: int = 30
    # One-time penalty on the terminating step. 5.0 matches v0's own `rew_c_penalty` so the task
    # speaks one penalty vocabulary; the termination itself is the dominant cost.
    rew_penalty_oov_terminal: float = 5.0
    # Kept as fields (the reward code reads them) but neutralised: the 800k revision had
    # neither term. Set >0 together with the weights above to re-enable the 09-29 shaping.
    rew_bearing_linear_deg: float = 15.0
    rew_penalty_bearing_linear: float = 0.0
    rew_penalty_yaw_sat: float = 0.0

    # Uniform scale applied to the target AUV only (1.0 = same size as the agent). 0.8 makes the
    # tracked vehicle 80% of the agent's length; see _apply_target_scale.
    target_scale: float = 0.8
    # Optional tint on the Target prim only (see _apply_target_tint). The target is deliberately
    # the *same* vehicle as the agent and the scene has a single uniform DomeLight, so the two are
    # nearly indistinguishable in RGB (measured target-vs-background contrast ~12/255 locally over
    # a ~14x12 px silhouette). A tint makes the figure/ground signal unambiguous.
    #
    # The policy receives RGB, so the tint is a colour difference. Orange was picked
    # because it is darker than the background and still separable:
    #     (0.00,0.00,1.00) pure blue -> luma 0.11
    #     (0.05,0.25,0.95) blue      -> luma 0.27
    #     (0.85,0.38,0.12) orange    -> luma 0.49   (selected below)
    #     (0.00,0.90,1.00) cyan      -> luma 0.64
    #     (0.80,0.80,0.80) untinted  -> luma 0.80
    # Against the bright grey background the orange is both darker and a different hue.
    #
    # NOTE: this deviates from HoloOcean v1, whose target is an untinted BlueROV2, i.e. it makes
    # our visual task easier than the reference. Set back to None before quoting numbers against
    # Holo v1's curve.
    target_tint: tuple[float, float, float] | None = (0.85, 0.38, 0.12)

    # --- camera ---
    # cam_* is the OBSERVATION size and must stay 64x64: it is HoloOcean v1's own ``obs.image``
    # (``configs/envs/v1_config.yml``) and it sets the replay-buffer footprint.
    cam_height: int = 64
    cam_width: int = 64
    # render_* is the RESOLUTION THE CAMERA ACTUALLY RENDERS AT; _read_camera_frame() area-averages
    # it down to cam_*. Rendering at 256x256 and downsampling is what HoloOcean did (it captured
    # LeftCamera at 960x540 and resized to 64x64). It matters because the target's silhouette is
    # only a handful of pixels: rendering straight at 64x64 quantises its edges to whole pixels, so
    # the observation jumps discontinuously as the vehicle yaws, whereas area-averaging a 4x denser
    # render gives the blob anti-aliased (sub-pixel) edges and therefore a smooth function of the
    # relative bearing. Cost is 16x the rendered pixels, so it is a real trade against fps.
    # NOTE: this does NOT change the field of view. FOV comes from the USD focal length and
    # aperture (see _apply_camera_intrinsics); the render product only samples that frustum.
    render_height: int = 256
    render_width: int = 256
    cam_hfov_deg: float = _CAM_HFOV_DEG
    # Aperture used to derive the focal length written onto the adopted camera prim. Defaults to
    # the value the asset was authored with; see _apply_camera_intrinsics.
    cam_aperture_mm: float = _CAM_APERTURE_MM
    frame_stack: int = 4
    # Scalar gain applied to the du/dv slots of the flow vector (see the module docstring,
    # "Caveat 2"). 1.0 = faithful, no gain. A vector can be rescaled where a raster cannot, so
    # this is the cheap knob to try first if the actor fails to learn; it is deliberately off by
    # default so the v5-vs-v6 comparison is not confounded by a hand-tuned pre-scale.
    flow_vec_dv_gain: float = 1.0
    # Mirror the robot's simulated pose into the robot prim's USD xform every render step.
    # Required: with replicate_physics=True PhysX writes poses to Fabric and never updates USD,
    # so the camera (a child prim, not a rigid body) would otherwise inherit a frozen transform
    # and the rendered view would never follow the vehicle. See _sync_robot_usd_pose.
    cam_sync_usd_pose: bool = True
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        # Wraps the LookFollowCamera0 prim that ships inside the asset copy. spawn=None is the
        # supported "adopt an existing prim" path (interactive_scene.py only spawns when
        # `asset_cfg.spawn is not None`).
        prim_path="/World/envs/env_.*/Robot/warpauv_base_link/LookFollowCamera0",
        data_types=["rgb", "distance_to_image_plane"],
        # Renders at render_* and is area-averaged down to cam_* in _read_camera_frame.
        width=render_width,
        height=render_height,
        # NOTE: left at the default (False) to avoid a redundant per-step XFormPrim read;
        # nothing in this env consumes Camera.data.pos_w. Derive the camera pose from the
        # robot's simulated pose plus CAM_MOUNT_OFFSET_LOCAL if ever needed.
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAM_MOUNT_OFFSET_LOCAL,
            rot=(1.0, 0.0, 0.0, 0.0),
            # "world": forward +X, up +Z in the Robot frame == the body convention the WarpAUV
            # USD and the sonar mount already use.
            convention="world",
        ),
        # No spawner: the mount pose and the intrinsics (focal 25.2949 mm on a 20.955 mm
        # aperture = 45 deg HFOV, clipping 0.05-30 m) come from the asset. `offset` above is
        # therefore inert -- camera.py only applies it inside the spawn branch.
        spawn=None,
    )

    # Asymmetric observation, mirroring HoloOcean v1's ``CustomSACPolicy``: the *actor* consumes
    # the frame stack as ``"image"`` while the *critic* consumes the 4-D KF belief as ``"state"``.
    #
    # Why the split: the v0 reward is ``c_mean * -log det(Sigma)``, i.e. a function of the filter
    # covariance. A symmetric CNN critic has to *infer* Sigma from four luma frames, which is not
    # recoverable from the image, and the resulting Q targets diverge (observed: critic_loss
    # 9 -> 97, actor_loss 0.8 -> 108, ep_rew -222 -> -320 while ep_len fell 192 -> 117). Holo's
    # asymmetric critic sidesteps this by reading the belief directly.
    #
    # NOTE: this does NOT use IsaacLab's ``state_space`` / ``single_observation_space["critic"]``
    # mechanism. `isaaclab_rl/sb3.py::_process_obs` drops everything but ``obs["policy"]``
    # ("Sb3 doesn't support asymmetric observation spaces"), so the state has to live *inside* the
    # policy space as a Dict key. The v2 flow task (AUVTrackingFlowEnvCfg) uses the same layout.
    #
    # Bounds are HoloOcean v0's own ``state_lower_bound`` / ``state_upper_bound`` for
    # [d, alpha, log det(Sigma), observed] with a single target (see world_auv_v0.py); the
    # obstacle pair is absent because this task has no obstacles.
    #
    # The actor input is a stack of ``frame_stack`` pre-structured flow VECTORS of 6 numbers each
    # -- (u/W, v/H, rad/12, du, dv, validity), oldest first -- so ``frame_stack * 6`` dims, 24 for
    # the default K=4. See the module docstring for what those six are and why the raster version
    # was replaced by them.
    #
    # Bounds (see ``_flow_vec_now``, which clips to exactly these):
    #   u/(W-1), v/(H-1), rad/12  -> [0, 1]
    #   du, dv                    -> [-1, 1]   (displacement / W and / H)
    #   validity                  -> {0, 1}
    # so the shared [-1, 1] bounds below are the tight superset and there is one Box for all three
    # groups. Note this means ``flow_vec_dv_gain > 1`` can push du/dv out of the space; that is a
    # deliberate opt-in (a gain is a modelling choice, and the extractor does not boundary-check),
    # it just must not be forgotten if a caller starts assuming the bounds hold.
    #
    # float32, NOT uint8: these are signed analytic quantities, so the RGB "1 byte/pixel" trick
    # does not apply. Two consequences, both wanted:
    #   * SB3's ``is_image_space`` requires uint8 + [0, 255] AND a 1/3/4-channel shape, so this is
    #     a plain Box and ``preprocess_obs`` leaves it alone (no hidden /255).
    #   * Replay memory is 4 bytes per value per step -- but there are only 24 values, so see the
    #     buffer size in ``sb3_sac_v6_flowvec_cfg.yaml``: this is thousands of times smaller than
    #     v5's raster.
    #
    # The key is still called ``image`` even though a pre-structured vector is not an image, so
    # that ``CustomSACPolicy``, the SAC actor plumbing and every existing consumer keep working
    # unchanged. The camera is still spawned and rendered (the ``"cam"`` measurement path and the
    # raster debug scripts use it), even though the current ``"gt"`` measurement does not read it.
    observation_space = {
        # GT flow vector stack, oldest frame first:
        # [u/W, v/H, rad/12, du, dv, validity] * frame_stack.
        "image": gym.spaces.Box(low=-1.0, high=1.0, shape=(frame_stack * 6,), dtype=np.float32),
        "state": gym.spaces.Box(
            low=np.array([0.0, -np.pi, -50.0, 0.0], dtype=np.float32),
            high=np.array([600.0, np.pi, 50.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        ),
    }


class AUVTrackingV6FlowVecEnv(AUVTrackingEnv):
    """Direct-v5-flow's sibling: the actor sees K stacked 6-D flow VECTORS instead of a raster.

    The second-AUV target, its motion model, the camera, the KF measurement, the reward, the reset
    schedule and the control chain are all Direct-v5-flow verbatim -- and therefore Direct-v4-cam
    and Direct-v0. The actor's input representation is the only change, so
    a run here compared against Direct-v5-flow at equal budget and equal information isolates what
    the convolutional encoder costs (or earns) on this task.
    """

    cfg: AUVTrackingV6FlowVecEnvCfg

    def _setup_scene(self):
        # Base builds robot + target, clones the envs and (because kf_measurement is gt/cam) skips
        # the sonar. The camera must come after clone_environments: SensorBase counts envs by
        # resolving the ".../env_.*/Robot" parent path.
        self._apply_target_tint()
        self._apply_target_scale()
        super()._setup_scene()
        # Only meaningful now: the binding this reads back did not exist until spawn_from_usd ran
        # inside super()._setup_scene().
        self._verify_target_tint()

        # Must run before TiledCamera(): the render product takes its intrinsics from the USD
        # camera attributes, so the focal length has to be on the prims by then.
        self._apply_camera_intrinsics()
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # Allocate the vector stack here: _setup_scene runs before the first _reset_idx, and
        # self.device / self.num_envs are already final. float32 because the values are signed
        # analytic quantities.
        #
        # `_flow_buf` is the (N, frame_stack * 6) actor observation. `_flow_vec_now` holds this
        # step's single 6-vector before it is pushed onto the stack; five of its six entries are
        # the same (u, v, rad, du, dv) that `_update_gt_flow_raster` draws into the raster, and the
        # sixth is the geometric visibility flag, so the two representations always agree.
        k = int(self.cfg.frame_stack)
        self._flow_buf = torch.zeros((self.num_envs, k * 6), dtype=torch.float32, device=self.device)
        self._flow_vec_now = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._teach: TeachRef | None = None
        self._teach_vec = torch.zeros(self.num_envs, 4, device=self.device)
        h, w = int(self.cfg.cam_height), int(self.cfg.cam_width)
        self._flow_image = torch.zeros((self.num_envs, 2, h, w), device=self.device)
        self._gt_px = torch.zeros(self.num_envs, 2, device=self.device)
        self._gt_px_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._flow_vec = torch.zeros(self.num_envs, 3, device=self.device)
        # Geometric visibility of the target (range + horizontal FOV), computed by the raster step
        # and published as the flow vector's 6th slot. Allocated here so the very first
        # `_publish_flow_vec` has something to read before any step has run.
        self._flow_vec_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Consecutive steps without a KF measurement, i.e. how long the target has been out of view
        # (range or horizontal FOV). Read by the reward's one-time penalty and by `_get_dones`,
        # which terminates the episode once it reaches `cfg.oov_limit_steps`; zeroed in `_reset_idx`.
        self._oov_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Reset-frame orange bbox, labelled 2 m. Later range uses the side that scaled
        # more like a zoom and less like a heading change.
        self._extent_ref_w = torch.zeros(self.num_envs, device=self.device)
        self._extent_ref_h = torch.zeros(self.num_envs, device=self.device)
        self._extent_ref_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # #region agent log
        self._core_ref_rms = torch.zeros(self.num_envs, device=self.device)
        self._core_ref_n = torch.zeros(self.num_envs, device=self.device)
        self._core_ref_w = torch.zeros(self.num_envs, device=self.device)
        self._core_ref_h = torch.zeros(self.num_envs, device=self.device)
        # #endregion
        self._load_teach_if_present()
        # Set on reset, consumed by _get_observations once a *new* frame has been rendered.
        self._stack_pending_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cam_templates: list[np.ndarray | None] = [None] * self.num_envs
        self._cam_locked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cam_length_m = max(0.68 * float(getattr(self.cfg, "target_scale", 1.0)), 0.15)

        # Writable USD handle over the robot roots; see _sync_robot_usd_pose.
        self._robot_usd = None
        if bool(getattr(self.cfg, "cam_sync_usd_pose", True)):
            self._init_usd_pose_sync()

    def _apply_target_tint(self) -> None:
        """Bind a distinct visual material on the Target prim only (see cfg.target_tint).

        Applied to the *cfg* before the target is spawned, because the material is bound during
        spawning. Nothing happens when target_tint is None, which keeps the default behaviour
        (target identical to the agent).
        """
        tint = getattr(self.cfg, "target_tint", None)
        if tint is None:
            return
        spawn = self.cfg.target_cfg.spawn
        if spawn is None or not hasattr(spawn, "visual_material"):
            print("[v6-flowvec] WARN: target_tint set but target spawn does not support visual_material", flush=True)
            return
        # PreviewSurfaceCfg is a VisualMaterialCfg, i.e. exactly the type the field expects.
        spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(c) for c in tint))
        # Deliberately no "tinted" success line here. Assigning the spawn cfg only *requests* the
        # material; the actual binding happens later inside spawn_from_usd, and it has to beat the
        # UsdPreviewSurface already bound to mesh_0 *inside* the `visuals` instance prototype. That
        # is precisely the kind of override that fails without raising. _verify_target_tint() reads
        # back what really landed on the stage instead of assuming this line means anything.
        print(f"[v6-flowvec] target tint requested {tuple(float(c) for c in tint)} (verified post-spawn)", flush=True)

    def _verify_target_tint(self) -> None:
        """Read back the material binding that actually landed on the spawned Target prim.

        The chain that has to hold: ``spawn_from_usd`` creates ``<Target>/material`` and calls
        ``bind_visual_material(<Target>, ...)`` (sim/spawners/from_files/from_files.py:324), which
        defaults to ``strongerThanDescendants`` (sim/utils.py:326). Strong-ancestor bindings do
        outrank a descendant's direct binding, which is what lets this beat the prototype's own
        material -- but only if the relationship exists and really carries that strength. Report
        the authored state so a silent no-op cannot be mistaken for a tinted target.
        """
        tint = getattr(self.cfg, "target_tint", None)
        if tint is None:
            return
        from pxr import Usd, UsdShade

        root_path = f"{self.scene.env_prim_paths[0]}/Target"
        prim = self.scene.stage.GetPrimAtPath(root_path)
        if not prim or not prim.IsValid():
            print(f"[v6-flowvec] WARN: target prim {root_path} missing; cannot verify tint", flush=True)
            return

        rel = prim.GetRelationship("material:binding")
        targets = [str(t) for t in rel.GetTargets()] if rel else []
        # Read the strength carefully: it decides whether this binding outranks the direct one on
        # mesh_0. The metadata is absent when the strength is the USD default, so "absent" must NOT
        # be read as "weak" -- that would turn a working tint into a false alarm. Prefer the USD API
        # (which applies the default) and only fall back to the raw metadata.
        strength, strength_known = "unknown", False
        if rel:
            try:
                strength = str(UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(rel))
                strength_known = True
            except Exception:
                try:
                    md = rel.GetMetadata("bindMaterialAs")
                    if md:
                        strength, strength_known = str(md), True
                except Exception as exc:  # pragma: no cover - metadata read should not be fatal
                    strength = f"unreadable ({exc})"

        # Whatever is bound, read its authored base colour rather than trusting the request.
        bound_colour = None
        for t in targets:
            mat = self.scene.stage.GetPrimAtPath(t)
            if not mat or not mat.IsValid():
                continue
            for node in Usd.PrimRange(mat):
                if not node.IsA(UsdShade.Shader):
                    continue
                for inp in UsdShade.Shader(node).GetInputs():
                    if "diffuse" in inp.GetBaseName().lower() or "base_color" in inp.GetBaseName().lower():
                        if inp.Get() is not None:
                            bound_colour = tuple(round(float(c), 4) for c in inp.Get())

        want = tuple(round(float(c), 4) for c in tint)
        # The colour landing is the thing that actually matters: it proves the material was created
        # and that the relationship points at it. An explicitly weak strength is a real failure; an
        # unknown one is reported but not treated as one.
        colour_ok = bound_colour is not None and all(abs(a - b) < 1e-3 for a, b in zip(bound_colour, want))
        explicitly_weak = strength_known and "weakerThanDescendants" in strength
        ok = bool(targets) and colour_ok and not explicitly_weak
        print(
            f"[v6-flowvec] target tint {'OK' if ok else 'NOT CONFIRMED'}: rel={targets or 'ABSENT'}"
            f" bound_diffuse={bound_colour} want={want} strength={strength}"
            f"{'' if strength_known else ' (absent -> USD default)'}",
            flush=True,
        )
        if not ok:
            print(
                "[v6-flowvec] WARN: the tint may be shadowed by the prototype binding on mesh_0"
                " (/__Prototype_2/mesh_0 -> .../Looks/material_Material_001). The camera would still"
                " see the target by silhouette, but not by colour.",
                flush=True,
            )

    def _apply_target_scale(self) -> None:
        """Apply ``cfg.target_scale`` to the target's spawn transform only (see cfg.target_scale).

        ``UsdFileCfg.scale`` is authored onto the Target prim, so the visual meshes, the collision
        geometry and the prim's AABB all shrink together -- i.e. the *vehicle* gets smaller rather
        than only its appearance. The agent and the base ``WARPAUV_CFG`` are untouched
        (``RigidObjectCfg.replace`` deep-copies the spawn config).
        """
        scale = float(getattr(self.cfg, "target_scale", 1.0))
        spawn = self.cfg.target_cfg.spawn
        if not hasattr(spawn, "scale"):
            print("[v6-flowvec] WARN: target spawn has no 'scale'; target left at its authored size", flush=True)
            return
        spawn.scale = (scale, scale, scale)
        print(f"[v6-flowvec] target AUV scaled x{scale:g}", flush=True)
    def _teach_npz_path(self) -> Path:
        # this file: source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/...
        root = Path(__file__).resolve().parents[5]
        return root / "models" / "auv_tracking" / "teach_v4_2m_0deg.npz"

    def _load_teach_if_present(self) -> None:
        path = self._teach_npz_path()
        if not path.is_file():
            return
        blob = np.load(path)
        image = blob["image"]
        pts = detect_points(torch.as_tensor(np.ascontiguousarray(image)).unsqueeze(0))
        keep = pts[0, :, 0] >= 0
        points = pts[0, keep].cpu().numpy()
        self._teach = teach_from_numpy(image, points, float(blob["range_m"]))

    def _camera_rgb(self) -> torch.Tensor:
        """Render-resolution RGB ``(N, H, W, 3)`` uint8."""
        rgb = self._tiled_camera.data.output["rgb"]
        if rgb.dtype != torch.uint8:
            rgb = rgb.clamp(0, 255).to(torch.uint8)
        return rgb[..., :3]

    def _capture_teach(self, rgb: torch.Tensor) -> None:
        """Save env 0's current render as the shared teach image, if it is not already on disk."""
        if self._teach is not None:
            return
        frame = rgb[0]
        if float(frame.float().std()) < 1.0:
            return
        pts = detect_points(frame.unsqueeze(0))
        keep = pts[0, :, 0] >= 0
        if int(keep.sum()) < 4:
            return
        image = frame.detach().cpu().numpy()
        points = pts[0, keep].detach().cpu().numpy()
        path = self._teach_npz_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, image=image, points=points.astype(np.float32), range_m=np.float32(2.0))
        self._teach = teach_from_numpy(image, points, 2.0)

    def _read_camera_frame(self) -> torch.Tensor:
        """Camera RGB ``(N, Rh, Rw, 3)`` uint8 -> observation RGB ``(N, 3, H, W)`` uint8.

        Area-averaged down to cam_* before the uint8 quantisation.
        """
        rgb = self._camera_rgb().permute(0, 3, 1, 2).to(torch.float32)
        if rgb.shape[-2:] != (int(self.cfg.cam_height), int(self.cfg.cam_width)):
            rgb = F.interpolate(rgb, size=(int(self.cfg.cam_height), int(self.cfg.cam_width)), mode="area")
        return rgb.round().clamp_(0.0, 255.0).to(torch.uint8)

    def _camera_prim_paths(self) -> list[str]:
        """One adopted LookFollowCamera0 prim per env (clone_environments copies it per env)."""
        return [f"{p}/Robot/warpauv_base_link/LookFollowCamera0" for p in self.scene.env_prim_paths]

    def _apply_camera_intrinsics(self) -> None:
        """Write the configured HFOV onto the adopted camera prims' USD intrinsics.

        The camera is *adopted* (``tiled_camera.spawn is None``), so every focal-length field on
        ``TiledCameraCfg`` is inert -- the effective intrinsics are whatever the asset authored on
        ``LookFollowCamera0`` (45 deg HFOV). The render product derives its intrinsics from the USD
        camera attributes, so widening the view means writing ``focalLength`` onto the prims.

        Deriving the focal length from the aperture (rather than hard-coding it) keeps whatever
        aperture the asset shipped, so only the FOV changes. This deliberately mutates USD and not
        the asset itself: v3-cam shares ``data/warpauv_cam/warpauv.usd`` and must stay at 45 deg so
        existing v3-cam runs and checkpoints remain comparable.
        """
        from pxr import UsdGeom

        hfov = float(self.cfg.cam_hfov_deg)
        aperture = float(getattr(self.cfg, "cam_aperture_mm", _CAM_APERTURE_MM))
        focal = float(aperture * 0.5 / np.tan(np.radians(hfov) * 0.5))
        for path in self._camera_prim_paths():
            prim = self.sim.stage.GetPrimAtPath(path)
            if prim is None or not prim.IsValid():
                print(f"[v6-flowvec] WARN: camera prim missing, intrinsics not applied: {path}", flush=True)
                continue
            UsdGeom.Camera(prim).GetFocalLengthAttr().Set(focal)

    def _init_usd_pose_sync(self) -> None:
        """Open a writable USD view over the robot roots (one prim per env)."""
        from isaacsim.core.prims import XFormPrim

        robot_paths = [f"{p}/Robot" for p in self.scene.env_prim_paths]
        try:
            # reset_xform_properties=False: keep the authored xform ops, we only write values.
            self._robot_usd = XFormPrim(robot_paths, reset_xform_properties=False)
        except Exception as exc:  # noqa: BLE001 - never let a USD nicety break the env
            self._robot_usd = None
            print(f"[v6-flowvec] WARN: USD pose sync disabled, could not open {robot_paths[:1]}: {exc}", flush=True)

    def _sync_robot_usd_pose(self) -> None:
        """Mirror the robot's simulated world pose into its USD xform.

        With ``replicate_physics=True`` PhysX keeps rigid-body poses in Fabric and never writes
        them back to USD, so every USD xform stays at its authored value forever. The camera is
        a *child* prim and is not a rigid body, so the renderer resolves it from USD only --
        meaning it would inherit a frozen parent transform and the rendered view would never
        move. Writing the robot's pose into USD makes the child camera track the vehicle through
        normal USD composition, so no camera-convention math is duplicated here.

        `XFormPrim.set_world_poses(usd=True)` converts our world pose into the local transform of
        each prim's (frozen) parent, which is exactly the correction needed.
        """
        # NOTE: this view uses the torch backend, so pass tensors -- numpy arrays raise
        # "AttributeError: 'numpy.ndarray' object has no attribute 'detach'".
        self._robot_usd.set_world_poses(
            positions=self._robot.data.root_pos_w,
            orientations=self._robot.data.root_quat_w,
            usd=True,
        )

    def _apply_action(self) -> None:
        super()._apply_action()
        # `step()` renders inside the decimation loop as
        #   if _sim_step_counter % sim.render_interval == 0: sim.render()
        # so writing here, on exactly those iterations, lands the fresh pose in USD before the
        # renderer reads it. _sim_step_counter is already incremented for this iteration, so the
        # modulo matches the engine's condition. Doing it every substep would mean ~24x the USD
        # writes for no benefit.
        interval = max(int(self.cfg.sim.render_interval), 1)
        on_render = self._sim_step_counter % interval == 0
        if self._robot_usd is not None and on_render:
            self._sync_robot_usd_pose()

    def _compute_step_reward(self) -> torch.Tensor:
        """v0 information reward + a bounded bearing cost + the out-of-view terminal penalty.

        Composition (all weights in ``AUVTrackingV6FlowVecEnvCfg``; see the measurements there):
          * ``rew_scale_uncertainty_mean * -log det(Sigma)`` -- v0's term, kept verbatim.
          * ``- rew_penalty_bearing * (|alpha|_clamped / bearing_sigma)^2`` -- the added angle
            term. ``alpha`` is clamped to ``fov_half`` first: that keeps the cost bounded and
            monotone without reviving the legacy ``clamp(max=1.0)``, which flattened the cost
            (and therefore zeroed the gradient) as soon as the target left the sigma band.
          * ``- rew_c_penalty`` on collision -- inside ``super()``, i.e. v0's own penalty.
          * the loss of the measurement only costs a reward on the step that trips
            ``oov_limit_steps`` (see ``_get_dones``); there is no per-step lost cost any more.

        Also maintains ``_oov_streak``: consecutive steps without a KF measurement. It is updated
        here because this runs exactly once per step, after ``_update_belief`` has refreshed
        ``_observed``, and before ``_get_dones`` has to decide whether to terminate.
        """
        rew = super()._compute_step_reward()

        agent_pos, agent_yaw = self._agent_pose_2d()
        _, gt_r, gt_a = self.kf.get_observation(self._target_pos, agent_pos, agent_yaw)

        range_sigma = max(float(self.cfg.rew_range_sigma), 1.0e-6)
        range_cost = ((gt_r - float(self.cfg.rew_range_star)) / range_sigma) ** 2

        bearing_sigma = max(
            float(self.cfg.rew_bearing_sigma_frac) * float(self._fov_half),
            1.0e-6,
        )
        # Clamp the ANGLE (not the cost) at the field-of-view half-angle. Beyond it the target is
        # not visible at all, so the exact bearing carries no control information and letting
        # tan-like blow-ups through would produce enormous one-step penalties. Inside the FOV the
        # quadratic is untouched, so the gradient exists everywhere the camera can actually see.
        bearing_eff = torch.clamp(gt_a.abs(), max=float(self._fov_half))
        bearing_cost = (bearing_eff / bearing_sigma) ** 2

        action_delta_cost = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)
        lost_cost = (~self._observed).to(dtype=rew.dtype)
        # Linear bearing tail beyond rew_bearing_linear_deg (default 15): 0.02/deg,
        # so the late-episode drift keeps a gradient where the quadratic is flat-ish.
        lin_deg = float(getattr(self.cfg, "rew_bearing_linear_deg", 15.0))
        bearing_lin_cost = torch.clamp(gt_a.abs() * 57.29578 - lin_deg, min=0.0)
        # Yaw-saturation term: |a_w| > 0.9 means commanding at the 0.5 rad/s stop.
        yaw_sat_cost = (self._actions[:, 1].abs() > 0.9).to(dtype=rew.dtype)

        # clip > 0 reproduces the legacy Holo-bounded shaping (constant cost outside the band);
        # clip <= 0 leaves the costs unbounded so the gradient survives off-target. The default is
        # now 0.0 -- boundedness comes from clamping the bearing *angle* above instead.
        clip = float(getattr(self.cfg, "rew_penalty_clip", 0.0))
        if clip > 0.0:
            range_cost = torch.clamp(range_cost, max=clip)
            bearing_cost = torch.clamp(bearing_cost, max=clip)
            action_delta_cost = torch.clamp(action_delta_cost, max=clip)

        rew = (
            rew
            - float(self.cfg.rew_penalty_range) * range_cost
            - float(self.cfg.rew_penalty_bearing) * bearing_cost
            - float(getattr(self.cfg, "rew_penalty_bearing_linear", 0.0)) * bearing_lin_cost
            - float(self.cfg.rew_penalty_action_delta) * action_delta_cost
            - float(getattr(self.cfg, "rew_penalty_yaw_sat", 0.0)) * yaw_sat_cost
            - float(self.cfg.rew_penalty_lost) * lost_cost
        )

        # --- out-of-view tracking: consecutive steps without a KF measurement ---
        # `_observed` (the KF accept gate) was refreshed by `_update_belief` at the top of
        # `_get_dones`, i.e. immediately before this runs. Counting here means `_get_dones` can
        # read a streak that already includes the current step. `_reset_idx` zeroes it.
        self._oov_streak = torch.where(
            self._observed,
            torch.zeros_like(self._oov_streak),
            self._oov_streak + 1,
        )
        oov_limit = int(self.cfg.oov_limit_steps)
        # `== limit` rather than `>= limit`: the streak advances by one per step and any
        # observation resets it, so the penalty lands on exactly one step per lost episode.
        lost_terminal = self._oov_streak == oov_limit
        if bool(lost_terminal.any()):
            rew = rew - lost_terminal.to(dtype=rew.dtype) * float(self.cfg.rew_penalty_oov_terminal)

        return torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=-50.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """v0 terminations plus the out-of-view reset.

        v0 ends an episode on (a) the rolling 100-step mean reward dropping below
        ``rew_done_threshold`` (``done_by_reward``) or (b) the 200-step timeout. Proximity /
        out-of-bounds only penalise, they never terminate -- matched here.

        This adds (c): ``oov_limit_steps`` consecutive steps with no KF measurement. The streak
        and its one-time penalty are maintained in ``_compute_step_reward``, which ``super()``
        calls just above, so the mask is already current for this step.

        Terminating on a *lost target* is not the same as v0's behaviour (sensor range 4 m, so v0
        simply accrued a negative information reward forever). It is safe here because 30 steps is
        an order of magnitude above the scattered drop-outs a working policy produces, while a
        blind policy runs 100+ steps without a measurement (measured: 84-126 under random actions).
        """
        terminated, truncated = super()._get_dones()
        oov_done = self._oov_streak >= int(self.cfg.oov_limit_steps)
        return terminated | oov_done, truncated

    def _apply_reset_extent_range(self, rgb_nhwc: torch.Tensor) -> None:
        """Replace range with the reset-frame bbox side that changed less.

        Reset is 2 m and 0 deg bearing, but target heading still faces the first waypoint,
        so the teach photo's orange size is the wrong ruler. The side of the bbox that
        scales more like a zoom, and less like a turn, is the range cue.
        """
        if not hasattr(self, "_extent_ref_ok"):
            return
        width, height, count = orange_extent(rgb_nhwc)
        width = width.to(device=self.device)
        height = height.to(device=self.device)
        count = count.to(device=self.device)
        fresh = (~self._extent_ref_ok) & (count >= 20.0) & (width >= 2.0) & (height >= 2.0)
        self._extent_ref_w = torch.where(fresh, width, self._extent_ref_w)
        self._extent_ref_h = torch.where(fresh, height, self._extent_ref_h)
        self._extent_ref_ok = self._extent_ref_ok | fresh
        # #region agent log
        if bool(fresh.any()):
            ref_rms, ref_n, ref_cw, ref_ch = orange_core(rgb_nhwc)
            ref_rms = ref_rms.to(device=self.device)
            ref_n = ref_n.to(device=self.device)
            ref_cw = ref_cw.to(device=self.device)
            ref_ch = ref_ch.to(device=self.device)
            self._core_ref_rms = torch.where(fresh, ref_rms, self._core_ref_rms)
            self._core_ref_n = torch.where(fresh, ref_n, self._core_ref_n)
            self._core_ref_w = torch.where(fresh, ref_cw, self._core_ref_w)
            self._core_ref_h = torch.where(fresh, ref_ch, self._core_ref_h)
        # #endregion
        usable = self._extent_ref_ok & (count >= 20.0) & (width >= 2.0) & (height >= 2.0)
        if not bool(usable.any()):
            return
        rw = (2.0 * self._extent_ref_w / width.clamp(min=1.0)).clamp(0.4, 13.3)
        rh = (2.0 * self._extent_ref_h / height.clamp(min=1.0)).clamp(0.4, 13.3)
        use_w = (width / self._extent_ref_w.clamp(min=1.0)).log().abs() <= (
            height / self._extent_ref_h.clamp(min=1.0)
        ).log().abs()
        r_hat = torch.where(use_w, rw, rh)
        self._teach_vec[usable, 0] = r_hat[usable]
        self._teach_vec[usable, 2] = (2.0 / r_hat[usable]).clamp(0.15, 5.0)

    def _get_observations(self) -> dict:
        # The base does the KF bookkeeping (predict + post-reset belief sense). Under
        # kf_measurement="cam" the belief sense reads the camera render straight off the sensor
        # (see _update_belief_from_camera), so the KF still gets its measurement even though the
        # render no longer feeds the actor. The base returns its "policy" entry as the 4-D belief
        # state [r, alpha, log det(Sigma), observed]; that vector is kept and becomes the critic's
        # input under the asymmetric split.
        obs = super()._get_observations()
        belief_state = obs["policy"]
        k = int(self.cfg.frame_stack)

        # Analytic GT flow for this step. Also advances the previous-pixel state (_gt_px) and
        # publishes self._flow_vec_now, so it must run exactly once per step and before the stack
        # is shifted.
        self._update_gt_flow_image()

        # .clone(): this mask is read twice below and the assignment below mutates the source.
        pending = self._stack_pending_reset.clone()
        if bool(pending.any()):
            # Fresh episode: repeat the current 6-vector across the stack. This is NOT a zero
            # stack -- see "Absolute vs differential" in the module docstring.
            # `_update_gt_flow_image` already zeroed du/dv for these envs (there is no previous
            # frame), while u/v/rad/validity are valid instantaneous measurements that must
            # survive. Zeroing the whole vector would report the target at pixel (0, 0) with
            # size 0 and "not visible".
            self._flow_buf[pending] = self._flow_vec_now[pending].repeat(1, k)
            self._stack_pending_reset[pending] = False

        live = ~pending
        if bool(live.any()):
            # Shift by one 6-vector and append the newest.
            self._flow_buf[live, :-6] = self._flow_buf[live, 6:].clone()
            self._flow_buf[live, -6:] = self._flow_vec_now[live]

        obs["policy"] = {
            "image": self._flow_buf,
            "state": belief_state,
        }
        return obs

    def _update_gt_flow_image(self) -> None:
        """Compute this step's GT flow and publish it BOTH ways.

        Writes the 2-channel raster into ``self._flow_image`` (kept for debugging, unchanged from
        v5) and the equivalent 6-number vector into ``self._flow_vec_now`` (what v6's actor
        actually consumes). One function, one set of intermediate values, so the two
        representations cannot drift apart -- the raster is exactly a picture of the vector.
        """
        self._update_gt_flow_raster()
        self._publish_flow_vec()

    def _update_gt_flow_raster(self) -> None:
        """Stamp the target's GT pixel motion onto a 2-channel map.

        The disk sits on the pinhole projection of the target. Its value is the
        displacement since the previous frame, divided by the image size, so a
        full-frame jump is ±1. Reset frames are zero.
        """
        agent_pos, agent_yaw = self._agent_pose_2d()
        rel = self._target_pos - agent_pos
        bear = torch.atan2(rel[:, 1], rel[:, 0]) - agent_yaw
        bear = torch.atan2(torch.sin(bear), torch.cos(bear))
        dist = torch.linalg.norm(rel, dim=-1).clamp(min=1.0e-3)
        dz = float(self.cfg.starting_depth) - self._robot.data.root_pos_w[:, 2]
        elev = torch.atan2(dz, dist)

        w = int(self.cfg.cam_width)
        h = int(self.cfg.cam_height)
        fx = focal_px(w, float(self.cfg.cam_hfov_deg))
        cx = 0.5 * (w - 1)
        cy = 0.5 * (h - 1)
        u = cx - fx * torch.tan(bear)
        v = cy - fx * torch.tan(elev)
        fresh = ~self._gt_px_ok
        du = torch.where(fresh, torch.zeros_like(u), (u - self._gt_px[:, 0]) / w)
        dv = torch.where(fresh, torch.zeros_like(v), (v - self._gt_px[:, 1]) / h)
        du = du.clamp(-1.0, 1.0)
        dv = dv.clamp(-1.0, 1.0)
        self._gt_px[:, 0] = u
        self._gt_px[:, 1] = v
        self._gt_px_ok[:] = True

        length = 0.8 * float(getattr(self.cfg, "target_scale", 1.0))
        rad = (0.5 * length / dist * fx).clamp(1.0, 12.0)
        ys = torch.arange(h, device=self.device, dtype=torch.float32)
        xs = torch.arange(w, device=self.device, dtype=torch.float32)
        dx = xs.view(1, 1, w) - u.view(-1, 1, 1)
        dy = ys.view(1, h, 1) - v.view(-1, 1, 1)
        mask = dx * dx + dy * dy <= rad.view(-1, 1, 1) ** 2
        self._flow_image.zero_()
        self._flow_image[:, 0] = torch.where(mask, du.view(-1, 1, 1), torch.zeros_like(dx))
        self._flow_image[:, 1] = torch.where(mask, dv.view(-1, 1, 1), torch.zeros_like(dy))

        # Hand the raw scalars to the vector publisher below. Saved rather than returned because
        # `_update_gt_flow_image` is called for its side effects from `_get_observations`.
        self._flow_vec_u = u
        self._flow_vec_v = v
        self._flow_vec_rad = rad
        self._flow_vec_du = du
        self._flow_vec_dv = dv
        # Visibility for the vector's validity slot. Deliberately recomputed from the same
        # geometry the raster just used (range + horizontal bearing vs the sensor frustum) rather
        # than reused from `_observed`: `_observed` is the KF's *accept* gate, which under
        # ``kf_measurement="cam"`` also depends on the appearance path, while this slot means "the
        # target projects inside the camera frustum". Under the current ``"gt"`` measurement the
        # two coincide exactly (verified: 0 mismatches over 200 steps).
        self._flow_vec_valid = (dist <= float(self.cfg.sensor_range)) & (bear.abs() <= float(self._fov_half))

    def _publish_flow_vec(self) -> None:
        """``(u, v, rad, du, dv, validity)`` for this step -> ``self._flow_vec_now`` ``(N, 6)``.

        Normalised onto a comparable scale so one Linear layer can use all six:
            u / (W-1), v / (H-1), rad / 12  -> [0, 1]   (rad's divisor is its own clamp ceiling)
            du, dv                          -> [-1, 1]  (already displacement / W and / H)
            validity                        -> {0, 1}

        Why the extra validity slot: ``u``/``v`` are clipped into the image, and ``rad`` is
        clamped to a 1 px floor, so a target that has left the frustum is otherwise reported as
        "pinned to the border, one pixel wide" -- indistinguishable from a target actually sitting
        at the edge. The flag restores that distinction (and is what the reward's 30-step
        out-of-view reset keys on).

        `u` and `v` are the raster's image coordinates, whose span is ``0 .. W-1`` (the raster is
        drawn with ``cx = (W-1)/2``). Dividing by ``W`` and ``H`` would leave the top row and last
        column one pixel-share short of 1, so the exact span is used instead.

        Clipped to the space bounds declared in the cfg. The clip is redundant on the faithful
        path by construction, but `flow_vec_dv_gain` can scale du/dv past 1 and an observation
        outside its Box is the kind of thing that only surfaces much later, so it is enforced
        here. NOTE this means a gain > 1 is silently saturated in the observation; if a gain is
        ever used for real, raise the Box bounds with it.
        """
        w = float(int(self.cfg.cam_width) - 1)
        h = float(int(self.cfg.cam_height) - 1)
        gain = float(getattr(self.cfg, "flow_vec_dv_gain", 1.0))
        vec = torch.stack(
            [
                (self._flow_vec_u / w).clamp(0.0, 1.0),
                (self._flow_vec_v / h).clamp(0.0, 1.0),
                (self._flow_vec_rad / 12.0).clamp(0.0, 1.0),
                (self._flow_vec_du * gain).clamp(-1.0, 1.0),
                (self._flow_vec_dv * gain).clamp(-1.0, 1.0),
                self._flow_vec_valid.to(dtype=self._flow_vec_u.dtype),
            ],
            dim=-1,
        )
        self._flow_vec_now[:] = torch.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)

    def _teach_gate(self, rgb_nhwc: torch.Tensor) -> torch.Tensor:
        """Rectangle around the teach-template detection. Empty when the photo does not match."""
        if getattr(self, "_teach_template", None) is None and self._teach is not None:
            self._teach_template = cut_teach_template(self._teach.image)
        h, w = int(rgb_nhwc.shape[1]), int(rgb_nhwc.shape[2])
        gate = torch.zeros(self.num_envs, h, w, dtype=torch.bool, device=self.device)
        if getattr(self, "_teach_template", None) is None:
            return gate
        det = detect_teach(rgb_nhwc, self._teach_template).to(self.device)
        self._det = det
        cx, cy, bw, bh, _score = det.unbind(-1)
        ys = torch.arange(h, device=self.device).view(1, h, 1)
        xs = torch.arange(w, device=self.device).view(1, 1, w)
        pad = 8.0
        x0 = (cx - 0.5 * bw - pad).view(-1, 1, 1)
        x1 = (cx + 0.5 * bw + pad).view(-1, 1, 1)
        y0 = (cy - 0.5 * bh - pad).view(-1, 1, 1)
        y1 = (cy + 0.5 * bh + pad).view(-1, 1, 1)
        return (cx >= 0.0).view(-1, 1, 1) & (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)

    def _measure_render_flow(self, rgb_nhwc: torch.Tensor) -> None:
        """Flow is the teach-detection centre moving on the 256 image.

        The network receives only that 3-vector. A missed detection writes zeros.
        """
        curr = rgb_nhwc.permute(0, 3, 1, 2).contiguous()
        n, _, h, w = curr.shape
        if getattr(self, "_det_prev_ok", None) is None or self._det_prev_ok.shape[0] != n:
            self._det_prev_ok = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._det_prev_xy = torch.zeros(n, 2, device=self.device)
            self._det_prev_wh = torch.zeros(n, 2, device=self.device)
            self._hi_hold = torch.ones(n, dtype=torch.bool, device=self.device)
        hold = self._hi_hold
        if bool(hold.any()):
            self._det_prev_ok = self._det_prev_ok.clone()
            self._det_prev_ok[hold] = False
        if getattr(self, "_teach_template", None) is None and self._teach is not None:
            self._teach_template = cut_teach_template(self._teach.image)
        if getattr(self, "_teach_template", None) is None:
            det = curr.new_zeros(n, 5)
            det[:, 0] = -1.0
        else:
            det = detect_teach(
                rgb_nhwc,
                self._teach_template,
                scales=(1.0,),
            ).to(self.device)
        self._det = det
        cx, cy, bw, bh, _score = det.unbind(-1)
        hit = cx >= 0.0
        use = hit & self._det_prev_ok & ~hold
        du = torch.where(use, cx - self._det_prev_xy[:, 0], torch.zeros_like(cx))
        dv = torch.where(use, cy - self._det_prev_xy[:, 1], torch.zeros_like(cy))
        vec = torch.stack([du / float(w), dv / float(h), use.float()], dim=-1)
        self._flow_vec = vec
        self._flow_xs = cx.view(n, 1)
        self._flow_ys = cy.view(n, 1)
        self._flow_du = du.view(n, 1)
        self._flow_dv = dv.view(n, 1)
        self._flow_ok = hit.view(n, 1)
        self._det_prev_xy = torch.where(hit[:, None], torch.stack([cx, cy], dim=-1), self._det_prev_xy)
        self._det_prev_wh = torch.where(hit[:, None], torch.stack([bw, bh], dim=-1), self._det_prev_wh)
        self._det_prev_ok = hit
        new_alive = hit.view(n, 1)
        self._hi_curr = curr
        # #region agent log
        try:
            import json as _json
            import time as _time

            _det = getattr(self, "_det", None)
            _d0 = None if _det is None else [round(float(v), 3) for v in _det[0].tolist()]
            with open("/home/lh/IsaacLab/.cursor/debug-ffd96a.log", "a", encoding="utf-8") as _fh:
                _fh.write(
                    _json.dumps(
                        {
                            "sessionId": "ffd96a",
                            "runId": "flow-follow",
                            "hypothesisId": "A-F",
                            "location": "auv_tracking_v6_flowvec_env.py:_measure_render_flow",
                            "message": "teach detect",
                            "data": {
                                "hold": bool(hold[0].item()),
                                "det_cx_cy_bw_bh_score": _d0,
                                "alive_after": int(new_alive[0].sum()),
                                "flow": [round(float(v), 4) for v in vec[0].tolist()],
                            },
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass
        # #endregion
        self._hi_hold.zero_()

    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        super()._reset_idx(env_ids)
        # Base __init__ may call _reset_idx through DirectRLEnv before _setup_scene has run, so
        # the buffer is not guaranteed to exist yet.
        if not hasattr(self, "_stack_pending_reset"):
            return
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        self._stack_pending_reset[ids] = True
        if hasattr(self, "_hi_hold"):
            self._hi_hold[ids] = True
        if hasattr(self, "_gt_px_ok"):
            self._gt_px_ok[ids] = False
        if hasattr(self, "_flow_alive"):
            self._flow_alive[ids] = False
        if hasattr(self, "_det_prev_ok"):
            self._det_prev_ok[ids] = False
        if hasattr(self, "_extent_ref_ok"):
            self._extent_ref_ok[ids] = False
        if hasattr(self, "_oov_streak"):
            self._oov_streak[ids] = 0
        if hasattr(self, "_cam_locked"):
            self._cam_locked[ids] = False
            for i in ids.tolist():
                self._cam_templates[int(i)] = None
        # Camera is a child prim. Sync USD before reset()'s render so the first frame is not frozen.
        if getattr(self, "_robot_usd", None) is not None:
            self._sync_robot_usd_pose()

    def _update_belief_from_camera(self, env_ids: torch.Tensor) -> None:
        """S1 camera (r, α) → KF. Teacher-lock first photo at GT bearing; NCC + Isaac depth."""
        if not hasattr(self, "_tiled_camera"):
            return
        agent_pos, agent_yaw = self._agent_pose_2d()
        _z_gt, r_true, a_true = self.kf.get_observation(self._target_pos, agent_pos, agent_yaw)
        rgb = self._tiled_camera.data.output["rgb"]
        if rgb.dtype != torch.uint8:
            rgb = rgb.clamp(0, 255).to(torch.uint8)
        depth = self._tiled_camera.data.output.get("distance_to_image_plane")
        hfov = float(self.cfg.cam_hfov_deg)
        ids = env_ids.view(-1).long()

        gt_vis = (r_true <= float(self.cfg.sensor_range)) & (a_true.abs() <= self._fov_half)
        need_lock = ids[~self._cam_locked[ids] & gt_vis[ids]]
        for i in need_lock.tolist():
            ii = int(i)
            frame = rgb[ii].detach().cpu().numpy()
            dframe = None
            if depth is not None:
                dframe = depth[ii].detach().cpu().numpy()
            locked = teacher_lock_rgb(
                frame, alpha_gt_rad=float(a_true[ii].item()), hori_fov_deg=hfov, depth=dframe
            )
            if locked is None:
                continue
            self._cam_templates[ii] = locked[0]
            self._cam_locked[ii] = True
            z_seed = torch.stack([r_true[ii : ii + 1], a_true[ii : ii + 1]], dim=-1)
            self.kf.update(z_seed, agent_pos[ii : ii + 1], agent_yaw[ii : ii + 1], env_ids=ids.new_tensor([ii]))

        az_pred, r_pred = body_polar_from_xy(self.kf.state[:, :2], agent_pos, agent_yaw)
        r_s, a_s, valid, _ncc, _u, _v = detect_s1_rgb(
            rgb,
            templates=self._cam_templates,
            prior_azimuth=az_pred,
            prior_range=r_pred,
            prior_gate_deg=float(getattr(self.cfg, "sonar_prior_gate_deg", 20.0)),
            hori_fov_deg=hfov,
            alpha_only=False,
            length_m=float(self._cam_length_m),
            depth_image=depth,
        )
        if r_s.device != self.device:
            r_s = r_s.to(self.device)
            a_s = a_s.to(self.device)
            valid = valid.to(self.device)
        accept = (valid > 0.5) & (r_s <= float(self.cfg.sensor_range) + 1.0e-3) & (a_s.abs() <= self._fov_half)
        accept = accept & self._cam_locked
        self._observed[ids] = accept[ids]
        self._sonar_locked[ids] = accept[ids]
        obs_ids = ids[accept[ids]]
        if obs_ids.numel() > 0:
            z = torch.stack([r_s[obs_ids], a_s[obs_ids]], dim=-1)
            self.kf.update(z, agent_pos[obs_ids], agent_yaw[obs_ids], env_ids=obs_ids)

    def _sync_target_entity(self, env_ids: torch.Tensor | None = None) -> None:
        """Same as base, but yaw the target AUV along its heading.

        Holo v0 teleports the target with a rotation derived from its velocity; without this the
        WarpAUV stays at identity quaternion and appears to drift sideways.
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        n = len(env_ids)
        pose = torch.zeros(n, 7, device=self.device)
        pose[:, 0] = self._target_pos[env_ids, 0]
        pose[:, 1] = self._target_pos[env_ids, 1]
        pose[:, 2] = self.cfg.starting_depth
        yaw = self.target_motion.yaw[env_ids]
        zeros = torch.zeros_like(yaw)
        pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        vel = torch.zeros(n, 6, device=self.device)
        vel[:, 0] = self._target_state[env_ids, 2]
        vel[:, 1] = self._target_state[env_ids, 3]
        self._target.write_root_pose_to_sim(pose, env_ids)
        self._target.write_root_velocity_to_sim(vel, env_ids)
