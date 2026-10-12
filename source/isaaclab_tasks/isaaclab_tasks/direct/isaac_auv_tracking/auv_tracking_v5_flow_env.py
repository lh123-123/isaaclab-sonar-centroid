# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AUV tracking Direct-v5-flow: Direct-v4-cam with the actor's RGB stack replaced by GT optical flow.

Lineage
-------
``auv_tracking_env.py`` (Direct-v0, the "整合项目" belief-SAC task) is kept untouched as the
baseline. ``auv_tracking_v4_cam_env.py`` (Direct-v4-cam) is the copy we iterate on, and it is
frozen as the home of the ``v4_cam_sb3_sac_800k`` checkpoint. **This** module is a second copy,
made so that checkpoint keeps a live, byte-identical source next to the new experiment.

Three things differ from Direct-v0. (1) and (2) are inherited verbatim from Direct-v4-cam; (3) is
new here, and is the only difference from Direct-v4-cam:

1. **Target entity** — the ``r=0.35 m`` sphere becomes a second WarpAUV. The target's
   *motion* is unchanged: the same ``BatchHoloTargetMotion`` (RRT_2d path + first-order
   velocity lag + the episode soft-start / leash schedule) drives it, and the pose is still
   teleported onto the rigid object every physics tick by ``_sync_target_entity``.
2. **Sensing** — the OceanSim FLS is dropped. KF measurement is the camera path
   (``kf_measurement="cam"``). The camera render feeds that measurement ONLY; the render
   resolution does not bound the observation. The critic sees the 6-D state
   ``[r, alpha, log det(Sigma), observed, v_surge, yaw_rate]`` (R8: 4-D belief + own-body
   velocity so it no longer has to infer egomotion); the actor's input is described in (3).
3. **Actor observation (the change in this module)** — Direct-v4-cam fed the actor a
   ``(frame_stack * 3, 64, 64)`` uint8 RGB stack plus a separate 2-channel ``flow_image``, i.e.
   ``frame_stack * 3 + 2 = 14`` channels into the first convolution. Here (R8) the actor sees
   ONE ``(frame_stack * 2 + 1, 64, 64)`` float32 stack -- 5 channels at the default ``K=2``:
   two optical-flow frames (du/W, dv/H) plus a 1-channel teach-NCC detection-box mask that
   cures the static-target aliasing (flow zero + mask lit = holding a still target; both zero
   = lost it). The flow is RAFT-estimated from consecutive renders by default
   (``flow_source="raft"``); the analytic GT stamp is the ablation branch. ``flow_image`` as a
   separate obs key is gone. The shared encoder needs no code change: it reads
   ``observations["image"]`` and takes its input width from ``shape[0]``, and its
   uint8/``max > 1.5`` rescale guard provably skips float32 flow bounded by 1.0.

One optical-flow frame already carries velocity, so stacking K of them hands the actor the
*history* of that velocity rather than the motion signal itself. The stack is retained so that K
stays a free knob to compare against the RGB scheme.

Target cruise is 0.5 m/s (max 0.6). Surge maps ``a_v`` onto ``[-1.0, 1.0]`` m/s.
Yaw scale stays 0.5 rad/s. No bearing cone. R8 step reward (``_compute_step_reward``):
alive 1.0 + bearing 1.1 * exp(-(a/15deg)^2) + range 1.0 * exp(-(log s / 0.5)^2) where s is
the self-calibrated size ratio - action-delta 0.03. Lost target is handled by termination
(OOV streak / envelope), not by a penalty term. Target a2t at reset is fixed at 2 m
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
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from isaaclab_tasks.direct.sonar_centroid_track.auv_platform import AUV_SONAR_MOUNT_OFFSET_LOCAL

from .assets.warpauv import WARPAUV_CFG
from .assets.warpauv_cam import WARPAUV_CAM_CFG
from .agents.raft_flow import estimate_flow
from .auv_tracking_env import AUVTrackingEnv, AUVTrackingEnvCfg
from .cam_meas import detect_s1_rgb, focal_px, teacher_lock_rgb
from .sonar_meas import body_polar_from_xy
from .teach_match import (
    cut_teach_template,
    detect_points,
    detect_teach,
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
class AUVTrackingV5FlowEnvCfg(AUVTrackingEnvCfg):
    """v0 dynamics / reward / motion, camera-only observation, AUV target."""

    # --- (2) sensing: no sonar. KF default is geometric GT; "cam" is the appearance path. ---
    kf_measurement: str = "cam"
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
    # `_compute_step_reward` needs no change: with `rew_penalty_clip = 1.0` it clamps
    # range / bearing / action-delta at 1.0 exactly as the legacy code did, and the two new
    # terms below are weighted 0.0, so the arithmetic is identical to the 800k revision.
    # --- v5-flow first-principles reward (2026-10-09) ---
    # Actor只看到光流，它唯一能控制的是：把目标保持在画面中心（bearing→0）、
    # 保持在2m左右、不跟丢。因此奖励只回答三个问题：
    #   活着吗（alive）+ 对得准吗（bearing）+ 距离对吗（range）。
    # 删掉的东西以及原因：
    #   * uncertainty项（-logdet）：目标进FOV约3步就饱和到EKF地板，之后是常数，
    #     对actor是不可归因的常数偏置，只给critic看（state里已有），不进reward。
    #   * lost每步扣分：丢目标已由OOV直接终止，终止即丢掉未来全部alive，
    #     再每步扣是双重惩罚，且教policy“丢了不如早点死”。
    #   * OOV终止步-5：同理，截断未来回报已是最大惩罚，一次性-5只增加方差。
    #   * clip=1.0：把二次项封顶恰好制造了零梯度死区（曾经median漂到20°的原因）。
    #   * yaw_sat：bearing对准了自然不打满舵，无需单列。
    # 基类rolling-mean done（rew_done_threshold=-3.5）随之自然失效：
    # 新奖励恒≥0，均值不可能<-3.5，终止只剩超时+OOV，正是我们想要的。
    rew_scale_uncertainty_mean: float = 0.0
    rew_penalty_clip: float = 0.0
    rew_penalty_range: float = 0.0
    rew_range_sigma: float = 1.0
    rew_penalty_bearing: float = 0.0
    rew_penalty_lost: float = 0.0
    rew_penalty_action_delta: float = 0.0
    # v5 不消费这些基类罚分键（_compute_step_reward 不调 super）；显式置 0 只为
    # 防止 yaml/继承链把它们重新打开。OOV 终止罚分已彻底删除，见下方 oov_limit_steps 注释。
    rew_penalty_oov_terminal: float = 0.0
    # 新奖励三项（有界、处处可导、无死区）：
    rew_alive: float = 1.0            # 每存活一步
    rew_bearing_scale: float = 1.1    # R5：从R2的1.0小步加码（近距对准要求略严；1.3已证明过猛）
    rew_bearing_sigma_deg: float = 15.0  # bearing高斯宽度（≈FOV半角45°的1/3）
    rew_range_scale: float = 1.0      # R8：尺寸比项权重（s=1拿满，4m处s≈0.42只剩~0.05）
    rew_size_sigma: float = 0.5       # R8：log(s)高斯宽度；自标定，无fx/L
    # --- range-cue通道（阶段1，diag-B：corr(w,1/r)=0.86，div符号一致率0.87）---
    flow_cue_ema: float = 0.3         # bbox宽度EMA系数（平滑检测抖动）
    rew_penalty_action_delta_v5: float = 0.03  # R5：0.01→0.03（压bang-bang；0.05留待R3式恶化时再用）
    # 近场尺度说明（R4）：r<1m时bbox宽度顶到1.0饱和、RAFT大位移被obs clip±1截断——
    # 近场本就不需要距离cue做精调，靠加码后的bearing+alive即可；cue只在中远场起作用。
    # 2026-10-08 bearing rebalance (single-channel change on top of the 800k snapshot):
    # the quadratic cost saturates at bearing_sigma = 0.45 * fov_half = 20.25 deg (HFOV 90), and
    # clip=1.0 makes it constant beyond that -- zero gradient exactly where the 1M checkpoint
    # drifted (median 20.4 deg, corr(|bearing|, lost) = 0.92). Raising the weight to 0.5 steepens
    # the in-band gradient (-0.023 -> -0.039 rew/deg at 16 deg); the linear tail re-arms the
    # gradient past 15 deg (-0.02 rew/deg sustained out to the 90 deg FOV edge), where the
    # historical value was 0. Everything else (uncertainty 0.2, range 0.2, clip 1.0, lost 0.5)
    # is frozen so the effect is attributable -- the 09-28 multi-variable rebalance collapsed.
    rew_bearing_linear_deg: float = 10.0
    rew_penalty_bearing_linear: float = 0.05
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
    frame_stack: int = 2              # R8：4→2（单帧flow已是速度，历史冗余；buffer减半）
    # --- estimated-flow actor (RAFT-Small) ---
    # "gt" keeps the analytic disk stamp (ablation baseline); "raft" replaces it with
    # RAFT-Small dense flow computed from consecutive render-res RGB frames. The actor's
    # input SHAPE is unchanged either way: (frame_stack * 2, 64, 64) float32.
    flow_source: str = "raft"
    # Pixels whose RAFT magnitude is below this are treated as background. The scene water
    # is textureless, so the target is the only thing that moves coherently; a relative
    # term (0.4 * frame max) isolates the dominant blob when ego-motion is large.
    flow_mag_min_px: float = 0.5
    flow_mag_rel_frac: float = 0.4
    # Fewer lit pixels than this means "no target in view" (flow cannot reacquire, so the
    # episode ends instead of wandering blind -- see oov_limit_steps).
    flow_min_lit_px: int = 20
    # Consecutive steps with no flow-detected target before the episode terminates.
    oov_limit_steps: int = 10
    # NOTE: the one-time OOV terminal penalty was deliberately removed from v5's reward
    # (see the reward block above -- termination itself already drops all future alive
    # reward). The field is NOT consumed by _compute_step_reward here; do not re-add a
    # value and assume a penalty exists (v6_flowvec defines and consumes its own).
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
    # the frame stack as ``"image"`` while the *critic* consumes the 6-D state (4-D KF belief +
    # own-body velocity, R8) as ``"state"``.
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
    # The actor input is a stack of ``frame_stack`` optical-flow frames, 2 channels each
    # (du/W, dv/H of the target's pixel displacement; RAFT-estimated under
    # ``flow_source="raft"``, analytic GT in the ablation branch), plus the 1-channel
    # orange-appearance mask (R8: cures the static-target aliasing), so
    # ``frame_stack * 2 + 1`` channels into the first convolution -- 5 at the default K=2.
    #
    # float32, NOT uint8: optical flow is signed and lives in [-1, 1], so the RGB "1 byte/pixel"
    # trick does not apply. Three consequences fall out of that, all of them wanted:
    #   * SB3's ``is_image_space`` requires uint8 + [0, 255], so this is a plain Box and
    #     ``preprocess_obs`` leaves it alone (no hidden /255).
    #   * The shared ``cam_encoder.MidCNNExtractor`` guard only divides when ``dtype == uint8``
    #     or ``float(max) > 1.5``, so it too leaves the flow alone.
    #   * Replay memory is 4 bytes/pixel/step instead of 1 -- see the buffer size in the
    #     matching ``sb3_sac_v5_flow_cfg.yaml``.
    #
    # The key is still called ``image`` so the encoder and every existing consumer keep working
    # unchanged; only its content is now a flow stack. Nothing here depends on the camera's render
    # resolution: the render is only used for the KF measurement (``_update_belief_from_camera``).
    observation_space = {
        # Flow stack (frame_stack*2 ch, oldest first) + 1ch外观mask（R8：治actor静止aliasing）。
        # 5ch @K=2。flow零+mask中央亮=跟住静止；flow零+mask黑=跟丢静止。
        "image": gym.spaces.Box(
            low=-1.0, high=1.0, shape=(frame_stack * 2 + 1, cam_height, cam_width), dtype=np.float32
        ),
        "state": gym.spaces.Box(
            # [r, alpha, logdet, observed] + 本体[v_surge, yaw_rate]（R8：critic不再猜自运动）
            low=np.array([0.0, -np.pi, -50.0, 0.0, -2.0, -2.0], dtype=np.float32),
            high=np.array([600.0, np.pi, 50.0, 2.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        ),
    }


class AUVTrackingV5FlowEnv(AUVTrackingEnv):
    """Direct-v4-cam's sibling: the actor sees a K-frame GT optical-flow stack instead of RGB.

    The second-AUV target, its motion model, the camera, the camera-based KF measurement, the
    rewards, the reset schedule and the control chain are all Direct-v4-cam verbatim -- and
    therefore Direct-v0 -- so a run here is directly comparable to the ``v4_cam_sb3_sac_800k``
    baseline apart from the actor's input representation.
    """

    cfg: AUVTrackingV5FlowEnvCfg

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
        # Allocate the flow stack here: _setup_scene runs before the first _reset_idx, and
        # self.device / self.num_envs are already final. float32, not uint8: flow is signed.
        k = int(self.cfg.frame_stack)
        self._flow_buf = torch.zeros(
            (self.num_envs, k * 2, self.cfg.cam_height, self.cfg.cam_width),
            dtype=torch.float32,
            device=self.device,
        )
        self._teach: TeachRef | None = None
        self._teach_template: torch.Tensor | None = None
        self._teach_vec = torch.zeros(self.num_envs, 4, device=self.device)
        h, w = int(self.cfg.cam_height), int(self.cfg.cam_width)
        self._flow_image = torch.zeros((self.num_envs, 2, h, w), device=self.device)
        # bbox测距EMA（reward用）：orange宽度/256，无效帧保持，reset清零。
        self._cue_w = torch.zeros(self.num_envs, device=self.device)
        self._cue_w_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # teach尺子（R8）：reset后首次有效bbox即2m/0°初始帧宽度；尺寸比s=w/teach。
        self._teach_w = torch.zeros(self.num_envs, device=self.device)
        self._teach_w_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._size_ratio = torch.ones(self.num_envs, device=self.device)
        # 外观平面（R8）：橙mask下采样到cam-res，actor第5通道。接口按teach框预留，
        # 当前实现用橙mask（tint脚手架，去橙化时换teach框mask，形状不变）。
        self._app_plane = torch.zeros((self.num_envs, 1, h, w), device=self.device)
        self._gt_px = torch.zeros(self.num_envs, 2, device=self.device)
        self._gt_px_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._flow_vec = torch.zeros(self.num_envs, 3, device=self.device)
        # RAFT branch state: previous render-res RGB frame plus the per-step flow-derived
        # geometry estimate (range/bearing/validity) that replaces GT in the reward.
        rh, rw = int(self.cfg.render_height), int(self.cfg.render_width)
        self._prev_rgb = torch.zeros((self.num_envs, rh, rw, 3), dtype=torch.uint8, device=self.device)
        self._prev_rgb_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._flow_est_r = torch.zeros(self.num_envs, device=self.device)
        self._flow_est_a = torch.zeros(self.num_envs, device=self.device)
        self._flow_est_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
            print("[v5-flow] WARN: target_tint set but target spawn does not support visual_material", flush=True)
            return
        # PreviewSurfaceCfg is a VisualMaterialCfg, i.e. exactly the type the field expects.
        spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(c) for c in tint))
        # Deliberately no "tinted" success line here. Assigning the spawn cfg only *requests* the
        # material; the actual binding happens later inside spawn_from_usd, and it has to beat the
        # UsdPreviewSurface already bound to mesh_0 *inside* the `visuals` instance prototype. That
        # is precisely the kind of override that fails without raising. _verify_target_tint() reads
        # back what really landed on the stage instead of assuming this line means anything.
        print(f"[v5-flow] target tint requested {tuple(float(c) for c in tint)} (verified post-spawn)", flush=True)

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
            print(f"[v5-flow] WARN: target prim {root_path} missing; cannot verify tint", flush=True)
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
            f"[v5-flow] target tint {'OK' if ok else 'NOT CONFIRMED'}: rel={targets or 'ABSENT'}"
            f" bound_diffuse={bound_colour} want={want} strength={strength}"
            f"{'' if strength_known else ' (absent -> USD default)'}",
            flush=True,
        )
        if not ok:
            print(
                "[v5-flow] WARN: the tint may be shadowed by the prototype binding on mesh_0"
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
            print("[v5-flow] WARN: target spawn has no 'scale'; target left at its authored size", flush=True)
            return
        spawn.scale = (scale, scale, scale)
        print(f"[v5-flow] target AUV scaled x{scale:g}", flush=True)
    def _teach_npz_path(self) -> Path:
        # this file: source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/...
        root = Path(__file__).resolve().parents[5]
        # v5 专属 teach（目标已染橙 0.85/0.38/0.12，与 v4 未染色的 teach_v4 外观不匹配，
        # 混用会把 ZNCC 压到 ~0.53——gate run2 实测）。npz 缺失时 _capture_teach 自捕获。
        return root / "models" / "auv_tracking" / "teach_v5_2m_0deg.npz"

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
        self._teach_template = cut_teach_template(self._teach.image)

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
        self._teach_template = cut_teach_template(self._teach.image)

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
                print(f"[v5-flow] WARN: camera prim missing, intrinsics not applied: {path}", flush=True)
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
            print(f"[v5-flow] WARN: USD pose sync disabled, could not open {robot_paths[:1]}: {exc}", flush=True)

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
        """First-principles tracking reward: alive + centered + well-ranged.

        actor能影响的只有三件事——别跟丢（活着）、把目标放中心（bearing→0）、
        保持2m。每项有界、处处可导：最优时≈1+1+0.3=2.3/步，
        丢目标时episode终止，未来alive全丢，不再另设扣分。
        几何量在raft下取RAFT估计（可部署），gt下取真值（ablation）。
        基类super()故意不调：它的uncertainty常数偏置+碰撞min(0,rew)会
        把这里的有界 shaping 冲掉；碰撞仍由_compute_is_col记录但不进奖励。
        """
        agent_pos, agent_yaw = self._agent_pose_2d()
        if str(getattr(self.cfg, "flow_source", "gt")) == "raft":
            r, a = self._flow_est_r, self._flow_est_a
        else:
            _, r, a = self.kf.get_observation(self._target_pos, agent_pos, agent_yaw)

        sig_b = max(float(getattr(self.cfg, "rew_bearing_sigma_deg", 15.0)) / 57.29578, 1.0e-6)
        r_bearing = torch.exp(-((a / sig_b) ** 2))
        # R8尺寸比距离：s=1拿满，4m处s≈0.42只剩~0.05；自标定，无fx/L。
        # r仍用bbox几何供包络/他处，reward距离语言只认s。
        sig_s = max(float(getattr(self.cfg, "rew_size_sigma", 0.5)), 1.0e-6)
        r_size = torch.exp(-((torch.log(self._size_ratio.clamp(min=0.05)) / sig_s) ** 2))
        smooth = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)

        rew = (
            float(self.cfg.rew_alive)
            + float(self.cfg.rew_bearing_scale) * r_bearing
            + float(self.cfg.rew_range_scale) * r_size
            - float(getattr(self.cfg, "rew_penalty_action_delta_v5", 0.01)) * smooth
        )
        return torch.nan_to_num(rew, nan=0.0, posinf=2.5, neginf=0.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Base terminations plus the out-of-view reset (RAFT branch only).

        Estimated flow has no reacquisition ability: once the target leaves the view there
        is no motion blob to steer back to, so ``oov_limit_steps`` consecutive invalid
        frames end the episode. Under ``flow_source="gt"`` this is disabled (GT always
        knows where the target is, so the run stays comparable with older baselines).
        """
        terminated, truncated = super()._get_dones()
        if str(getattr(self.cfg, "flow_source", "gt")) == "raft" and hasattr(self, "_oov_streak"):
            terminated = terminated | (self._oov_streak >= int(self.cfg.oov_limit_steps))
        # 硬边界：目标出传感器包络（4m/±45°，GT判定）立刻重置，不等10帧计数。
        # 估计器在远场会编假blob，计数门靠不住；包络是任务定义，以真值为准。
        agent_pos, agent_yaw = self._agent_pose_2d()
        _, gt_r, gt_a = self.kf.get_observation(self._target_pos, agent_pos, agent_yaw)
        out_of_envelope = (gt_r > float(self.cfg.sensor_range)) | (gt_a.abs() > float(self._fov_half))
        if str(getattr(self.cfg, "flow_source", "gt")) == "raft":
            terminated = terminated | out_of_envelope
        return terminated, truncated

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

        # Flow for this step -> self._flow_image (N, 2, H, W). Source is the cfg switch:
        # "gt" stamps the analytic target disk (ablation baseline), "raft" estimates dense
        # flow from consecutive renders. Either way this runs exactly once per step.
        if str(getattr(self.cfg, "flow_source", "gt")) == "raft":
            self._update_est_flow_image()
        else:
            self._update_gt_flow_image()

        # .clone(): this mask is read twice below and the assignment below mutates the source.
        pending = self._stack_pending_reset.clone()
        if bool(pending.any()):
            # Fresh episode: flow has no history, so the flow update left _flow_image at
            # zero for these envs and the whole stack legitimately starts as zeros.
            self._flow_buf[pending] = self._flow_image[pending].repeat(1, k, 1, 1)
            self._stack_pending_reset[pending] = False

        live = ~pending
        if bool(live.any()):
            # Shift by one 2-channel flow frame and append the newest.
            self._flow_buf[live, :-2] = self._flow_buf[live, 2:].clone()
            self._flow_buf[live, -2:] = self._flow_image[live]

        obs["policy"] = {
            "image": torch.cat([self._flow_buf, self._app_plane], dim=1),
            "state": torch.cat(
                [
                    belief_state,
                    self._ego_vel_state(),
                ],
                dim=-1,
            ),
        }
        return obs

    def _ego_vel_state(self) -> torch.Tensor:
        """本体速度（R8）：v_surge + yaw_rate，world→body。critic专用，actor看不见。"""
        _, agent_yaw = self._agent_pose_2d()
        vel_w = self._robot.data.root_lin_vel_w[:, :2]
        cy, sy = torch.cos(agent_yaw), torch.sin(agent_yaw)
        v_surge = cy * vel_w[:, 0] + sy * vel_w[:, 1]
        yaw_rate = self._robot.data.root_ang_vel_w[:, 2]
        return torch.stack([v_surge.clamp(-2.0, 2.0), yaw_rate.clamp(-2.0, 2.0)], dim=-1)

    def _update_est_flow_image(self) -> None:
        """RAFT-Small dense flow -> actor frame + flow-derived geometry.

        Runs RAFT on (prev render, current render), area-downsamples to cam res with the
        pixel-unit rescale, and writes (du/W, dv/H) into ``self._flow_image`` -- the same
        tensor and units the GT branch writes, so the stack, the encoder and the Box
        bounds are all untouched.

        Geometry for the reward comes from the same estimate, not from GT: the lit blob's
        centroid is the target pixel (u, v) and its area gives the disk radius, which
        inverts to range via rad = 0.5 * length / dist * fx. No lit blob (fewer than
        ``flow_min_lit_px`` pixels) means the target is out of view: ``_flow_est_ok`` is
        False and the OOV streak advances (see ``_get_dones``). Reset frames are zero
        (no previous frame yet) and report invalid.
        """
        curr = self._camera_rgb()
        fresh = ~self._prev_rgb_ok
        h, w = int(self.cfg.cam_height), int(self.cfg.cam_width)
        if self._teach is None and int(self.episode_length_buf[0]) >= 5:
            # 首帧自捕获会踩中 spawn 位姿（目标还在海床上、未同步到2m/0°悬浮reset位姿，
            # gate run3 实测：模板变俯视海床版，ZNCC 锁背景，ncc_only 方位误差 27°）。
            # 延后 5 步（soft-start 3s 内目标位移 cm 级，仍是标称 2m/0°）。
            self._capture_teach(curr)
        if bool((~fresh).any()):
            est = estimate_flow(self._prev_rgb[~fresh], curr[~fresh])
            rh = float(curr.shape[1])
            small = torch.nn.functional.interpolate(est, size=(h, w), mode="area") * (h / rh)
            du, dv = small[:, 0], small[:, 1]
            mag = torch.sqrt(du * du + dv * dv)
            thresh = torch.clamp(
                mag.amax(dim=(-2, -1), keepdim=True) * float(self.cfg.flow_mag_rel_frac),
                min=float(self.cfg.flow_mag_min_px),
            )
            mask = mag > thresh
            lit = mask.sum(dim=(-2, -1))
            ok = lit >= int(self.cfg.flow_min_lit_px)

            self._flow_image.zero_()
            self._flow_image[~fresh] = torch.stack(
                [du / w, dv / h], dim=1
            ).clamp(-1.0, 1.0)

            xs = torch.arange(w, device=self.device, dtype=torch.float32)
            mx = (mask * xs.view(1, 1, w)).sum(dim=(-2, -1)) / lit.clamp(min=1)
            # --- teach NCC 测距（render-res检测框；去橙化 2026-10-12） ---
            # flow面积在远处系统性偏大（r_hat谎报致远），改用NCC框宽度反推距离；
            # 橙mask量测整体换成teach模板匹配（detect_teach：0.65阈值+贴边拒检+尺度门）。
            if getattr(self, "_teach_template", None) is None and self._teach is not None:
                self._teach_template = cut_teach_template(self._teach.image)
            if self._teach_template is not None:
                det = detect_teach(curr, self._teach_template).to(self.device)
            else:
                det = torch.zeros((curr.shape[0], 5), dtype=torch.float32, device=self.device)
                det[:, 0] = -1.0
            self._det = det
            bw = det[:, 2]
            rw_f = max(float(curr.shape[2]), 1.0)
            ema = float(getattr(self.cfg, "flow_cue_ema", 0.3))
            w_now = (bw / rw_f).clamp(0.0, 1.0)
            hit = det[:, 0] >= 0
            first = hit & ~self._cue_w_ok
            self._cue_w = torch.where(
                first, w_now, torch.where(hit, ema * w_now + (1.0 - ema) * self._cue_w, self._cue_w)
            )
            self._cue_w_ok = self._cue_w_ok | hit
            # teach：本episode首次有效bbox即初始帧尺子（2m/0°同姿态reset保证）
            teach_first = hit & ~self._teach_w_ok
            self._teach_w = torch.where(teach_first, w_now, self._teach_w)
            self._teach_w_ok = self._teach_w_ok | hit
            # 尺寸比（R8距离语言）：teach无效时s=1中性；钳死防log炸
            self._size_ratio = torch.where(
                self._teach_w_ok,
                (self._cue_w / self._teach_w.clamp(min=1e-3)).clamp(0.05, 4.0),
                torch.ones_like(self._cue_w),
            )
            fx_r = float(self.cfg.render_width) * 0.5 / float(np.tan(np.radians(float(self.cfg.cam_hfov_deg)) * 0.5))
            length = 0.8 * float(getattr(self.cfg, "target_scale", 1.0))
            # 针孔模型：w_px = L*fx/dist → dist = L*fx/w_px（w用EMA平滑值，防抖）
            w_px = (self._cue_w * rw_f).clamp(min=2.0)
            r_bbox_full = (length * fx_r / w_px).clamp(0.4, 13.3)
            # 远场拍平钳制（range_shots实测：4~5m bbox 16~17px基本不缩小）：平滑宽度<10px
            # 说明目标已到地板区，反推无意义，直接报far（sensor_range），让铰链/OOV接管。
            r_bbox_full = torch.where(
                (self._cue_w * rw_f) < 10.0,
                torch.full_like(r_bbox_full, float(self.cfg.sensor_range)),
                r_bbox_full,
            )
            area = lit.to(torch.float32)
            rad = torch.sqrt(area / torch.pi).clamp(1.0, 12.0)
            fx = fx_r * (w / float(self.cfg.render_width))
            r_area = (0.5 * length * fx / rad).clamp(0.4, 13.3)
            cx = 0.5 * (w - 1)
            a_hat = torch.atan((cx - mx) / max(fx, 1e-6))

            idx = torch.where(~fresh)[0]
            # 诚实距离：本步bbox命中才用bbox（EMA已含新鲜值），否则回退面积法；
            # flow无效且bbox无效时下面回退4.0认瞎。EMA跨无效帧的hold只做平滑，不做真值。
            hit_sub = hit[idx]
            r_hat = torch.where(hit_sub, r_bbox_full[idx], r_area)
            # NCC框中心（render-res→cam-res）→ bearing备选：flow无但模板命中时用（治静止致盲）
            a_det_full = torch.atan((cx - det[:, 0] * (w / rw_f)) / max(fx, 1e-6))
            # 外观平面：teach检测框（render-res，pad 8px）→ cam-res（area）→ actor第5通道
            ys_r = torch.arange(int(curr.shape[1]), device=self.device, dtype=torch.float32).view(1, -1, 1)
            xs_r = torch.arange(int(curr.shape[2]), device=self.device, dtype=torch.float32).view(1, 1, -1)
            pad = 8.0
            box = (
                (det[:, 0] >= 0).view(-1, 1, 1)
                & (xs_r >= (det[:, 0] - 0.5 * det[:, 2] - pad).view(-1, 1, 1))
                & (xs_r <= (det[:, 0] + 0.5 * det[:, 2] + pad).view(-1, 1, 1))
                & (ys_r >= (det[:, 1] - 0.5 * det[:, 3] - pad).view(-1, 1, 1))
                & (ys_r <= (det[:, 1] + 0.5 * det[:, 3] + pad).view(-1, 1, 1))
            )
            app_small = (
                torch.nn.functional.interpolate(box.float().unsqueeze(1), size=(h, w), mode="area") > 0.3
            ).float()
            self._app_plane.zero_()
            self._app_plane[~fresh] = app_small[~fresh]
            self._flow_est_ok[idx] = ok
            self._flow_est_r[idx] = torch.where(
                ok, r_hat, torch.where(hit_sub, r_bbox_full[idx], torch.full_like(r_hat, float(self.cfg.sensor_range)))
            )
            self._flow_est_a[idx] = torch.where(
                ok, a_hat, torch.where(hit_sub, a_det_full[idx], torch.full_like(a_hat, float(self._fov_half)))
            )
            self._flow_image[fresh] = 0.0
            self._flow_est_ok[fresh] = False
            # 存在性判据（治静止致盲）：flow或外观任一命中即存在，只有全无才涨OOV
            exist_sub = ok | hit_sub
            self._oov_streak[idx] = torch.where(exist_sub, torch.zeros_like(self._oov_streak[idx]), self._oov_streak[idx] + 1)
        else:
            self._flow_image.zero_()
            self._flow_est_ok[:] = False
        self._prev_rgb[:] = curr
        self._prev_rgb_ok[:] = True
        if bool(fresh.any()):
            self._oov_streak[fresh] = 0
    def _update_gt_flow_image(self) -> None:
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
        # GT分支外观平面：真值disk即mask（ablation上限）
        self._app_plane.zero_()
        self._app_plane[:, 0] = mask.float()

    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        super()._reset_idx(env_ids)
        # 初始位形：目标定在正前方2m/0°（cfg），姿态与追踪机一致（正视后视角teach）。
        # base把目标yaw设成首航向（朝waypoint），这里改写成agent yaw并重同步实体。
        if True:
            ids_t = torch.as_tensor(
                self._robot._ALL_INDICES if env_ids is None else env_ids,
                device=self.device,
                dtype=torch.long,
            ).view(-1)
            _, _, ay = euler_xyz_from_quat(self._default_root_state[ids_t, 3:7])
            self.target_motion.yaw[ids_t] = ay
            self._sync_target_entity(ids_t)
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
        if hasattr(self, "_prev_rgb_ok"):
            self._prev_rgb_ok[ids] = False
        if hasattr(self, "_flow_est_ok"):
            self._flow_est_ok[ids] = False
        if hasattr(self, "_cue_w"):
            self._cue_w[ids] = 0.0
            self._cue_w_ok[ids] = False
        if hasattr(self, "_teach_w"):
            self._teach_w[ids] = 0.0
            self._teach_w_ok[ids] = False
            self._size_ratio[ids] = 1.0
        if hasattr(self, "_app_plane"):
            self._app_plane[ids] = 0.0
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
        # 特权critic（R6回滚结论）：teacher_lock用真值播种KF是Holo v1标准做法，
        # 部署时critic本来就扔掉，只有纯净actor上机。R5去GT锁导致训练初期belief全错→
        # 死亡螺旋（bearing 36.5°），证明锁定瞬间需要这个“睁眼”。之后全是NCC测量。
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
