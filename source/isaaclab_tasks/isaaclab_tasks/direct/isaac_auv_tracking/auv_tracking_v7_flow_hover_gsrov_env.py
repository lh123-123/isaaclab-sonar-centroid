# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AUV station keeping: global optical-flow nulling (Direct-v7-flow-hover).

No target object: the vehicle holds position by driving the whole-image optical flow
to zero. Camera is upright (no 180 deg rotation). Uses the GS-ROV 8-thruster hull
(thruster_layout="gsrov8"); real dynamics parameters (mass/inertia/thruster coords/
thrust curves) are baked into the cfg (see thruster_dynamics.py).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.configclass import field
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, quat_from_euler_xyz
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_env import (
    AUVTrackingEnv,
    AUVTrackingEnvCfg,
    OceanCurrentCfg,
)
from isaaclab_tasks.direct.isaac_auv_tracking.template_anchor import update_template_anchor
from isaaclab_tasks.direct.isaac_auv_tracking.assets.gsrov import DEFAULT_MASS, gsrov_robot_cfg
from isaaclab_tasks.direct.residual_rl.thruster_allocation import allocate_thrusters

_CAM_HFOV_DEG = 81.65
_CAM_APERTURE_MM = 20.955


def _farneback_flow(prev_rgb: torch.Tensor, curr_rgb: torch.Tensor) -> torch.Tensor:
    """Farneback optical flow returning (N, 2, H, W) in pixels.

    Uses OpenCV OpenCL (UMat) when available — on the RTX 3090 this is ~3.5x faster
    than the pure-CPU path with <1e-3 px difference, and avoids the torchvision/RAFT
    dependency. Falls back to CPU automatically if OpenCL is unavailable.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "cv2 (opencv-python) is required for Farneback flow. Install with: pip install opencv-python"
        )
    use_ocl = False
    try:
        cv2.ocl.setUseOpenCL(True)
        use_ocl = bool(cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL())
    except Exception:
        use_ocl = False

    n = prev_rgb.shape[0]
    frames_prev = prev_rgb.cpu().numpy()   # (N, H, W, 3) uint8
    frames_curr = curr_rgb.cpu().numpy()
    h, w = frames_prev.shape[1], frames_prev.shape[2]
    flows = np.empty((n, 2, h, w), dtype=np.float32)

    for i in range(n):
        if use_ocl:
            pu = cv2.UMat(frames_prev[i])
            cu = cv2.UMat(frames_curr[i])
            g_prev = cv2.cvtColor(pu, cv2.COLOR_RGB2GRAY)
            g_curr = cv2.cvtColor(cu, cv2.COLOR_RGB2GRAY)
            flow_hwc = cv2.calcOpticalFlowFarneback(
                g_prev, g_curr, None,
                pyr_scale=0.5, levels=3, winsize=21,
                iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
            ).get()  # UMat -> (H, W, 2) float32
        else:
            g_prev = cv2.cvtColor(frames_prev[i], cv2.COLOR_RGB2GRAY)
            g_curr = cv2.cvtColor(frames_curr[i], cv2.COLOR_RGB2GRAY)
            flow_hwc = cv2.calcOpticalFlowFarneback(
                g_prev, g_curr, None,
                pyr_scale=0.5, levels=3, winsize=21,
                iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
            )
        flows[i] = flow_hwc.transpose(2, 0, 1)  # -> (2, H, W) channel-first
    return torch.from_numpy(flows).to(prev_rgb.device).float()


def _estimate_flow(prev_rgb: torch.Tensor, curr_rgb: torch.Tensor, method: str, device: str) -> torch.Tensor:
    """Unified flow estimator: 'farneback' (CPU) or 'raft' (GPU).

    Both return (N, 2, H, W) float32 in render-space pixels.
    """
    if method == "raft":
        from isaaclab_tasks.direct.isaac_auv_tracking.agents.raft_flow import estimate_flow as _raft_estimate
        # RAFT expects uint8 on the target device
        if prev_rgb.device != torch.device(device):
            prev_rgb = prev_rgb.to(device)
            curr_rgb = curr_rgb.to(device)
        return _raft_estimate(prev_rgb, curr_rgb)
    elif method == "farneback":
        return _farneback_flow(prev_rgb, curr_rgb)
    else:
        raise ValueError(f"Unknown flow_method: {method!r}, choose 'raft' or 'farneback'")


@configclass
class AUVTrackingV7FlowHoverGsrovEnvCfg(AUVTrackingEnvCfg):
    """Station keeping via global optical-flow nulling (GS-ROV 8-thruster)."""

    # No sonar / no target belief.
    kf_measurement: str = "gt"
    # Fixed AUV starting depth: ROV centered at world origin (0,0,0).
    # Pool walls span z=[-2, 2], so starting_depth=0.0 puts the ROV at the pool's vertical center.
    starting_depth: float = 0.0

    # Agent rides the camera-enabled asset copy.
    # Default = lightweight cuboid proxy (the rectangle currently used for training/UI).
    # Set use_real_usd=True to load the prepared GS-ROV CAD hull instead
    # (produced by rov/tools/prepare_gsrov_usd.py: centred on COM, mass/inertia baked in).
    use_real_usd: bool = False
    robot_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.56, 0.43, 0.30),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            # Keep the proxy rigid-body mass equal to cfg.mass (13.15 kg) so it matches the
            # hydrodynamic model; a mismatched pair makes buoyancy/weight inconsistent.
            mass_props=sim_utils.MassPropertiesCfg(mass=13.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )

    # Policy frequency: 15 Hz.  decimation=8 at sim.dt=1/120 → 1/(1/120*8)=15 Hz.
    decimation: int = 8
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=8)
    # Policy actions: [surge, sway, yaw] ∈ [-1, 1]. Depth is NOT action-controlled:
    # the retuned PI depth loop below holds starting_depth (= the first-frame depth)
    # on its own, so a 4th depth-offset channel was tried and dropped — with the
    # loop fixed it carried zero learnable signal (the policy cannot speed up the
    # integrator with a setpoint shift) and only added exploration noise.
    action_space: gym.spaces.Box = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    # Depth-loop retune, v7-LOCAL (applied post-hoc in __init__; the class defaults in
    # batch_pid.py stay untouched so v0/v4/v5/v6 keep the old P+D law).
    #   Why: the old P-only loop (kp=5) equilibrated 0.4 m LOW of target against the
    #   ~2 N residual weight — the 30k/300k models all hovered 0.43 m deep, capping
    #   anchor_sim at ~0.75 (image scale drift). Adding an integral alone was NOT
    #   enough: the char. poly m·s³+kd·s²+kp·s+ki is Routh-stable only if
    #   kd·kp > m_eff·ki, and with the old kp=5, kd=3 (product 15) any useful ki
    #   (≥0.6) diverges — ki=4 produced a bounded ±7 cm limit cycle that never
    #   settled (verified in a zero-action run). Raising kp/kd first moves the
    #   ceiling to kd·kp = 6×12 = 72, leaving ki=1.5 a 1.8x margin (26×1.5 = 39).
    #   Validated: zero-action dip to the −6 cm deadband edge, integral clears the
    #   bias in ~12 s, episode end z = +1.2 cm, monotone convergence.
    #   heave clamp 3 → 6 N: transient P+I demand exceeds 3 N; the four verticals
    #   deliver ~70 N so this is trivial headroom, and attitude moments never came
    #   near the old 3 N·m cap anyway.
    depth_kp: float = 12.0
    depth_ki: float = 1.5
    depth_kd: float = 6.0
    heave_wrench_max: float = 6.0

    # Work attitude: hold LEVEL (roll=pitch=0) with continuous thruster moment. The
    # free-floating trim (~20° nose-up) is the zero-thrust attitude; holding level
    # instead requires the verticals to carry the buoyancy pitch couple
    # ρ·g·V·dx ≈ 997·9.81·0.01295·0.040 ≈ 5.07 N·m — about 56% of the verticals'
    # 9.1 N·m pitch authority — for the whole episode. That is exactly the handover
    # state the policy should inherit: someone flew the ROV to the work site at level
    # attitude with the thrusters already holding it, frame 1 is that scene, and the
    # policy keeps it while fighting the current.
    # att_ki adds attitude integral action. P+D alone cannot hold a pose away from
    # the free-floating trim, so the integrator is pre-charged at reset with the
    # wrench that actually realizes the buoyancy couple. Dividing N·m by T_max
    # (≈ 0.26) is short: allocation solves a linear command, while thrust is
    # T_max·u·|u| and reverse is weaker than forward, so the same physical 5.07 N·m
    # needs a larger command (≈ 0.35). The init solve finds that command, and the
    # heave command that cancels the ~2.4 N residual weight, and both integrators
    # are pre-charged. att_int_limit 0.6 stays above that command and below saturation.
    # NOTE: with this on, checkpoints trained on the trim spawn are no longer
    # directly comparable (spawn attitude, camera geometry, and att reward reference
    # all change). Set False to recover the legacy trim behavior.
    level_attitude: bool = True
    att_ki: float = 1.0
    att_int_limit: float = 0.6

    # Episode cap 15 s (225 steps @ 15 Hz). The parent's 40 s → 600 steps, but observed
    # behavior resolves within ~4–13 s (the ROV either holds station or drifts into the
    # textured plane and goes blind); a shorter cap gives denser episode statistics and
    # faster reset diversity. SAC's bootstrap horizon (gamma 0.99 ≈ 100 steps ≈ 6.7 s)
    # is well below 225, so value bootstrapping is unaffected.
    episode_length_s: float = 15.0

    # Velocity-command scale: unit policy action → cmd_vel target in m/s (surge/sway)
    # and rad/s (yaw). 0.75 covers the ocean-current peak (0.4 m/s × 1.6 modulation =
    # 0.64 m/s) with ~15% margin. The old 0.5 could not counter the modulation peaks,
    # so even a perfect policy was pushed off station whenever the current surged —
    # reward the policy could not control, followed by drift into the textured plane.
    cmd_vel_scale: float = 0.75

    # Camera (upright, no rotation). 640×360 render → 64×64 policy input.
    # 640×360 is the sweet spot: ~13 Hz RL freq (vs 10.5 Hz at 1280×720) while keeping
    # enough texture detail for flow-based hover learning. Benchmarked on RTX 3090, 8 envs.
    cam_height: int = 64
    cam_width: int = 64
    render_height: int = 360
    render_width: int = 640
    # Resolution fed into Farneback for optical flow estimation. Final flow is still
    # downsampled to (cam_height, cam_width)=64x64 for the policy, so running Farneback
    # at full render res (1280x720) is pure waste — we downsample first for ~16x speedup
    # with negligible quality loss (empirically: RMS differs <1 px at 320x180 vs 1280x720).
    # H and W must be divisible by 8 (RAFT requirement); 176×320 gives ~17.8× fewer pixels
    # than 1280×720 while keeping the aspect ratio close to 16:9.
    flow_render_height: int = 176
    flow_render_width: int = 320
    cam_hfov_deg: float = _CAM_HFOV_DEG
    cam_aperture_mm: float = _CAM_APERTURE_MM
    frame_stack: int = 1
    cam_sync_usd_pose: bool = True

    # Extra plain (non-tiled) camera for eyeballing the ROV's forward view in the viewport,
    # or for recording. Same pose and intrinsics as ForwardCamera0, but full resolution and
    # not wired into observations, so it costs nothing during training. Its pose is copied
    # from the flow camera (front face of the cuboid, looking +X); set the offsets below to
    # move it somewhere else.
    spawn_view_camera: bool = True
    view_camera_name: str = "ViewCamera0"
    view_camera_offset: tuple = (0.28, 0.0, 0.0)  # body frame, metres (0.28 = front face)
    view_camera_roll_deg: float = -90.0  # -90 matches the flow camera's upright orientation

    # Flow (Farneback @ 1280×720 → 64×64, 20 Hz policy chain).
    # flow_method="raft"  uses GPU-accelerated RAFT-Small (torchvision), ~10x faster on RTX 3090.
    # flow_method="farneback" uses CPU-based OpenCV Farneback (matches real-robot inference).
    flow_method: str = "farneback"
    flow_device: str = "cuda:0"
    min_texture_px: int = 20
    texture_grad_threshold: float = 8.0
    # ~3.3 s of consecutive texture loss (50 steps @ 15 Hz) → terminate.
    texture_lost_limit_steps: int = 50

    # Ocean current (water flow): v7 must hold GROUND position against the flow, not drift
    # with it. The flow observation (du/dv) still tells the agent which way the water is
    # moving; the reward below is reshaped so a still (ground-locked) ROV is rewarded and a
    # drifting one is not.
    # The field itself is the steady uniform current (strength 0.3 m/s along [1, 0.3, 0];
    # an explicit strength wins over level="medium"). Each episode multiplies it by a
    # linear ramp, so frame 1 is still water: the level pose and the pre-charged thrusters
    # are already holding, and the current then rises to full over current_ramp_s.
    # 0 disables the ramp (full current from the first step).
    current_ramp_s: float = 10.0
    ocean_current: OceanCurrentCfg = field(
        default_factory=lambda: OceanCurrentCfg(
            enabled=True,
            type="uniform",
            level="medium",  # unused while strength stays at its 0.3 default
            affect_target=False,
            clip_speed=5.0,
            params={
                "direction": [1.0, 0.3, 0.0],
                "time_modulation": False,
                # random field only. Each axis is max_velocity*sin*cos, so one horizontal
                # component peaks at 0.4 m/s and the vertical component at 0.12 m/s.
                "max_velocity": 0.8,
                "spatial_scale": 5.0,
                "temporal_scale": 5.0,
                "seed": 0.0,
            },
        )
    )

    # Textured plane in front of camera (4x4 by default): replaces the 4 dark spheres.
    # Pool: 4 m × 4 m cross-section, 4 m tall, centered at world origin (0,0,0).
    #   - plane_size=4.0, aspect=1.0 → plane is 4×4 (matches pool cross-section)
    #   - plane_distance=2.0 → plane at x=+2.0, one half-width ahead of ROV
    #   - starting_depth=0.0 → ROV at world origin; pool spans z∈[-2, 2]
    plane_size: float = 4.0
    plane_aspect: float = 1.0
    plane_distance: float = 2.0
    plane_texture_tile: float = 0.8
    plane_texture_path: str = ""
    plane_texture_paths: tuple = ()
    randomize_plane_texture: bool = False
    plane_occluders: bool = True

    # Four bright-green reference spheres forming a 0.5 m × 0.5 m square,
    # centered on the camera axis (y=0, z=starting_depth) at x=plane_distance - 0.5.
    # Visible in the camera viewport as visual landmarks.
    ref_sphere_radius: float = 0.1
    ref_sphere_color: tuple = (0.0, 1.0, 0.0)  # bright green
    ref_sphere_offset_x: float = 1.5  # m ahead of ROV on +X axis
    ref_sphere_half_side: float = 0.25  # half the 0.5 m square side → corners at ±0.25 m in y/z

    # GS-ROV physics parameters (from real-vehicle measurements).
    # thruster_layout "gsrov8" swaps in the real hull's 4 vertical + 4 horizontal thrusters
    # (T ~ u*|u| in Newtons) and the 8-channel PID allocation, instead of the WarpAUV proxy.
    thruster_layout: str = "gsrov8"
    # ROVMAKER 水下推进器 datasheet @ 6S/24 V: pulse 1000-2000 us, neutral 1500 us,
    # max thrust 2.0 kgf forward / 1.8 kgf reverse. Datasheet values, not measured.
    thruster_max_thrust_fwd_n: float = 19.613
    thruster_max_thrust_rev_n: float = 17.652
    thruster_neutral_us: float = 1500.0
    thruster_span_us: float = 500.0
    thruster_rated_voltage_v: float = 24.0
    # T_max ~ (V/24)^2. Bench pack measured 22.2-23.1 V (=87 % of rated); at 3S/12 V only
    # 25 % remains. Kept at rated here so the calibrated dynamics are unchanged -- set this
    # (or randomise it) when matching a specific pack.
    thruster_battery_voltage_v: float = 24.0
    thruster_deadband: float = 0.05
    # mass:        13.15 kg (weighed in air).
    # volume:      displaced volume ~12.95 L; with mass 13.15 kg the residual sink is
    #              only ~0.2 kgf (near-neutral buoyancy). Geometry/volume unchanged.
    # com_to_cob:  vector COM->COB in body frame. Only its *direction* sets the
    #              free-float attitude (buoyancy torque = offset x F_buoy); the body
    #              settles with COM->COB pointing up. Real hull floats ~20 deg nose-up
    #              with COM a touch aft of centre, so offset has a fore/aft part:
    #              atan(0.040 / 0.11) ≈ 20 deg.
    # inertia_diag: uniform-box estimate from CAD bbox 0.562 x 0.429 x 0.295 m, m=13.15.
    #              scaled from the 13.75 kg estimate by 13.15/13.75.
    mass: float = 13.15
    volume: float = 0.01295
    com_to_cob_offset: list = (0.040, 0.0, 0.11)
    inertia_diag: tuple = (0.297, 0.440, 0.545)
    # The real hull's measured free-floating trim is about 20 deg nose-up. With a +X (fore)
    # COM->COB offset, buoyancy lifts the nose, i.e. a *negative* euler-Y pitch:
    #   pitch_trim = -atan(com_x / com_z) = -atan(0.040 / 0.11) = -20.0 deg.
    trim_roll_rad: float = 0.0
    trim_pitch_rad: float = float(-np.arctan2(0.040, 0.11))

    # ---------------------------------------------------------------------------------
    # DOMAIN RANDOMIZATION (sim→real).
    #
    # Sampling horizon differs by knob because the physics differs:
    #   * water/pose terms are resampled EVERY EPISODE (a dive starts in a new current
    #     and from a new hand-flown hover pose);
    #   * hardware/image terms are sampled ONCE PER ENV and held for the whole run (one
    #     camera, one battery pack, one hull — a real unit does not re-draw its sensor
    #     noise level between dives).
    # Ranges are conservative on purpose: the real ROV is qualitatively like the sim, so
    # this should widen the basin around the current operating point, not invent a new
    # distribution. `dr_enable=False` reproduces the 2026-11-15 boxed configuration
    # exactly (the last three ablations were trained that way).
    dr_enable: bool = True
    # EVALUATION ONLY. Redraw the per-run ("hardware") knobs on every episode reset so each
    # episode is an independent sample of the unit distribution. Without it, N episodes on
    # 4 envs are 4 hardware draws x N/4 episodes: the effective sample size is 4 and every
    # confidence interval computed from them is optimistic. Must stay False for training —
    # a real unit keeps its camera gain and compass offset for the whole run.
    eval_redraw_per_run_dr: bool = False
    # Actuator-side latency, in control steps (1 step = 66.7 ms @ 15 Hz). Sampled per episode
    # as a uniform integer in [0, max]; 0 disables it. The real vehicle's sensor->compute->
    # link->ESC path measured 21.8 Hz command pacing plus a 50 Hz PWM frame and an
    # unmeasured thruster spin-up, so its dead time is 1-3 control periods, while the policy
    # below was trained with none. A zero-delay eval sweep on 2026-11-16 (model_960000) gave
    # 30.7 / 32.7 / 58.9 / 80.4 mm at 0 / 67 / 133 / 200 ms, i.e. the cliff starts between
    # one and two steps. Evaluation must leave this at 0 and use eval_v7_checkpoint.py's own
    # --action_delay, otherwise the two delays stack and the number is uninterpretable.
    train_action_delay_max_steps: int = 0
    # A single constant dead time for every episode; takes priority when >= 0. The [0, 3]
    # randomized arm was abandoned at 374k steps (2026-11-16): it plateaued at 130-165 mm
    # against 22-25 mm with no delay, then collapsed from 250k with critic_loss climbing
    # 0.48 -> 5.5. A hidden per-episode delay makes the MDP partially observable, so the
    # policy has to hedge between mutually incompatible compensations. The real vehicle's
    # dead time is a fixed property of the hardware rather than a per-dive draw, so a
    # constant is the faithful model, and training at the worst plausible value leaves any
    # shorter deployment delay inside the trained operating point.
    train_action_delay_fixed_steps: int = -1
    # ---- payload DR (tier 1: wrench emulation, no USD rework) --------------------------
    # A tool/sensor pod adds weight and moves the CG. Emulated as a world-frame weight
    # applied at a body-frame offset: F_body = R^T (-m_p g z), tau = r x F_body. Buoyancy
    # is deliberately left alone, so the vehicle gets heavier and the depth loop has to pay
    # for it — which is what actually happens when you bolt hardware onto a real ROV.
    # Draws are per episode (the payload changes between dives, not between control ticks),
    # matching the existing physical-DR convention (thrust gain, drag, ...).
    # Enable with e.g. (0.0, 0.30) / (±0.08) / (-0.15, 0.20). All-zero = disabled, bit-
    # identical to the current behaviour.
    dr_payload_mass_frac_range: tuple = (0.0, 0.0)   # fraction of cfg.mass
    dr_payload_offset_xy_range: tuple = (0.0, 0.0)   # m, body frame
    dr_payload_offset_z_range: tuple = (0.0, 0.0)    # m, body frame
    # ---- vision latency ---------------------------------------------------------------
    # Camera -> policy latency, FIXED for the run rather than drawn per episode. Per-episode
    # randomisation of a hidden sensor/actuator delay already failed here: the [0, 3] action
    # delay arm plateaued at 130-165 mm against 22-25 mm and then collapsed (see the note
    # above), because a hidden per-dive lag forces the policy to hedge between incompatible
    # compensations. Constants are also what the hardware gives: the camera pipeline does not
    # change between dives. Realistic budget: exposure/readout 10-30 ms + link 10-50 ms +
    # flow/odom 5-15 ms + inference 5-10 ms => 50-200 ms. 0.0 = disabled.
    vision_delay_ms: float = 0.0
    # Uniform current in the horizontal plane. The fixed 0.3 m/s along [1, 0.3, 0] field
    # was exactly one sample of this distribution.
    dr_current_speed_range: tuple = (0.20, 0.45)   # m/s, per episode
    # Ceiling kept at 0.45 on purpose. Measured 2026-11-20 with
    # debug_auv_v7_thrust_budget.py --sweep: the four horizontals reach only 38% of their
    # range at 0.45 m/s (right at the DR ceiling), 50% at 0.60, 84% at 1.00 — saturation is
    # around ~1.0 m/s, i.e. 2.4x authority in reserve at the ceiling. So the gate's
    # high-current failures are a control-precision problem, NOT an authority limit. Do not
    # narrow this range to make the gate pass: that lowers the task with no physical basis.
    dr_current_dir_full_circle: bool = True         # ±180° heading; False → fixed dir ± range
    dr_current_dir_range_deg: float = 20.0          # only when dr_current_dir_full_circle=False
    dr_current_vert_frac: float = 0.10              # |v_z| ≤ frac · speed (currents are ~horizontal)
    # Thrust gain: a pack at 87–100 % of rated voltage scales thrust by (V/24)² ≈ 0.76–1.0.
    # The gain is applied to the NORMALISED COMMAND, so thrust ∝ g² (u·|u| law): the range
    # below spans roughly ±10 % of delivered thrust.
    dr_thrust_gain_range: tuple = (0.90, 1.05)
    # Extra linear drag on the hull, N per (m/s), per env. The real added-mass/viscous
    # coefficients are estimates, so perturb the damping instead of pretending to know it.
    dr_drag_gain_range: tuple = (0.0, 3.0)
    # Camera: exposure/gain spread (multiplicative) and per-frame read noise, σ in [0,1]
    # units (σ·255 ≈ LSB). Kept small enough that the texture mask (gradient > 8) and the
    # template match stay meaningful — this is sensor realism, not a texture-loss test.
    dr_cam_brightness_range: tuple = (0.92, 1.08)
    dr_cam_noise_std_range: tuple = (0.004, 0.010)
    # Spawn pose: the ROV starts at a hand-flown hover, not teleported to the exact origin.
    # The anchor template is captured from wherever it starts, so an offset means the
    # policy must return to a first frame that is NOT the nominal one.
    dr_spawn_xy_range: float = 0.12      # m, ± on each horizontal axis
    dr_spawn_yaw_range_deg: float = 8.0  # ± heading at capture
    dr_spawn_depth_range: float = 0.04   # m, ± around starting_depth (the PI pulls it back)
    # Per-env texture drawn from plane_texture_paths (needs that list non-empty). One
    # seabed per env for the whole run, seeded from cfg.seed. On by default since
    # 2026-11-18: with a single asset the template matcher never sees an unfamiliar wall,
    # and the measured jump rate (0–2 %) was measured on exactly one texture.
    dr_texture_choice: bool = True

    # ---- Perception-calibration DR: the scale visodom turns pixels into metres with ----
    # visodom converts pixels to metres through three numbers that on a real hull get
    # measured once, by hand: the pixel focal length f_px, the initial standoff Z(0), and
    # the camera mount angles. All three are wrong by a few percent on a real build, and
    # the error lands DIRECTLY on the reported displacement — Δ = Δpx·Z/f laterally,
    # Δ_axial = k·Z along the axis. Until now they were exact in sim, so visodom_err
    # measured the integrator while pretending the ruler under it was perfect.
    #
    # These perturb the CALIBRATION THE ESTIMATOR BELIEVES IN, not the hardware: the true
    # focal stays at cam_hfov_deg and the true wall stays at plane_distance. That is what
    # an uncalibrated intrinsics / unknown standoff actually looks like — the world is
    # fine, your numbers about it are not. 1.0 = exact no-op, so the 2026-11-15
    # configuration is reproduced bit-for-bit and no existing result moves.
    # 2026-11-17 REVERTED to no-op after one run showed it makes the position reward
    # WORSE. With these at (0.90,1.10) / (0.90,1.10) / 0.05 / 0.020 (train_visodom_dr.log,
    # E 200k→286k, 47 points) disp_spawn_m went 48.8 → 61.2 mm while ep_rew_mean sat frozen
    # at 548-550 — precisely the symptom the DR was supposed to remove. Mechanism: the
    # injected error reached 15-58 mm, the same order as the 50-88 mm displacement being
    # optimised, so the reward turned noise-dominated. A policy cannot chase a 30 mm
    # improvement through a 50 mm error.
    #
    # If this is ever turned back on: perturbing the ruler only helps when the ruler's
    # error is SMALLER than the target precision. It is not, here — the real estimator's
    # own error (24 mm at hover, 586 mm at a 0.87 m excursion) is already larger than the
    # 3-30 mm accuracy we want, which is why this whole lever is closed.
    #
    # The modelling is still right in principle (one hand calibration per hull, and a real
    # integrator's error does move with standoff/lighting/flow-fit), so the code paths and
    # the per-episode draws stay — only the magnitudes are back to no-op.
    # 1.0 / 0.0 = exact no-op: the 2026-11-15 configuration is reproduced bit-for-bit.
    dr_visodom_focal_rel_range: tuple = (1.0, 1.0)   # × the estimator's f_px,  per env
    dr_visodom_range_rel_range: tuple = (1.0, 1.0)   # × the estimator's Z(0), per env
    dr_visodom_ep_scale_rel: float = 0.0    # ±, per episode, on the lateral scale
    dr_visodom_bias_m: float = 0.0          # ±, per episode, per axis, additive (m)
    # Camera mount, realised in the WALL's frame: on a planar target only the relative
    # camera-plane transform is observable, so tilting the textured plane is the same
    # image as mis-aligning the camera — and it cannot corrupt the camera prim. Degrees,
    # ± about the wall centre, col 0 = pitch (wall leans top toward/away), col 1 = roll
    # (image rotates). Baked into the mesh at setup, so per-ENV and fixed for the run,
    # like a real mount. The estimator is never told: this is what actually breaks the
    # fronto-parallel assumption _fit_translational_flow leans on, because a tilted wall
    # makes du vary with x and the (a, k) fit reads part of that gradient as surge.
    # 0.0 = exact no-op.
    dr_cam_mount_deg_range: float = 0.6

    # ---- Sensor-side DR: the two numbers the estimator borrows from the IMU ----
    # De-rotation and anchor-yaw used the ground-truth quaternion, i.e. a perfect compass
    # that no real hull has. A biased/flaky heading lands directly on the visodom integral
    # (a wrong dyaw rotates the whole accumulated displacement) and on the template window.
    # Bias is per env and held for the run (a real compass has one heading offset after
    # calibration); noise is per frame. Degrees. 0.0 = exact no-op.
    dr_imu_yaw_bias_deg: float = 1.0
    dr_imu_yaw_noise_deg: float = 0.15
    # ---- Hull drag anisotropy: the single scalar drag_gain above assumes the added mass
    # is the same along all three axes, which is false for a boxy hull. Per-axis
    # multipliers on the drag gain, ± fraction, per env, held for the run. 0.0 = no-op.
    dr_drag_aniso_range: float = 0.15
    # ---- Water visibility: turbidity attenuates contrast and softens edges, which is how
    # real flow RMS drops without the vehicle moving. Left/right bounds are the fraction
    # of contrast REMOVED (0 = crystal clear, 0.6 = 40 % of the original contrast); sampled
    # per episode because a dive drifts through different water. 0.0 = exact no-op.
    dr_turbidity_range: tuple = (0.0, 0.25)

    # Reward weights.
    rew_flow: float = 1.0
    rew_alive: float = 0.3
    rew_depth: float = 0.1
    rew_att: float = 0.1
    rew_action: float = 0.02
    # Bounded action-rate penalty, borrowed from MarineGym (IROS 2025). Their effort term is
    # ``0.1 * exp(-|a|)``, i.e. range [0, 0.1]: a penalty term that cannot exceed its weight.
    # The previous unbounded hinge ``3.0 * sum(relu(|da|-0.08)^2)`` reached ~33/step during
    # SAC's random exploration against a ~2.45/step reward and killed arm J at 26.9 k steps
    # (critic_loss 2.47). Saturating forms make that class of failure impossible.
    # With scale=0.15 the term goes 0.10 at |da|=0.05, 0.13 at 0.15, and saturates at 0.2.
    rew_action_rate: float = 0.2
    action_rate_scale: float = 0.15
    # Effort (not rate): rewards low continuous thrust, same form as MarineGym's
    # ``0.1 * exp(-effort)``. Rate alone cannot separate "steady push against the current"
    # from "buzzing": both move the stick every step. Missing half of the signal.
    #
    # ARM L RESULT (2026-11-20): 0.1 -> 0.0 was tested as a single-variable change and it
    # COLLAPSED the policy -- path_m 2627 mm (K: 285-363), late_drift 172 mm/s (K: 12.4),
    # est_err p90 192 mm, gate 3-11% on every bar. Signature identical to arm G (2475 mm).
    # So the effort term is NOT making the policy lazy: it is suppressing a real limit cycle.
    # Removing it buys nothing at high current and loses the hover outright. Keep 0.1.
    # (The high-current drift gap is therefore not an under-command problem -- 2.4x thrust
    # reserve was measured but the issue is control/perception, not reward weight.)
    rew_effort: float = 0.1
    sigma_flow: float = 0.05  # calibrated: 1x std of measured flow_rms_n (≈0.047, clamped [0.05,0.5])
    sigma_depth: float = 1.5
    sigma_att: float = 0.3
    # REMOVED 2026-11-15: the slow-drift term (rew_drift / sigma_drift / drift_tau_s) and
    # its leaky integral of the signed mean flow. Leave-one-out (ablate_drift, 461k steps,
    # best 25-window disp_end_m 0.0107 m — the best of all four ablations) showed the term
    # is redundant: the first-frame anchor offset measures the same drift ABSOLUTELY, and
    # the visodom position term already supplies the velocity-scale gradient that this
    # term was introduced to provide. Dropping it also removes a per-step de-rotation +
    # integration from the hot path. Do not reintroduce it without an ablation that shows
    # a gap against this configuration.

    # First-frame visual anchor: hover = "make the camera see the episode's first frame
    # again". At reset the first textured frame is captured as a template (center crop at
    # flow-res); every step the current frame is template-matched against it
    # (TM_CCOEFF_NORMED) producing (a) the pixel offset of the scene vs frame 1 — the
    # goal-directed error signal the leaky integrator never provided (it forgets with
    # τ=3 s, and the 287k model yaw-gamed it to 0) — and (b) the match score ncc ∈
    # [-1, 1], the direct "current frame looks like the first frame" measure. Both are
    # pure vision → deployable on the real ROV. WORKING RANGE: the match window spans
    # ±(fw-cw)/2 px ≈ ±0.9 m lateral @ Z=2 m (f_px≈185 @ 320-wide, HFOV 81.65°);
    # outside it the offset saturates at the window edge with its SIGN still pointing
    # home, and sim collapses — the GT anchor below carries the global pull there.
    anchor_crop_height: int = 96   # template crop @ flow-res 176×320; 96×160 → window ±40/±80 px
    anchor_crop_width: int = 160
    rew_anchor_vis: float = 0.3      # exp(-d_vis²/σ²) on the anchor offset, in metres
    # --- two-scale twins (2026-11-16) -----------------------------------------------
    # A single Gaussian stops discriminating once the error is a few sigma out: at 30 mm
    # it reads 0.96-0.99, i.e. "5 mm hover" and "38 mm hover" differ by ~0.5% of the
    # episode return, below the DR noise. These finer twins are added ON TOP of the broad
    # terms rather than replacing them — arm A (broad sigma replaced by sharp sigma)
    # collapsed at 105k steps from ~25 mm to 872 mm, because sharpening had also removed
    # the only global restoring force while the flow term stays aliased at range
    # (receding from the plane reduces image motion, so flow_term held 0.966 at 0.87 m).
    # 2026-11-17 REVERTED after it diverged. Trial: move the fine weight off the estimator
    # and onto the template (rew_position_fine 0.15→0.0, rew_anchor_vis_fine 0.15→0.30,
    # total still 2.55). Over 61 points (200k→309k) disp_spawn_m held 20-50 mm to ~260k,
    # then left the basin: 1.1 m at 268k, 2.9 m at 284k, 5.1 m at 300k. ep_rew_mean peaked
    # 541 at ~250k, then fell to 421.
    #
    # What that proves: the sigma=0.1 m visodom term was load-bearing for STABILITY, not
    # only accuracy. It is the single cue with a usable gradient across 50 mm → the
    # template's ±0.9 m window edge; the sigma=0.05 template twin is near step-like
    # (≈0.30 inside ~10 mm, ≈0 past 70 mm), and position(sigma=1.0) / anchor_vis(sigma=0.2)
    # are both nearly flat in that band. Remove the 0.1 term and there is a blind gap the
    # policy drifts through; once it is past the window edge every position cue reads zero
    # and nothing pulls it home. Being "the noisy cue billed twice" is not the same as
    # being redundant — it was the only medium-grain term.
    #
    # Two reward-level attempts are now on record as failed (this one, and the visodom
    # error-DR run before it). The reward is not the lever. Reverted to the E-arm values.
    rew_anchor_vis_fine: float = 0.15
    sigma_anchor_vis_fine_m: float = 0.05
    rew_position_fine: float = 0.15
    sigma_visodom_fine_m: float = 0.1
    sigma_anchor_vis_m: float = 0.2  # ≈ half the working basin (d=0.4 ⇒ exp(-4)≈0.02)
    rew_sim: float = 0.2             # (ncc+1)/2, present-gated: "how much the image matches frame 1"
    rew_yaw_anchor: float = 0.1       # yaw error to the capture heading — image matching is only
    sigma_yaw_anchor: float = 0.3     # well-defined at the same heading (yaw ≠ 0 rotates the template)

    # Visual odometry (visodom): the POSITION cue that replaces the privileged GT
    # anchor in the reward. Frame-1 matching (dense Farneback or template) dies
    # once the ROV has moved — the images diverge (probe 2026-11-13: dense
    # frame-1 flow read a ≈ 0 px at 0.4 m true sway). The consecutive-frame flow
    # keeps working at ANY distance because it is a ground-relative velocity.
    # Integrating its uniform component (a_px, b_px) without leak, de-rotated by
    # yaw so heading changes cannot cancel it, plus the expansion coefficient k
    # with a tracked Z for the surge leg, gives a body-frame displacement
    # estimate. Probe (v7/scripts/debug/debug_auv_v7_visodom.py): corr vs GT = 1.000 on
    # both axes; a plain P servo on it returns from 0.57 m to 0.044 m under the
    # full 0.3 m/s current ramp while zero-action drifts to 1.02 m. It
    # overestimates ~15–20 % (113.5 px/m sway vs 92.6 geometric; surge 1.14x) —
    # the reward does not care about the scale, it learns against it.
    # When True, rew_anchor pays the visodom term; the GT displacement stays in
    # the logs (disp_end_m / visodom_err_m) and nowhere else.
    vis_odometry: bool = True
    sigma_visodom_m: float = 1.0  # basin of the visodom reward term (≈ sigma_anchor_m)
    # Observation normalization for the visodom channels, kept SEPARATE from the reward
    # basin above. If they shared one field, sweeping the reward sharpness would also
    # change what the actor sees (±1 saturates at sigma_visodom_m metres), and a win or
    # loss could no longer be attributed to reward sharpness alone. Default 1.0 = the
    # observation semantics every run before 2026-11-16 was trained with.
    sigma_visodom_obs_m: float = 1.0
    # Hard physical bound on the visodom integrator, expressed as a speed (m/s). Between
    # two control steps the hull cannot travel farther than v_max·dt, so any single flow
    # fit that claims more is corrupt by definition. Arm F (2026-11-18) showed the cost of
    # leaving it unbounded: with heading bias + wall tilt + turbidity the estimator went
    # from 0.03 m to 1.8 m of error, and because the reward pays that estimate, the policy
    # degenerated into a bimodal "either 46 mm or 2.6 m" strategy. 1.0 m/s is far above the
    # vehicle's real speed while still bounding the damage of one bad frame.
    visodom_max_speed_mps: float = 1.0
    # Which displacement the POSITION REWARD pays: "gt" (true horizontal offset from this
    # episode's spawn point) or "visodom" (the estimator's own integral).
    #
    # "gt" is the default since 2026-11-18, and it is not a relaxation of the task: the
    # actor still sees ONLY the image, the estimate, the anchor channels — never the true
    # pose. Difficulty is unchanged. What changes is the learning signal: paying the
    # estimate made the reward noise-dominated exactly when the DR got harder, because the
    # injected error (0.3-1.8 m) reached the same order as the 30-50 mm being optimised.
    # That is what killed arm F (~70 k) and arm G (~190 k): critic_loss 0.06 -> 0.6 in one
    # block, then the policy split into "46 mm or 2.6 m" episodes. Rewarding truth keeps
    # the harder distribution while giving the critic a signal it can actually fit.
    # "visodom" reproduces the old behaviour bit-for-bit (use it for the ablation that
    # answers "how much does an uncalibrated ruler hurt?").
    #
    # 2026-11-19 REVERTED to "visodom" after arm H measured the opposite of what I expected.
    # Paying truth made the TRAINING LOG look great (disp_spawn 21-28 mm at 100 k vs arm G's
    # 33-37 mm) and made the 40-episode eval WORSE (mean 63.6 mm vs 51.7 mm). Mechanism: the
    # actor only perceives the estimate, so rewarding truth asks it to optimise a quantity it
    # cannot observe; the proxy mapping it learns does not survive a fresh set of DR draws,
    # whereas "drive your own estimate to zero" is a target it can actually see and control at
    # deployment. The pretty log number was self-deception. The collapse arms F/G/H suffered
    # is also not caused by this — F, G and H collapsed at 130-200 k with three different
    # reward sources, so that failure lives in late-training stability, not in the reward.
    rew_position_source: str = "visodom"
    # Leave-one-out. All default False, which is the system trained on 2026-11-13.
    # A True flag zeros that cue's reward and the observation channels the policy
    # reads. Image stays (11, 64, 64) and state stays 17-D, so the network shape
    # does not change. Internal estimates keep running and the logs stay filled.
    # ablate_visodom pays nothing for position: it does not fall back to GT.
    # ablate_flow hides the flow image and the flow reward; the integrator still
    # reads the internal flow. ablate_template drops the match reward, the
    # similarity reward, and the capture-heading reward together.
    ablate_flow: bool = False
    ablate_template: bool = False
    ablate_visodom: bool = False

    # Privileged anchor (reward-only, ground truth from sim): penalize the horizontal
    # displacement from the episode spawn origin, in metres. The 287k-step model gamed
    # the vision-only integrator by yawing steadily while drifting: the body-frame flow
    # signature of a world-fixed drift rotates with the head, so the leaky integral
    # cancelled to ~0 (drift_mag≈0.018) while the ROV travelled 2.8 m per 15 s episode.
    # The anchor cannot be gamed — it measures the metric we actually care about.
    # The actor input stays vision-only (asymmetric training; the critic already sees
    # privileged state). sigma_anchor_m=1.0: exploit d≈2.8 m ⇒ exp(-7.8)≈0 per step.
    # Weight REDUCED 0.3 → 0.15 now that the visual anchor measures the same metric
    # deployably; with vis_odometry=True the visodom term is paid INSTEAD of this
    # one; this block stays as the fallback (vis_odometry=False) backstop.
    rew_anchor: float = 0.15
    sigma_anchor_m: float = 1.0

    # Depth / attitude targets (wider tolerance).
    depth_target: float = 0.0  # set in reset to starting_depth (AUV at z=0)
    depth_limit: float = 1.5  # meters from target -> terminate
    att_limit: float = 0.3  # radians

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/ForwardCamera0",
        data_types=["rgb", "distance_to_image_plane"],
        width=1280,
        height=720,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        spawn=None,
    )

    # Observation space (v3.1 spec):
    #   image: RGB (3) + du_norm (1) + dv_norm (1) + |u|_norm (1) + mask (1) = 7 channels @ 64×64
    #   state: depth, roll, pitch, yaw, v_surge, v_sway, v_heave, yaw_rate, flow_rms,
    #          present, tex_frac = 11 dims
    # `present` (texture gate) is REQUIRED in the critic state: flow_term and alive are
    # both multiplied by present, and a blind drifting ROV (present=0, flow_rms≈0) is
    # otherwise indistinguishable from a perfect hover (present=1, flow_rms=0) — that
    # aliasing injected unresolvable noise into Q (critic_loss 20→85 collapse at ~1M steps).
    # Divergence convention: du/dv normalized by flow_render_width/height (px @ flow res).
    _flow_img_channels: int = 3  # du_n, dv_n, mag_n
    # RGB + flow_image(3) + mask(1) + anchor offset(2, broadcast constants: first-frame
    # template-match offset du/dv, ±1 at the match-window edge). Replaces the leaky drift
    # integral broadcast: the actor is image-only (CustomSACPolicy), and the anchor offset
    # is its goal-directed "which way is frame 1 / how far am I" cue — the integral forgot
    # (τ=3 s) and was yaw-gamed by the 287k model; the offset is absolute and ungameable.
    # + visodom(2, broadcast): integrated consecutive-flow displacement (m / sigma,
    # ±1 clamped) — the position cue that keeps counting OUTSIDE the template-match
    # window (the anchor saturates at ~±0.9 m; visodom does not).
    _image_channels: int = 3 + _flow_img_channels + 1 + 2 + 2
    # state: depth, roll, pitch, yaw, v_surge, v_sway, v_heave, yaw_rate, flow_rms,
    #        present, tex_frac, anchor du_n, anchor dv_n, anchor_sim(ncc), anchor yaw_err,
    #        visodom_x (m), visodom_y (m) = 17 dims
    _state_dims: int = 17
    observation_space = {
        "image": gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(_image_channels, int(cam_height), int(cam_width)),
            dtype=np.float32,
        ),
        "state": gym.spaces.Box(
            low=np.array(
                [-10.0, -np.pi, -np.pi, -np.pi, -3.0, -3.0, -3.0, -3.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, -np.pi, -10.0, -10.0],
                dtype=np.float32,
            ),
            high=np.array(
                [10.0, np.pi, np.pi, np.pi, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.pi, 10.0, 10.0], dtype=np.float32
            ),
            dtype=np.float32,
        ),
    }


class AUVTrackingV7FlowHoverGsrovEnv(AUVTrackingEnv):
    """Station keeping via global optical-flow nulling."""

    cfg: AUVTrackingV7FlowHoverGsrovEnvCfg

    def __init__(self, cfg: AUVTrackingV7FlowHoverGsrovEnvCfg, render_mode: str | None = None, **kwargs):
        # Keep the TiledCamera resolution in sync with the (overridable) render_width/height.
        # tiled_camera.width/height are otherwise hard-coded at class-definition time, so
        # command-line / benchmark overrides of render resolution would be silently ignored.
        cfg.tiled_camera.width = int(cfg.render_width)
        cfg.tiled_camera.height = int(cfg.render_height)
        # Attitude hold target, resolved BEFORE super().__init__: the base __init__ calls
        # _reset_idx at its end, and the spawn quaternion there must already use the work
        # attitude (level) instead of the free-floating trim.
        if bool(getattr(cfg, "level_attitude", False)):
            self._att_roll_target = 0.0
            self._att_pitch_target = 0.0
        else:
            self._att_roll_target = float(cfg.trim_roll_rad)
            self._att_pitch_target = float(cfg.trim_pitch_rad)
        # Override action buffers to 3-D: [surge, sway, yaw]. The parent class
        # initializes _actions / _prev_actions as (N, 2) — we re-create after
        # super().__init__ so the parent's own buffers are already allocated.
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        self._actions = torch.zeros(self.num_envs, 3, device=dev)
        self._prev_actions = torch.zeros(self.num_envs, 3, device=dev)
        # Actuator delay: one slot per possible delay, newest at index -1. With max=0 the
        # buffer is just the pass-through and costs one clone per step.
        _delay_max = int(getattr(self.cfg, "train_action_delay_max_steps", 0))
        self._act_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._action_hist = torch.zeros((self.num_envs, _delay_max + 1, 3), device=dev)
        self._cmd_vel = torch.zeros(self.num_envs, 3, device=dev)
        # Control period, so latencies can be expressed in control steps.
        self._ctrl_dt = float(self.cfg.sim.dt) * float(self.cfg.decimation)
        # Vision latency (fixed, see cfg): the whole visual observation is delayed, because
        # image, flow, anchor and visodom all come out of the same frame. The ego-velocity
        # state is NOT delayed — IMU/DVL latency is a different, much smaller number, and
        # delaying it would silently double-count the lag.
        self._vis_delay_steps = max(
            0,
            int(round(float(getattr(self.cfg, "vision_delay_ms", 0.0)) / (self._ctrl_dt * 1e3))),
        )
        self._vis_hist = None
        self._vis_push_needed = False
        self._vis_fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        # Payload wrench (tier 1). Zero until _sample_domain_randomization draws it.
        self._payload_mass = torch.zeros(self.num_envs, device=dev)
        self._payload_off = torch.zeros((self.num_envs, 3), device=dev)
        if self._vis_delay_steps > 0 or float(self.cfg.dr_payload_mass_frac_range[1]) > 0.0:
            # Banner so every run log states the latency/payload regime it was trained under.
            print(
                "[v7][LAT] vision_delay=%.0f ms (%d step(s)) | payload up to %.2f kg, "
                "xy +-%.3f m, z [%.3f, %.3f] m"
                % (
                    self._vis_delay_steps * self._ctrl_dt * 1e3,
                    self._vis_delay_steps,
                    float(self.cfg.dr_payload_mass_frac_range[1]) * float(self.cfg.mass),
                    float(self.cfg.dr_payload_offset_xy_range[1]),
                    float(self.cfg.dr_payload_offset_z_range[0]),
                    float(self.cfg.dr_payload_offset_z_range[1]),
                ),
                flush=True,
            )
        self._prev_cmd_vel = torch.zeros(self.num_envs, 3, device=dev)
        # v7-local depth-loop retune (see cfg comment for the Routh analysis).
        # Class defaults in batch_pid.py are untouched — other tasks keep the old law.
        self.pid.kp_depth = float(self.cfg.depth_kp)
        self.pid.ki_depth = float(self.cfg.depth_ki)
        self.pid.kd_depth = float(self.cfg.depth_kd)
        self.pid.max_heave_wrench = float(self.cfg.heave_wrench_max)
        # Level-attitude hold: aim the attitude loop at the work attitude and give it
        # integral action so the verticals can carry the constant buoyancy couple.
        if bool(getattr(self.cfg, "level_attitude", False)):
            self.pid.trim_roll_rad = self._att_roll_target
            self.pid.trim_pitch_rad = self._att_pitch_target
            self.pid.att_ki = float(self.cfg.att_ki)
            self.pid.att_int_limit = float(self.cfg.att_int_limit)
            # Base __init__ already reset every env with att_ki still 0, which zeroed
            # the integrators. Solve the quadratic thrust map and pre-charge again.
            fz_cmd, my_cmd, got_fz, got_my, want_fz, want_my = self._solve_level_hold_wrench()
            self._level_heave_wrench = fz_cmd
            self._level_pitch_moment = my_cmd
            self._apply_level_hold_precharge(torch.arange(self.num_envs, device=self.device))
            print(
                f"[v7] level-hold pre-charge: heave cmd {fz_cmd:.3f} -> {got_fz:.2f}/{want_fz:.2f} N, "
                f"pitch cmd {my_cmd:.3f} -> {got_my:.2f}/{want_my:.2f} N·m",
                flush=True,
            )
        else:
            self._level_heave_wrench = 0.0
            self._level_pitch_moment = 0.0
        print(
            "[v7-ablate] "
            f"flow={int(bool(self.cfg.ablate_flow))} "          f"template={int(bool(self.cfg.ablate_template))} "
            f"visodom={int(bool(self.cfg.ablate_visodom))}",
            flush=True,
        )

    def _thruster_body_fz_my(self, fz_cmd: float, my_cmd: float) -> tuple[float, float]:
        """Body heave (N) and pitch moment (N·m) produced by one allocation command.

        The allocator treats the command as linear in ``u``. Propeller thrust is
        ``T_max·u·|u|`` with separate forward and reverse ratings, so the realized
        wrench is not the command times ``T_max``.
        """
        cap = float(self.pid.max_heave_wrench)
        wrench = torch.zeros(1, 6, device=self.device)
        wrench[0, 2] = max(-cap, min(cap, fz_cmd))
        wrench[0, 4] = max(-cap, min(cap, my_cmd))
        motor = allocate_thrusters(wrench, self.pid._b_pinv, clamp=1.0, priority=False)[0]
        deadband = float(getattr(self.cfg, "thruster_deadband", 0.05))
        motor = torch.where(motor.abs() < deadband, torch.zeros_like(motor), motor)
        t_max = torch.where(
            motor >= 0.0,
            torch.full_like(motor, self._thrust_fwd_n),
            torch.full_like(motor, self._thrust_rev_n),
        )
        thrust = t_max * motor * motor.abs()
        n = self.num_thrusters
        axes = torch.zeros(n, 3, device=self.device)
        axes[:, 0] = 1.0
        forces = quat_apply(self.thruster_quats[:n], axes) * thrust.unsqueeze(-1)
        torques = torch.cross(self.thruster_com_offsets[0], forces, dim=-1)
        return float(forces.sum(0)[2]), float(torques.sum(0)[1])

    def _solve_level_hold_wrench(self) -> tuple[float, float, float, float, float, float]:
        """Allocation commands whose quadratic thrust cancels weight and the pitch couple.

        Returns ``(fz_cmd, my_cmd, realized_fz, realized_my, target_fz, target_my)``.
        At level, buoyancy torque about +Y is ``-dx·ρ·g·V`` (COB is forward of the
        COM), so the thrusters must supply the opposite moment. Net heave is the
        residual weight ``m·g - ρ·g·V``.
        """
        g = float(self._gravity_mag)
        rho = float(self.cfg.water_rho)
        vol = float(self.cfg.volume)
        dx = float(self.cfg.com_to_cob_offset[0])
        target_fz = float(self.cfg.mass) * g - rho * g * vol
        target_my = dx * rho * g * vol
        # Linear guess (command = newtons / T_max). Newton then accounts for u·|u|.
        x = target_fz / max(self._thrust_fwd_n, 1e-6)
        y = target_my / max(self._thrust_fwd_n, 1e-6)
        cap = float(self.pid.max_heave_wrench)
        for _ in range(12):
            f0, m0 = self._thruster_body_fz_my(x, y)
            rf, rm = f0 - target_fz, m0 - target_my
            if abs(rf) < 0.05 and abs(rm) < 0.05:
                break
            eps = 1.0e-3
            fx, mx = self._thruster_body_fz_my(x + eps, y)
            fy, my = self._thruster_body_fz_my(x, y + eps)
            j00, j10 = (fx - f0) / eps, (mx - m0) / eps
            j01, j11 = (fy - f0) / eps, (my - m0) / eps
            det = j00 * j11 - j01 * j10
            if abs(det) < 1.0e-8:
                break
            x -= (j11 * rf - j01 * rm) / det
            y -= (-j10 * rf + j00 * rm) / det
            x = max(-cap, min(cap, x))
            y = max(-cap, min(cap, y))
        got_fz, got_my = self._thruster_body_fz_my(x, y)
        if abs(got_fz - target_fz) > 0.2 or abs(got_my - target_my) > 0.2:
            raise RuntimeError(
                f"level-hold pre-charge did not converge: "
                f"Fz {got_fz:.2f}/{target_fz:.2f} N, My {got_my:.2f}/{target_my:.2f} N·m"
            )
        return x, y, got_fz, got_my, target_fz, target_my

    def _apply_level_hold_precharge(self, env_ids: torch.Tensor) -> None:
        """Write the solved hold into the depth and pitch integrators.

        ``fz = ki·sum`` and ``my = -att_ki·sum_y`` when the proportional and
        derivative errors are zero, which is the spawn state.
        """
        if not bool(getattr(self.cfg, "level_attitude", False)):
            return
        if float(self.pid.att_ki) > 0.0:
            self.pid.att_error_sum[env_ids, 1] = -float(self._level_pitch_moment) / float(self.pid.att_ki)
        if float(self.pid.ki_depth) > 0.0:
            self.pid.depth_error_sum[env_ids] = float(self._level_heave_wrench) / float(self.pid.ki_depth)

    def _setup_scene(self):
        # Base builds robot + target + ground + light + sonar. We override to drop target/sonar.
        if bool(getattr(self.cfg, "use_real_usd", False)):
            self.cfg.robot_cfg = gsrov_robot_cfg(mass=float(getattr(self.cfg, "mass", DEFAULT_MASS)))
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self._robot = RigidObject(self.cfg.robot_cfg)
        # No textured seabed ground — pool is self-contained with solid-color walls and floor.
        # self._build_textured_ground()  # removed
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["robot"] = self._robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Camera (adopted prim inside asset).
        self._spawn_robot_camera()
        self._apply_camera_intrinsics()
        self._spawn_cam_mount_dr()
        self._spawn_view_camera()
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera

        # 4x4 textured plane ahead of AUV camera (replaces 4 dark spheres).
        self._spawn_textured_plane()

        # 4 bright reference spheres in a 1 m square ahead of the camera.
        self._spawn_reference_spheres()

        # Flow / texture buffers.
        # frame_stack=1, policy resolution 64×64.  Farneback outputs (N,2,H,W) in pixels;
        # stored as flow_image[N, 2, 64, 64] with du/1280, dv/720 normalization.
        rh, rw = int(self.cfg.render_height), int(self.cfg.render_width)
        h, w = int(self.cfg.cam_height), int(self.cfg.cam_width)
        self._flow_image = torch.zeros((self.num_envs, 2, h, w), dtype=torch.float32, device=self.device)
        self._texture_mask = torch.zeros((self.num_envs, 1, h, w), dtype=torch.float32, device=self.device)
        self._flow_rms = torch.zeros(self.num_envs, device=self.device)
        # Visual odometry: non-leaky de-rotated integral of the consecutive-flow
        # uniform component (u_px, v_px @ flow-res), the tracked plane distance
        # _visodom_z (m) and the surge displacement _visodom_x (m, from the
        # expansion coefficient k). Updated once per policy step in
        # _update_flow_and_texture; reset in _reset_idx. All vision — deployable.
        self._visodom = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._visodom_x = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._visodom_z = torch.full(
            (self.num_envs,), float(self.cfg.plane_distance), dtype=torch.float32, device=self.device
        )
        self._visodom_yaw_prev = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Diagnostic only (not in the observation or the reward): translational
        # flow split into a uniform field (lateral) and a radial field (along the
        # optical axis). Current-aligned creep is mostly axial and cancels in the
        # signed mean; the radial coefficient keeps it.
        self._flow_uniform = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._flow_expand = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # First-frame visual anchor. Template = center crop of the episode's first
        # textured frame at flow-res (uint8 gray). Offsets are the scene's pixel
        # displacement vs that frame, in the CURRENT body frame (+u = scene moved
        # right in the image = ROV moved left). Captured/updated once per policy
        # step in _update_anchor_state.
        ach, acw = int(self.cfg.anchor_crop_height), int(self.cfg.anchor_crop_width)
        self._anchor_tmpl = torch.zeros((self.num_envs, ach, acw), dtype=torch.uint8, device=self.device)
        self._anchor_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._anchor_off_px = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        # Match score TM_CCOEFF_NORMED ∈ [-1, 1] — 1.0 = identical to frame 1.
        self._anchor_ncc = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # |offset| converted to metres via plane geometry (reward/logging quantity).
        self._anchor_disp_m = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Heading at anchor capture — reference for the yaw-to-anchor state/reward.
        self._anchor_yaw = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Half-window of the template search (px @ flow-res): offset normalization, ±1 at edge.
        self._anchor_margin = (
            float((int(self.cfg.flow_render_width) - acw) // 2),
            float((int(self.cfg.flow_render_height) - ach) // 2),
        )
        # Pixel focal length @ flow-res for px→m conversion (fronto-parallel plane).
        self._anchor_f_px = float(self.cfg.flow_render_width) / (
            2.0 * float(np.tan(np.radians(float(self.cfg.cam_hfov_deg)) * 0.5))
        )
        self._texture_count = torch.zeros(self.num_envs, device=self.device)
        self._prev_rgb = torch.zeros((self.num_envs, rh, rw, 3), dtype=torch.uint8, device=self.device)
        self._prev_rgb_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._texture_lost_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Per-episode diagnostics: reward-term sums and termination-cause flags. Flushed
        # into extras["log"] on reset, where Sb3VecEnvWrapper + EpisodeRewardLogCallback
        # surface them in TensorBoard as Episode_Reward/<key> — so the next collapse
        # shows its cause (texture-lost vs depth vs attitude vs timeout) directly.
        self._ep_flow_rms_sum = torch.zeros(self.num_envs, device=self.device)
        self._ep_present_sum = torch.zeros(self.num_envs, device=self.device)
        self._ep_flow_term_sum = torch.zeros(self.num_envs, device=self.device)
        # Action-rate / effort diagnostics. Rate is what the penalty bounds; effort is what
        # separates "holding against the current" from "buzzing", so both are logged.
        self._ep_action_delta_sum = torch.zeros(self.num_envs, device=self.device)
        self._ep_rate_pen_sum = torch.zeros(self.num_envs, device=self.device)
        self._ep_effort_sum = torch.zeros(self.num_envs, device=self.device)
        self._ep_anchor_sum = torch.zeros(self.num_envs, device=self.device)
        # First-frame-anchor episode sums: mean visual displacement (m) and sim score.
        self._ep_anchor_vis_sum = torch.zeros(self.num_envs, device=self.device)
        self._ep_anchor_sim_sum = torch.zeros(self.num_envs, device=self.device)
        # Fraction of steps where the first-frame template was actually captured. Capture
        # is refused while the centre crop is nearly textureless, and a refused episode has
        # anchor_ncc == 0 the whole way, which reads as "bad match" when it is really "no
        # template at all" — the two need to be distinguishable before any policy eval.
        self._ep_anchor_valid_sum = torch.zeros(self.num_envs, device=self.device)
        # cause order: [texture_lost, depth, attitude, timeout]
        self._ep_cause = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)

        # ---- domain-randomization state (ranges live in the cfg block) ----
        # Water/pose draws are refreshed every episode in _reset_idx.
        self._dr_cur_speed = torch.zeros(self.num_envs, device=self.device)
        self._dr_cur_dir = torch.zeros((self.num_envs, 2), device=self.device)
        self._dr_cur_dir[:, 0] = 1.0
        self._dr_cur_vz = torch.zeros(self.num_envs, device=self.device)
        # Spawn point per episode (x, y): the "home" the anchor defines and the origin
        # that disp_end_m must be measured against under spawn randomization.
        self._spawn_xy = torch.zeros((self.num_envs, 2), device=self.device)
        # Spawn depth offset, kept as state (not just applied to the root pose) so the
        # coverage audit can verify it is redrawn every episode like the other pose draws.
        self._dr_spawn_depth = torch.zeros(self.num_envs, device=self.device)
        # Hardware/image draws are held for the whole run (one camera, one pack).
        self._dr_thrust_gain = torch.ones(self.num_envs, device=self.device)
        self._dr_drag_gain = torch.zeros(self.num_envs, device=self.device)
        self._dr_cam_bright = torch.ones(self.num_envs, device=self.device)
        self._dr_cam_noise = torch.zeros(self.num_envs, device=self.device)
        # Sensor/actuator draws added 2026-11-18: heading belief and drag anisotropy are
        # per-env constants (one compass offset, one hull), turbidity moves per episode.
        self._imu_yaw_bias = torch.zeros(self.num_envs, device=self.device)
        self._imu_yaw_noise = torch.zeros(self.num_envs, device=self.device)
        self._drag_aniso = torch.ones((self.num_envs, 3), device=self.device)
        self._turbidity = torch.zeros(self.num_envs, device=self.device)
        self._sensed_yaw_cache = torch.zeros(self.num_envs, device=self.device)
        self._sensed_yaw_key = -1
        # Perception-calibration draws. The camera and the wall stay nominal; these two
        # multipliers bias only the numbers the estimator divides/multiplies by, so the
        # scale error is a belief error, not a geometry change.
        self._visodom_f_err = torch.ones(self.num_envs, device=self.device)
        self._visodom_range_err = torch.ones(self.num_envs, device=self.device)
        # Per-episode residual on the scale and the additive offset: redrawn every reset
        # so no single bias survives long enough for the policy to settle into it.
        self._visodom_ep_scale = torch.ones(self.num_envs, device=self.device)
        self._visodom_bias = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._visodom_f_px = torch.full(
            (self.num_envs,), float(self._anchor_f_px), dtype=torch.float32, device=self.device
        )
        self._sample_domain_randomization(torch.arange(self.num_envs, device=self.device), per_episode=False)
        # _visodom_z exists already; rewrite it from the freshly drawn range error.
        self._refresh_visodom_calibration()
        if bool(getattr(self.cfg, "dr_enable", False)):
            print(
                "[v7][DR] enabled  current=%.2f-%.2f m/s (full-circle=%s)  thrust_gain=%.2f-%.2f"
                "  drag=%.2f-%.2f N/(m/s)  cam_bright=%.2f-%.2f  cam_noise=%.3f-%.3f"
                "  spawn_xy=±%.2fm yaw=±%.1f° depth=±%.2fm"
                % (
                    self.cfg.dr_current_speed_range[0], self.cfg.dr_current_speed_range[1],
                    bool(self.cfg.dr_current_dir_full_circle),
                    self.cfg.dr_thrust_gain_range[0], self.cfg.dr_thrust_gain_range[1],
                    self.cfg.dr_drag_gain_range[0], self.cfg.dr_drag_gain_range[1],
                    self.cfg.dr_cam_brightness_range[0], self.cfg.dr_cam_brightness_range[1],
                    self.cfg.dr_cam_noise_std_range[0], self.cfg.dr_cam_noise_std_range[1],
                    float(self.cfg.dr_spawn_xy_range), float(self.cfg.dr_spawn_yaw_range_deg),
                    float(self.cfg.dr_spawn_depth_range),
                ),
                flush=True,
            )
            print(
                "[v7][DR] calibration  visodom f_px=%.3f-%.3f ×, Z(0)=%.3f-%.3f ×"
                "  cam_mount=±%.2f° pitch/roll (wall baked)   -> drawn f_px %.1f-%.1f px,"
                " range_err %.3f-%.3f, tilt max %.2f°"
                % (
                    self.cfg.dr_visodom_focal_rel_range[0], self.cfg.dr_visodom_focal_rel_range[1],
                    self.cfg.dr_visodom_range_rel_range[0], self.cfg.dr_visodom_range_rel_range[1],
                    float(self.cfg.dr_cam_mount_deg_range),
                    float(self._visodom_f_px.min()), float(self._visodom_f_px.max()),
                    float(self._visodom_range_err.min()), float(self._visodom_range_err.max()),
                    float(self._cam_tilt_deg.abs().max()) if hasattr(self, "_cam_tilt_deg") else 0.0,
                ),
                flush=True,
            )
        else:
            print("[v7][DR] disabled (boxed 2026-11-15 configuration)", flush=True)

        # Reward-sharpness banner. These sigmas set the precision the return is ABLE to ask
        # for: with sigma_visodom=1.0 m and sigma_anchor_vis=0.2 m a 5 mm hover and a 38 mm
        # hover differ by ~0.5% of the episode return, which is below the DR-induced noise,
        # so the policy has no gradient to go past ~30 mm (measured 2026-11-16). Printed on
        # every run so the reward configuration of a checkpoint is never a matter of guesswork.
        print(
            "[v7][REW] sigma_flow=%.3f  sigma_anchor_vis=%.3f m  sigma_visodom=%.3f m"
            "  sigma_visodom_obs=%.3f m | rew flow/anchor_vis/sim/yaw/pos = %.2f/%.2f/%.2f/%.2f/%.2f"
            % (
                float(self.cfg.sigma_flow),
                float(self.cfg.sigma_anchor_vis_m),
                float(self.cfg.sigma_visodom_m),
                float(getattr(self.cfg, "sigma_visodom_obs_m", 1.0)),
                float(self.cfg.rew_flow),
                float(self.cfg.rew_anchor_vis),
                float(self.cfg.rew_sim),
                float(self.cfg.rew_yaw_anchor),
                float(self.cfg.rew_anchor),
            ),
            flush=True,
        )
        print(
            "[v7][REW] fine twins: rew_anchor_vis_fine=%.2f (sigma=%.3f m)"
            "  rew_position_fine=%.2f (sigma=%.3f m)  -> max %.2f/step"
            % (
                float(self.cfg.rew_anchor_vis_fine),
                float(self.cfg.sigma_anchor_vis_fine_m),
                float(self.cfg.rew_position_fine),
                float(self.cfg.sigma_visodom_fine_m),
                float(self.cfg.rew_flow + self.cfg.rew_anchor_vis + self.cfg.rew_anchor_vis_fine
                      + self.cfg.rew_sim + self.cfg.rew_yaw_anchor + self.cfg.rew_anchor
                      + self.cfg.rew_position_fine + self.cfg.rew_alive
                      + self.cfg.rew_depth + self.cfg.rew_att),
            ),
            flush=True,
        )

        # USD pose sync for camera.
        self._robot_usd = None
        if bool(getattr(self.cfg, "cam_sync_usd_pose", True)):
            self._init_usd_pose_sync()

        # Ground truth not needed; keep _target_pos as agent pos for compatibility.
        self._target_pos = torch.zeros(self.num_envs, 2, device=self.device)

    def _spawn_robot_camera(self):
        """Spawn a forward-facing camera at the front face center of the proxy cuboid, pointing +X."""
        try:
            from pxr import UsdGeom, Gf
            half_x = 0.56 / 2.0  # proxy cuboid half-length in X
            for i in range(self.num_envs):
                env_p = self.scene.env_prim_paths[i]
                fwd_path = f"{env_p}/Robot/ForwardCamera0"
                fwd = self.sim.stage.DefinePrim(fwd_path, "Camera")
                xf = UsdGeom.Xformable(fwd)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(half_x, 0.0, 0.0))
                # Face +X and keep the image upright:
                #   q_fwd  = -90 deg about Y, turning the default -Z view axis onto +X
                #   q_roll = -90 deg about the camera's own X, bringing local +Y up onto world +Z
                # Product is (0.5, 0.5, -0.5, -0.5): forward=+X, up=+Z, image-right=world -Y.
                q_fwd = Gf.Quatf(0.0, 0.70710678, 0.0, -0.70710678)
                q_roll = Gf.Quatf(0.70710678, -0.70710678, 0.0, 0.0)
                xf.AddOrientOp().Set(q_roll * q_fwd)
                xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
                UsdGeom.Camera(fwd).GetFocalLengthAttr().Set(25.294921875)
        except Exception as e:
            print(f"[gsrov] WARN camera spawn failed: {e}", flush=True)

    def _build_textured_ground(self):
        """Build a simple textured sand ground plane using low-res texture."""
        try:
            from pxr import UsdGeom, UsdShade, Sdf, Gf
            # Try low-res texture first
            tex_paths = [
                "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/ground_lite/sand_01_8k/textures/sand_01_diff_256.jpg",
                "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/textures/ground_diff_4096.png",
            ]
            diff_tex = None
            for p in tex_paths:
                if os.path.isfile(p):
                    diff_tex = p
                    break
            stage = self.sim.stage
            size = 30.0
            tile = 10.0
            half = size * 0.5
            plane = UsdGeom.Mesh.Define(stage, "/World/ground")
            plane.GetPointsAttr().Set([Gf.Vec3f(-half, -half, 0.0), Gf.Vec3f(half, -half, 0.0),
                                       Gf.Vec3f(half, half, 0.0), Gf.Vec3f(-half, half, 0.0)])
            plane.GetFaceVertexCountsAttr().Set([4])
            plane.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
            stp = UsdGeom.PrimvarsAPI(plane).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            stp.Set([Gf.Vec2f(0, 0), Gf.Vec2f(tile, 0), Gf.Vec2f(tile, tile), Gf.Vec2f(0, tile)])
            xf = UsdGeom.Xformable(plane.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.03))
            mp = "/World/Ground/Material"
            mat = UsdShade.Material.Define(stage, mp)
            sh = UsdShade.Shader.Define(stage, mp + "/Preview")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.15, 0.25, 0.65))  # pool-blue
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(plane.GetPrim()).Bind(mat)
            plane.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        except Exception as e:
            print(f"[v7] WARN textured ground failed, fallback to plain: {e}", flush=True)
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    def _default_plane_texture_candidates(self):
        """Discover a preset group of swappable textures (256px diffuse maps)."""
        cands = []
        base = "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/ground_lite"
        if os.path.isdir(base):
            try:
                for d in sorted(os.listdir(base)):
                    texdir = os.path.join(base, d, "textures")
                    if not os.path.isdir(texdir):
                        continue
                    for f in sorted(os.listdir(texdir)):
                        if f.endswith("_diff_256.jpg"):
                            cands.append(os.path.join(texdir, f))
            except OSError:
                pass
        # seabed-level textures
        for p in [
            "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/textures/ground_diff_4096.png",
            "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/textures/ground_detiled_2048.png",
        ]:
            if os.path.isfile(p):
                cands.append(p)
        # rock / collected extras
        for p in [
            "/home/lh/IsaacLab/data_downloads/OceanSim_assets/collected_rock/textures/cropped_rock.jpg",
            "/home/lh/IsaacLab/data_downloads/OceanSim_assets/collected_MHL/textures/mhl_3hz_high_smooth_scaled.jpg",
        ]:
            if os.path.isfile(p):
                cands.append(p)
        return cands

    def _spawn_plane_occluders(self, env_prim, ox, oy, oz, depth, size, dist, half, tile, tex, half_w=None, back=0.75):
        """Spawn side/top/bottom occluder walls forming a pool-shaped open-top box around the front plane.

        Four walls (left / right / bottom / back) bound the scene laterally and vertically; the top is
        left open so the camera can see upward and the agent is not forced to track only the floor plane.
        Vertices are wound with inward-facing normals and the material is double-sided, so walls are
        visible from inside the box.
        """
        try:
            from pxr import UsdGeom, UsdShade, Sdf, Gf
            if half_w is None:
                half_w = half
            half_h = half
            # Coordinates are env-local: the prim sits under /World/envs/env_i, whose own
            # transform already carries the environment origin.
            z_lo = depth - half_h
            z_hi = depth + half_h
            y_lo = -half_w
            y_hi = half_w
            x_lo = -float(back)
            x_hi = dist
            dark_color = Gf.Vec3f(0.15, 0.25, 0.65)  # pool-blue walls
            # Bottom wall uses a deeper blue for visual distinction.
            bottom_color = Gf.Vec3f(0.05, 0.12, 0.45)  # deep pool-blue floor
            # (name, ordered corners, inward normal, color) — left/right/bottom/back walls; top is open.
            walls = [
                # left wall at y=y_lo, inward normal +Y
                ("OccLeft", [
                    Gf.Vec3f(x_lo, y_lo, z_lo),
                    Gf.Vec3f(x_lo, y_lo, z_hi),
                    Gf.Vec3f(x_hi, y_lo, z_hi),
                    Gf.Vec3f(x_hi, y_lo, z_lo),
                ], Gf.Vec3f(0.0, 1.0, 0.0), dark_color),
                # right wall at y=y_hi, inward normal -Y
                ("OccRight", [
                    Gf.Vec3f(x_lo, y_hi, z_lo),
                    Gf.Vec3f(x_hi, y_hi, z_lo),
                    Gf.Vec3f(x_hi, y_hi, z_hi),
                    Gf.Vec3f(x_lo, y_hi, z_hi),
                ], Gf.Vec3f(0.0, -1.0, 0.0), dark_color),
                # bottom wall at z=z_lo, inward normal +Z
                ("OccBottom", [
                    Gf.Vec3f(x_lo, y_lo, z_lo),
                    Gf.Vec3f(x_hi, y_lo, z_lo),
                    Gf.Vec3f(x_hi, y_hi, z_lo),
                    Gf.Vec3f(x_lo, y_hi, z_lo),
                ], Gf.Vec3f(0.0, 0.0, 1.0), bottom_color),
                # back wall at x=x_lo, inward normal +X
                ("OccBack", [
                    Gf.Vec3f(x_lo, y_lo, z_lo),
                    Gf.Vec3f(x_lo, y_lo, z_hi),
                    Gf.Vec3f(x_lo, y_hi, z_hi),
                    Gf.Vec3f(x_lo, y_hi, z_lo),
                ], Gf.Vec3f(1.0, 0.0, 0.0), dark_color),
            ]
            for name, pts, normal, wall_color in walls:
                path = f"{env_prim}/{name}"
                mesh = UsdGeom.Mesh.Define(self.sim.stage, path)
                mesh.GetPointsAttr().Set(pts)
                mesh.GetFaceVertexCountsAttr().Set([4])
                mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
                mesh.GetNormalsAttr().Set([normal] * 4)
                stp = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                stp.Set([Gf.Vec2f(0, 0), Gf.Vec2f(tile, 0), Gf.Vec2f(tile, tile), Gf.Vec2f(0, tile)])
                mat_path = f"{env_prim}/{name}Mat"
                mat = UsdShade.Material.Define(self.sim.stage, mat_path)
                sh = UsdShade.Shader.Define(self.sim.stage, mat_path + "/Preview")
                sh.CreateIdAttr("UsdPreviewSurface")
                # Solid color — no texture for pool walls.
                sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(wall_color)
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
                sh.CreateInput("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
                mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)
                mesh.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        except Exception as e:
            print(f"[v7] WARN occluder spawn failed: {e}", flush=True)

    def _resolve_plane_texture(self, rng_choice=None):
        """Resolve a texture path for the plane."""
        import random
        if isinstance(rng_choice, str) and rng_choice and os.path.isfile(rng_choice):
            return rng_choice
        p = getattr(self.cfg, "plane_texture_path", "") or ""
        if p and os.path.isfile(p):
            base = p
        else:
            envp = os.environ.get("V7_PLANE_TEXTURE")
            if envp and os.path.isfile(envp):
                base = envp
            else:
                cands = [c for c in (getattr(self.cfg, "plane_texture_paths", ()) or ()) if isinstance(c, str) and os.path.isfile(c)]
                if cands:
                    if getattr(self.cfg, "randomize_plane_texture", False):
                        base = random.choice(cands)
                    else:
                        base = cands[0]
                else:
                    # Prefer a textured default (sand) for stable gradients.
                    preferred = "/home/lh/IsaacLab/data_downloads/OceanSim_assets/seabed_demo/ground_lite/sand_01_8k/textures/sand_01_diff_256.jpg"
                    if os.path.isfile(preferred):
                        base = preferred
                    else:
                        dc = self._default_plane_texture_candidates()
                        base = dc[0] if dc else None
        if getattr(self.cfg, "randomize_plane_texture", False):
            cands = [c for c in (getattr(self.cfg, "plane_texture_paths", ()) or ()) if isinstance(c, str) and os.path.isfile(c)]
            if not cands:
                cands = self._default_plane_texture_candidates()
            if cands:
                base = random.choice(cands)
        return base

    def _spawn_cam_mount_dr(self):
        """Draw the per-env camera-mount misalignment, in degrees, shape (N, 2).

        Called once from _setup_scene before the wall mesh exists, because the tilt is
        baked into the geometry and cannot be changed afterwards. Kept separate from
        _sample_domain_randomization (which runs later, from __init__) for that reason.
        """
        span = float(getattr(self.cfg, "dr_cam_mount_deg_range", 0.0))
        if not bool(getattr(self.cfg, "dr_enable", False)) or span <= 0.0:
            self._cam_tilt_deg = torch.zeros((self.num_envs, 2), device=self.device)
            return
        self._cam_tilt_deg = (torch.rand((self.num_envs, 2), device=self.device) * 2.0 - 1.0) * span

    def _dr_texture_for_env(self, ei: int):
        """Deterministic per-env seabed texture when ``dr_texture_choice`` is on.

        One wall asset per env for the whole run (a site does not change its seabed
        mid-dive), seeded from ``cfg.seed`` + env index so the draw is reproducible.
        Returns None when the knob is off, which reproduces the single-asset default.
        """
        if not bool(getattr(self.cfg, "dr_enable", False)):
            return None
        if not bool(getattr(self.cfg, "dr_texture_choice", False)):
            return None
        import random

        cands = [
            c for c in (getattr(self.cfg, "plane_texture_paths", ()) or ()) if isinstance(c, str) and os.path.isfile(c)
        ]
        if not cands:
            cands = self._default_plane_texture_candidates()
        if not cands:
            return None
        seed = int(getattr(self.cfg, "seed", 0) or 0) + 7919 * int(ei)
        return random.Random(seed).choice(cands)

    def _spawn_textured_plane(self):
        """Spawn a 4x4 (configurable) textured plane per env, in front of the AUV camera.

        The plane is defined by 4 corner points (4 spheres' positions conceptually) and
        blocks the camera's view of neighboring environments while providing a textured
        surface whose material can be swapped to test texture effects.
        """
        try:
            from pxr import UsdGeom, UsdShade, Sdf, Gf
            import random
            depth = self.cfg.starting_depth
            size = float(self.cfg.plane_size)
            aspect = float(getattr(self.cfg, "plane_aspect", 1.0)) or 1.0
            dist = float(self.cfg.plane_distance)
            half_w = size * 0.5 * aspect  # horizontal (Y) half-extent
            half_h = size * 0.5           # vertical (Z) half-extent
            tile = float(self.cfg.plane_texture_tile)
            dark_color = Gf.Vec3f(0.15, 0.25, 0.65)  # pool-blue front face
            for ei in range(self.num_envs):
                origin = self.scene.env_origins[ei]
                ox = float(origin[0]); oy = float(origin[1]); oz = float(origin[2])
                env_prim = self.scene.env_prim_paths[ei]
                path = f"{env_prim}/FlowPlane"
                mesh = UsdGeom.Mesh.Define(self.sim.stage, path)
                pts = [
                    Gf.Vec3f(dist, -half_w, depth - half_h),
                    Gf.Vec3f(dist,  half_w, depth - half_h),
                    Gf.Vec3f(dist,  half_w, depth + half_h),
                    Gf.Vec3f(dist, -half_w, depth + half_h),
                ]
                # Camera-mount DR, applied to the wall instead of the camera: on a planar
                # target only the RELATIVE camera-plane transform is observable, so tilting
                # the wall is the same image as tilting the camera — and the camera prim is
                # left untouched. Pitch rotates about world Y (wall leans top toward/away),
                # roll about world X (wall spins about its own normal -> rotated image).
                pitch = roll = 0.0
                tilt = getattr(self, "_cam_tilt_deg", None)
                if tilt is not None and float(tilt[ei].abs().max()) > 0.0:
                    pitch = float(tilt[ei, 0]) * np.pi / 180.0
                    roll = float(tilt[ei, 1]) * np.pi / 180.0
                    cp, sp = float(np.cos(pitch)), float(np.sin(pitch))
                    cr, sr = float(np.cos(roll)), float(np.sin(roll))
                    tilted = []
                    for corner in (-1.0, 1.0):
                        for side in (-1.0, 1.0):
                            dy, dz = side * half_w, corner * half_h
                            # pitch about Y through the wall centre
                            dx_p, dz_p = dz * sp, dz * cp
                            # roll about X (the wall normal)
                            dy_r = dy * cr - dz_p * sr
                            dz_r = dy * sr + dz_p * cr
                            tilted.append(Gf.Vec3f(dist + dx_p, dy_r, depth + dz_r))
                    pts = [tilted[0], tilted[1], tilted[3], tilted[2]]
                mesh.GetPointsAttr().Set(pts)
                mesh.GetFaceVertexCountsAttr().Set([4])
                mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
                stp = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                # UVs: V flipped so image "up" matches world +Z; 180° rotated.
                stp.Set([Gf.Vec2f(tile, 0), Gf.Vec2f(0, 0), Gf.Vec2f(0, tile), Gf.Vec2f(tile, tile)])
                mesh.GetPrim().CreateAttribute("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
                # compute normals — rotated with the wall so the tilt does not fake a
                # shading gradient that the texture mask / Farneback could read as motion
                try:
                    if pitch or roll:
                        _cp, _sp = float(np.cos(pitch)), float(np.sin(pitch))
                        _cr, _sr = float(np.cos(roll)), float(np.sin(roll))
                        n = Gf.Vec3f(-_cp, -_sp * _sr, _sp * _cr)
                    else:
                        n = Gf.Vec3f(-1.0, 0.0, 0.0)
                    mesh.GetNormalsAttr().Set([n] * 4)
                except Exception:
                    pass

                tex = self._resolve_plane_texture(self._dr_texture_for_env(ei))
                mat_path = f"{env_prim}/FlowPlaneMat"
                mat = UsdShade.Material.Define(self.sim.stage, mat_path)
                sh = UsdShade.Shader.Define(self.sim.stage, mat_path + "/Preview")
                sh.CreateIdAttr("UsdPreviewSurface")
                if tex:
                    rd = UsdShade.Shader.Define(self.sim.stage, mat_path + "/stReader")
                    rd.CreateIdAttr("UsdPrimvarReader_float2")
                    rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
                    tx = UsdShade.Shader.Define(self.sim.stage, mat_path + "/Tex")
                    tx.CreateIdAttr("UsdUVTexture")
                    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex))
                    tx.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(rd.ConnectableAPI(), "result")
                    tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                    tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
                else:
                    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(dark_color)
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
                mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)
                mesh.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)

                # Side/top/bottom occluders to reduce neighboring envs entering camera view.
                if getattr(self.cfg, "plane_occluders", True):
                    # Side walls use solid blue (no texture) so the pool walls are visible.
                    self._spawn_plane_occluders(env_prim, ox, oy, oz, depth, size, dist, half_h, tile, None, half_w)
        except Exception as e:
            print(f"[v7] WARN plane spawn failed: {e}", flush=True)

    def _spawn_reference_spheres(self):
        """Spawn 4 bright reference spheres in a 0.5 m × 0.5 m square ahead of the ROV camera.

        The square is centered on the camera axis (y=0, z=starting_depth) at
        x = ref_sphere_offset_x. Corners are at (±half_side) in y and z, so the
        square side is 2*half_side = 0.5 m. Spheres are purely visual
        (no physics) so they do not interfere with the AUV dynamics.
        """
        try:
            from pxr import UsdGeom, UsdShade, Sdf, Gf
            r = float(self.cfg.ref_sphere_radius)
            cx = float(self.cfg.ref_sphere_offset_x)
            half = float(self.cfg.ref_sphere_half_side)
            depth = float(self.cfg.starting_depth)
            color = Gf.Vec3f(*self.cfg.ref_sphere_color)
            offsets = [
                (cx, -half, depth - half),
                (cx, -half, depth + half),
                (cx,  half, depth - half),
                (cx,  half, depth + half),
            ]
            for ei in range(self.num_envs):
                ox = float(self.scene.env_origins[ei][0])
                oy = float(self.scene.env_origins[ei][1])
                oz = float(self.scene.env_origins[ei][2])
                env_prim = self.scene.env_prim_paths[ei]
                for idx, (dx, dy, dz) in enumerate(offsets):
                    sphere_path = f"{env_prim}/RefSphere{idx}"
                    sphere = UsdGeom.Sphere.Define(self.sim.stage, sphere_path)
                    sphere.GetRadiusAttr().Set(r)
                    xf = UsdGeom.Xformable(sphere.GetPrim())
                    xf.ClearXformOpOrder()
                    # Env-local: the parent env transform supplies the origin.
                    xf.AddTranslateOp().Set(Gf.Vec3d(dx, dy, dz))
                    mat_path = f"{sphere_path}/Mat"
                    mat = UsdShade.Material.Define(self.sim.stage, mat_path)
                    sh = UsdShade.Shader.Define(self.sim.stage, mat_path + "/Preview")
                    sh.CreateIdAttr("UsdPreviewSurface")
                    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
                    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(color)
                    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
                    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                    UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(mat)
                    print(f"[v7] RefSphere{idx} env{ei} local @ ({dx:.2f}, {dy:.2f}, {dz:.2f}) r={r}", flush=True)
        except Exception as e:
            print(f"[v7] WARN ref sphere spawn failed: {e}", flush=True)

    def _spawn_view_camera(self):
        """Plain USD camera co-located with the flow camera (viewport / recording only).

        Not a sensor: it allocates no render product, so it does not slow training down.
        Pick it in the Isaac Sim viewport camera list to look through the ROV's eyes.
        """
        if not bool(getattr(self.cfg, "spawn_view_camera", True)):
            return
        try:
            from pxr import UsdGeom, Gf
            hfov = float(self.cfg.cam_hfov_deg)
            aperture = float(self.cfg.cam_aperture_mm)
            focal = float(aperture * 0.5 / np.tan(np.radians(hfov) * 0.5))
            v_aperture = float(aperture * self.cfg.render_height / self.cfg.render_width)
            off = self.cfg.view_camera_offset
            roll = float(np.radians(float(getattr(self.cfg, "view_camera_roll_deg", -90.0))))
            q_fwd = Gf.Quatf(0.0, 0.70710678, 0.0, -0.70710678)  # -90 deg about Y -> view axis +X
            q_roll = Gf.Quatf(
                float(np.cos(roll * 0.5)), float(np.sin(roll * 0.5)), 0.0, 0.0
            )
            name = str(self.cfg.view_camera_name)
            for i in range(self.num_envs):
                path = f"{self.scene.env_prim_paths[i]}/Robot/{name}"
                prim = self.sim.stage.DefinePrim(path, "Camera")
                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(float(off[0]), float(off[1]), float(off[2])))
                xf.AddOrientOp().Set(q_roll * q_fwd)
                xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
                cam = UsdGeom.Camera(prim)
                cam.GetFocalLengthAttr().Set(focal)
                cam.GetHorizontalApertureAttr().Set(aperture)
                cam.GetVerticalApertureAttr().Set(v_aperture)
                cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 200.0))
                if i == 0:
                    print(
                        f"[v7] view camera '{name}' at body offset {tuple(round(float(o), 3) for o in off)} m, "
                        f"looking +X (focal {focal:.2f} mm)",
                        flush=True,
                    )
        except Exception as e:
            print(f"[v7] WARN view camera spawn failed: {e}", flush=True)

    def _apply_camera_intrinsics(self):
        """Match the real camera: HFOV 81.65 deg on a 16:9 sensor.

        The focal length alone fixes HFOV only if the vertical aperture follows the render
        aspect. USD defaults verticalAperture to horizontalAperture, which would keep VFOV ==
        HFOV, so both apertures are set explicitly here.
        """
        from pxr import UsdGeom, Gf
        hfov = float(self.cfg.cam_hfov_deg)
        aperture = float(self.cfg.cam_aperture_mm)
        aspect = float(self.cfg.render_width) / max(float(self.cfg.render_height), 1.0)
        focal = float(aperture * 0.5 / np.tan(np.radians(hfov) * 0.5))
        v_aperture = float(aperture / aspect)
        for i in range(self.num_envs):
            path = f"{self.scene.env_prim_paths[i]}/Robot/ForwardCamera0"
            prim = self.sim.stage.GetPrimAtPath(path)
            if prim is None or not prim.IsValid():
                continue
            cam = UsdGeom.Camera(prim)
            cam.GetFocalLengthAttr().Set(focal)
            cam.GetHorizontalApertureAttr().Set(aperture)
            cam.GetVerticalApertureAttr().Set(v_aperture)
            cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 200.0))
        vfov = 2.0 * np.degrees(np.arctan(v_aperture * 0.5 / focal))
        print(
            f"[v7] camera intrinsics: focal={focal:.3f} mm  hAperture={aperture:.3f} mm  "
            f"vAperture={v_aperture:.3f} mm  HFOV={hfov:.2f} deg  VFOV={vfov:.2f} deg",
            flush=True,
        )

    def _init_usd_pose_sync(self):
        from isaacsim.core.prims import XFormPrim
        robot_paths = [f"{p}/Robot" for p in self.scene.env_prim_paths]
        try:
            self._robot_usd = XFormPrim(robot_paths, reset_xform_properties=False)
        except Exception as exc:
            self._robot_usd = None
            print(f"[v7-hover] WARN: USD pose sync disabled: {exc}", flush=True)

    def _sync_robot_usd_pose(self):
        if self._robot_usd is None:
            return
        self._robot_usd.set_world_poses(
            positions=self._robot.data.root_pos_w,
            orientations=self._robot.data.root_quat_w,
            usd=True,
        )

    def _camera_rgb(self):
        rgb = self._tiled_camera.data.output["rgb"]
        if rgb.dtype != torch.uint8:
            rgb = rgb.clamp(0, 255).to(torch.uint8)
        rgb = rgb[..., :3]
        # Camera domain randomization at RENDER resolution, upstream of everything, so
        # the flow, the template match, the texture mask and the policy frame all see the
        # same imperfect sensor. The exposure/gain spread is fixed per env for the whole
        # run (like a real unit); the read noise is per frame. Cached per sim step because
        # both callers (_update_flow_and_texture and _get_observations) must see the SAME
        # frame — drawing twice would make the policy's RGB and the flow disagree.
        if bool(getattr(self.cfg, "dr_enable", False)) and hasattr(self, "_dr_cam_bright"):
            key = int(getattr(self, "_sim_step_counter", -1))
            if getattr(self, "_dr_frame_key", -1) == key:
                return self._dr_frame_cache
            out = rgb.to(torch.float32) * self._dr_cam_bright.view(-1, 1, 1, 1)
            if float(self.cfg.dr_cam_noise_std_range[1]) > 0.0:
                out = out + torch.randn_like(out) * (self._dr_cam_noise.view(-1, 1, 1, 1) * 255.0)
            turb = self._turbidity.view(-1, 1, 1, 1)
            if bool((turb > 0.0).any()):
                # Scattering water attenuates contrast and softens edges: what reaches the
                # sensor is a washed-out, low-gradient frame. This is real sensor realism
                # and the reason flow RMS can read near-zero while the hull is moving —
                # the policy must learn to distrust a quiet image in murky water.
                mean = out.mean(dim=(1, 2, 3), keepdim=True)
                out = mean + (out - mean) * (1.0 - 0.7 * turb)
                o = F.avg_pool2d(out.permute(0, 3, 1, 2), 3, stride=1, padding=1)
                out = o.permute(0, 2, 3, 1)
            rgb = out.round().clamp_(0.0, 255.0).to(torch.uint8)
            self._dr_frame_key = key
            self._dr_frame_cache = rgb
        return rgb

    def _read_camera_frame(self):
        rgb = self._camera_rgb().permute(0, 3, 1, 2).to(torch.float32)
        if rgb.shape[-2:] != (self.cfg.cam_height, self.cfg.cam_width):
            rgb = F.interpolate(rgb, size=(self.cfg.cam_height, self.cfg.cam_width), mode="area")
        return rgb.round().clamp_(0.0, 255.0).to(torch.uint8)

    def _texture_from_rgb(self, rgb_nhwc):
        # gray gradient magnitude -> texture mask
        gray = rgb_nhwc.float().mean(dim=-1)  # (N,Rh,Rw)
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[:, 1:-1, 1:-1] = gray[:, 1:-1, 2:] - gray[:, 1:-1, :-2]
        gy[:, 1:-1, 1:-1] = gray[:, 2:, 1:-1] - gray[:, :-2, 1:-1]
        mag = torch.sqrt(gx * gx + gy * gy)
        rh, rw = gray.shape[-2], gray.shape[-1]
        h, w = self.cfg.cam_height, self.cfg.cam_width
        # threshold at render res, then MAX-pool down to cam res so thin texture survives.
        hot = (mag > self.cfg.texture_grad_threshold).float().unsqueeze(1)  # (N,1,Rh,Rw)
        if (rh, rw) != (h, w):
            mask = F.adaptive_max_pool2d(hot, (h, w))
        else:
            mask = hot
        count = mask.flatten(1).sum(dim=-1)
        return mask, count

    def _fit_translational_flow(self, du_px, dv_px, tex, fw, fh):
        """Fit du = a + k x, dv = b + k y on textured pixels.

        du, dv and (x, y) are in the same pixel units (flow resolution). a, b are
        the uniform component (motion across the optical axis). k is the radial
        expansion per pixel per frame, equal to v_axial * dt / Z on a
        fronto-parallel plane. Returns a, b, k, each shape (m,).
        """
        m, h, w = du_px.shape
        device, dtype = du_px.device, du_px.dtype
        xs = (torch.arange(w, device=device, dtype=dtype) - (w - 1) * 0.5) * (float(fw) / float(w))
        ys = (torch.arange(h, device=device, dtype=dtype) - (h - 1) * 0.5) * (float(fh) / float(h))
        x = xs.view(1, 1, w).expand(m, h, w)
        y = ys.view(1, h, 1).expand(m, h, w)
        wt = tex.reshape(m, h, w).to(dtype)
        sw = wt.sum(dim=(1, 2)).clamp(min=1.0)
        sx = (wt * x).sum(dim=(1, 2))
        sy = (wt * y).sum(dim=(1, 2))
        sxx = (wt * x * x).sum(dim=(1, 2))
        syy = (wt * y * y).sum(dim=(1, 2))
        su = (wt * du_px).sum(dim=(1, 2))
        sv = (wt * dv_px).sum(dim=(1, 2))
        sux = (wt * du_px * x).sum(dim=(1, 2))
        svy = (wt * dv_px * y).sum(dim=(1, 2))
        k = (sux + svy) / (sxx + syy).clamp(min=1.0)
        a = (su - k * sx) / sw
        b = (sv - k * sy) / sw
        return a, b, k

    def _update_flow_and_texture(self):
        curr_raw = self._camera_rgb()  # (N, render_h, render_w, 3) uint8 at 1280×720
        rh, rw = int(self.cfg.render_height), int(self.cfg.render_width)
        h, w = int(self.cfg.cam_height), int(self.cfg.cam_width)
        # Resolution fed into Farneback — much cheaper than full render res.
        # Final flow is interpolated to (h, w)=64×64 anyway, so we lose nothing useful.
        fh, fw = int(self.cfg.flow_render_height), int(self.cfg.flow_render_width)

        # _texture_from_rgb expects NHWC input at render resolution.
        tex_mask, tex_count = self._texture_from_rgb(curr_raw)
        self._texture_mask = tex_mask
        self._texture_count = tex_count

        fresh = ~self._prev_rgb_ok
        self._flow_image.zero_()
        self._flow_rms[:] = 0.0
        self._flow_uniform[fresh] = 0.0
        self._flow_expand[fresh] = 0.0
        if bool((~fresh).any()):
            # Downsample previous and current frames for flow estimation.
            # Convert NHWC (uint8) → NCHW (float32) → area-downsample → NHWC (uint8).
            # F.interpolate doesn't support Byte type, so cast through float32 first.
            prev_nchw = self._prev_rgb[~fresh].permute(0, 3, 1, 2).contiguous().to(torch.float32)
            curr_nchw = curr_raw[~fresh].permute(0, 3, 1, 2).contiguous().to(torch.float32)
            prev_down = F.interpolate(prev_nchw, size=(fh, fw), mode="area")
            curr_down = F.interpolate(curr_nchw, size=(fh, fw), mode="area")
            prev_down = prev_down.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
            curr_down = curr_down.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
            # Unified flow estimation (RAFT on GPU or Farneback on CPU).
            est = _estimate_flow(prev_down, curr_down,
                                 method=str(self.cfg.flow_method),
                                 device=str(self.cfg.flow_device))
            # Downsample to 64×64.  Deployment normalizes by flow_render_width/height,
            # so we match that convention here for policy consistency.
            small = F.interpolate(est, size=(h, w), mode="area")
            du = small[:, 0, :, :] / float(fw)
            dv = small[:, 1, :, :] / float(fh)
            self._flow_image[~fresh, 0] = du.clamp(-1.0, 1.0)
            self._flow_image[~fresh, 1] = dv.clamp(-1.0, 1.0)
            # flow rms (pixels) over textured pixels for reward
            mag_px = torch.sqrt(small[:, 0, :, :] ** 2 + small[:, 1, :, :] ** 2)
            tex = tex_mask[~fresh].squeeze(1)  # (m, H, W)
            tex_flat = tex.reshape(tex.shape[0], -1)
            mag_flat = mag_px.reshape(mag_px.shape[0], -1)
            has_tex = tex_flat.sum(dim=-1) >= 4
            sq = mag_flat * mag_flat
            rms_tex = torch.sqrt((sq * tex_flat).sum(dim=-1) / tex_flat.sum(dim=-1).clamp(min=1))
            rms_all = torch.sqrt(sq.mean(dim=-1))
            rms = torch.where(has_tex, rms_tex, rms_all)
            self._flow_rms[~fresh] = rms
            # Axial vs lateral split of the same flow field: (a_px, b_px) drive the
            # visodom integrator below, k_exp is the expansion coefficient (k ≈ ΔZ/Z).
            a_px, b_px, k_exp = self._fit_translational_flow(small[:, 0], small[:, 1], tex, fw, fh)
            self._flow_uniform[~fresh, 0] = a_px
            self._flow_uniform[~fresh, 1] = b_px
            self._flow_expand[~fresh] = k_exp
            rows = ~fresh
            yaw_now = self._sensed_yaw()[rows]
            # Visual odometry: same de-rotation, NO leak. The surge leg comes from
            # the expansion coefficient (k per frame ≈ ΔZ/Z): displacement += k·Z,
            # then Z shrinks by 1/(1+k). Sway px→m happens in _visodom_m via
            # f_px and the tracked Z.
            if bool(getattr(self.cfg, "vis_odometry", False)):
                dyaw_o = yaw_now - self._visodom_yaw_prev[rows]
                dyaw_o = torch.atan2(torch.sin(dyaw_o), torch.cos(dyaw_o))
                co, so = torch.cos(dyaw_o), torch.sin(dyaw_o)
                u0 = self._visodom[rows, 0] * co - self._visodom[rows, 1] * so
                v0 = self._visodom[rows, 0] * so + self._visodom[rows, 1] * co
                zz = self._visodom_z[rows]
                # Physical guard: between two control steps the hull cannot travel farther
                # than v_max·dt. Heading bias, wall tilt and turbidity can each push a
                # single flow fit far past that, and an unbounded increment accumulates
                # into metres of phantom travel (visodom_err 0.03 → 1.8 m, arm F). Clamping
                # the increment stops one corrupted frame from poisoning the integrator,
                # the actor's two channels and the position reward at the same time.
                step_m = float(getattr(self.cfg, "visodom_max_speed_mps", 1.0)) * (
                    float(self.cfg.sim.dt) * float(self.cfg.decimation)
                )
                px_lim = (step_m * self._visodom_f_px[rows] / zz.clamp(min=0.1)).clamp(min=1.0)
                a_px = a_px.clamp(-px_lim, px_lim)
                b_px = b_px.clamp(-px_lim, px_lim)
                k_exp = (k_exp * zz).clamp(-step_m, step_m) / zz.clamp(min=0.1)
                self._visodom[rows, 0] = u0 + a_px
                self._visodom[rows, 1] = v0 + b_px
                self._visodom_x[rows] = self._visodom_x[rows] + k_exp * zz
                self._visodom_z[rows] = zz / (1.0 + k_exp).clamp(min=0.1)
                self._visodom_yaw_prev[rows] = yaw_now
        # First-frame anchor: capture (first textured frame) or match (every later frame).
        self._update_anchor_state(curr_raw)
        self._prev_rgb[:] = curr_raw
        self._prev_rgb_ok[:] = True
        self._flow_rms[fresh] = 0.0

    def _update_anchor_state(self, curr_raw):
        """First-frame template match. The algorithm lives in template_anchor.py."""
        local_x = self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]
        yaw = self._sensed_yaw()
        update_template_anchor(
            curr_raw,
            texture_count=self._texture_count,
            min_texture_px=int(self.cfg.min_texture_px),
            anchor_valid=self._anchor_valid,
            anchor_tmpl=self._anchor_tmpl,
            anchor_off_px=self._anchor_off_px,
            anchor_ncc=self._anchor_ncc,
            anchor_disp_m=self._anchor_disp_m,
            anchor_yaw=self._anchor_yaw,
            flow_h=int(self.cfg.flow_render_height),
            flow_w=int(self.cfg.flow_render_width),
            crop_h=int(self.cfg.anchor_crop_height),
            crop_w=int(self.cfg.anchor_crop_width),
            f_px=self._visodom_f_px,
            plane_distance=float(self.cfg.plane_distance) * self._visodom_range_err,
            local_x=local_x,
            yaw=yaw,
        )

    def _sample_domain_randomization(self, env_ids, per_episode: bool) -> None:
        """Draw the domain-randomization knobs for ``env_ids``.

        Two horizons, because the physics differs:
          * ``per_episode=True`` redraws the water current — every dive starts in a new
            flow, which is what the fixed [1, 0.3, 0]·0.3 m/s field never did;
          * ``per_episode=False`` draws the hardware/image terms once — a real unit keeps
            its camera gain and its battery pack for the whole run.
        """
        if not bool(getattr(self.cfg, "dr_enable", False)):
            return
        n = int(env_ids.numel())
        if n == 0:
            return
        dev = self.device

        def _uniform(rng) -> torch.Tensor:
            lo, hi = float(rng[0]), float(rng[1])
            return lo + (hi - lo) * torch.rand(n, device=dev)

        if per_episode:
            speed = _uniform(self.cfg.dr_current_speed_range)
            if bool(self.cfg.dr_current_dir_full_circle):
                ang = (torch.rand(n, device=dev) * 2.0 - 1.0) * float(np.pi)
            else:
                d0 = self.cfg.ocean_current.params.get("direction", [1.0, 0.3, 0.0])
                base = float(np.arctan2(float(d0[1]), float(d0[0])))
                half = float(np.radians(float(self.cfg.dr_current_dir_range_deg)))
                ang = base + (torch.rand(n, device=dev) * 2.0 - 1.0) * half
            self._dr_cur_speed[env_ids] = speed
            self._dr_cur_dir[env_ids, 0] = torch.cos(ang)
            self._dr_cur_dir[env_ids, 1] = torch.sin(ang)
            vf = float(self.cfg.dr_current_vert_frac)
            self._dr_cur_vz[env_ids] = (torch.rand(n, device=dev) * 2.0 - 1.0) * vf * speed
            # Estimator error that moves with the dive rather than with the hull. Kept
            # out of the per_env branch on purpose: a run-long constant is what the
            # policy settles into, which is the failure this DR exists to remove.
            ep = (torch.rand(n, device=dev) * 2.0 - 1.0) * float(self.cfg.dr_visodom_ep_scale_rel)
            self._visodom_ep_scale[env_ids] = 1.0 + ep
            self._visodom_bias[env_ids] = (
                (torch.rand((n, 2), device=dev) * 2.0 - 1.0) * float(self.cfg.dr_visodom_bias_m)
            )
            # Water visibility moves with the dive, so it belongs to the episode horizon.
            self._turbidity[env_ids] = _uniform(self.cfg.dr_turbidity_range)
            # Payload (tier 1): the pod changes between dives, so mass and CG offset belong
            # to the episode horizon. Guards the WHOLE block, not just the draw: the parent
            # __init__ runs one _reset_idx (and therefore this draw) before this subclass has
            # allocated its buffers, so the else-branch would touch a missing attribute.
            if hasattr(self, "_payload_mass"):
                if float(self.cfg.dr_payload_mass_frac_range[1]) > 0.0:
                    self._payload_mass[env_ids] = _uniform(
                        self.cfg.dr_payload_mass_frac_range
                    ) * float(self.cfg.mass)
                else:
                    self._payload_mass[env_ids] = 0.0
                hi_xy = float(self.cfg.dr_payload_offset_xy_range[1])
                if hi_xy > 0.0:
                    self._payload_off[env_ids, 0:2] = (
                        torch.rand((n, 2), device=dev) * 2.0 - 1.0
                    ) * hi_xy
                else:
                    self._payload_off[env_ids, 0:2] = 0.0
                lo_z = float(self.cfg.dr_payload_offset_z_range[0])
                hi_z = float(self.cfg.dr_payload_offset_z_range[1])
                if hi_z > lo_z:
                    self._payload_off[env_ids, 2] = _uniform(self.cfg.dr_payload_offset_z_range)
                else:
                    self._payload_off[env_ids, 2] = lo_z
            self._refresh_visodom_calibration(env_ids)
        else:
            self._dr_thrust_gain[env_ids] = _uniform(self.cfg.dr_thrust_gain_range)
            self._dr_drag_gain[env_ids] = _uniform(self.cfg.dr_drag_gain_range)
            self._dr_cam_bright[env_ids] = _uniform(self.cfg.dr_cam_brightness_range)
            self._dr_cam_noise[env_ids] = _uniform(self.cfg.dr_cam_noise_std_range)
            # Heading belief of this unit: one compass offset for the run, one noise level.
            self._imu_yaw_bias[env_ids] = (
                torch.rand(n, device=dev) * 2.0 - 1.0
            ) * float(np.radians(float(self.cfg.dr_imu_yaw_bias_deg)))
            self._imu_yaw_noise[env_ids] = torch.rand(n, device=dev) * float(
                np.radians(float(self.cfg.dr_imu_yaw_noise_deg))
            )
            # Added-mass anisotropy: one hull, one set of axis multipliers, for the run.
            aniso = (torch.rand((n, 3), device=dev) * 2.0 - 1.0) * float(
                self.cfg.dr_drag_aniso_range
            )
            self._drag_aniso[env_ids] = (1.0 + aniso).clamp(min=0.0)
            # Calibration beliefs: same horizon as the camera itself (one unit, one
            # intrinsics calibration, one standoff estimate, for the whole run).
            self._visodom_f_err[env_ids] = _uniform(self.cfg.dr_visodom_focal_rel_range)
            self._visodom_range_err[env_ids] = _uniform(self.cfg.dr_visodom_range_rel_range)
            self._refresh_visodom_calibration(env_ids)

    def _refresh_visodom_calibration(self, env_ids=None) -> None:
        """Rewrite the metric conversions the estimator believes in, from the drawn errors.

        Called once at init and again on every reset, so a run cannot start with a
        _visodom_z that disagrees with the multiplier it was drawn from.
        """
        ids = slice(None) if env_ids is None else env_ids
        self._visodom_f_px[ids] = (
            float(self._anchor_f_px) * self._visodom_f_err[ids] * self._visodom_ep_scale[ids]
        )
        self._visodom_z[ids] = float(self.cfg.plane_distance) * self._visodom_range_err[ids]

    def _sensed_yaw(self) -> torch.Tensor:
        """Heading the estimator is allowed to see, shape (N,).

        Ground truth plus this unit's compass offset and per-frame noise, so de-rotation
        and the anchor window stop being free information. A wrong dyaw rotates the whole
        visodom integral, which is exactly the failure a real compass bias produces.
        Cached per sim step: every caller within one step must agree, otherwise the flow
        de-rotation and the anchor bookkeeping would drift apart.
        """
        key = int(getattr(self, "_sim_step_counter", -1))
        if self._sensed_yaw_key == key:
            return self._sensed_yaw_cache
        yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)[2]
        if bool(getattr(self.cfg, "dr_enable", False)):
            yaw = yaw + self._imu_yaw_bias
            if float(getattr(self.cfg, "dr_imu_yaw_noise_deg", 0.0)) > 0.0:
                yaw = yaw + torch.randn_like(yaw) * self._imu_yaw_noise
        self._sensed_yaw_cache = yaw
        self._sensed_yaw_key = key
        return yaw

    def _payload_wrench(self):
        """Body-frame weight force and CG torque of the payload (tier-1 emulation).

        F_world = [0, 0, -m_p g]; the wrench is applied at the link origin, so a CG offset r
        shows up as tau = r x F_body. Buoyancy is untouched on purpose: a real tool pod is
        not neutrally buoyant, and the depth loop paying for that is the point.
        """
        zero = torch.zeros((self.num_envs, 3), device=self.device)
        # hasattr: the parent __init__ drives one step before this subclass has allocated
        # its buffers, so the payload is undefined during construction.
        if not hasattr(self, "_payload_mass") or float(self._payload_mass.abs().max()) <= 0.0:
            return zero, zero
        f_world = torch.zeros_like(zero)
        f_world[:, 2] = -self._payload_mass * 9.81
        f_body = quat_apply_inverse(self._robot.data.root_quat_w, f_world)
        return f_body, torch.linalg.cross(self._payload_off, f_body)

    def _apply_vision_delay(self, image):
        """Return the visual observation captured ``_vis_delay_steps`` control steps ago.

        Newest frame is pushed at the end and read from the front, mirroring the action
        history convention. The push is gated on ``_vis_push_needed`` (set once per control
        step in ``_pre_physics_step``) so the extra ``_get_observations`` calls that happen
        on reset cannot advance the queue twice and shrink the effective latency.
        """
        if self._vis_hist is None:
            self._vis_hist = torch.zeros(
                (self.num_envs, self._vis_delay_steps + 1) + tuple(image.shape[1:]),
                device=image.device,
                dtype=image.dtype,
            )
        if bool(self._vis_fresh.any()):
            # A freshly reset env has no history; duplicating the current frame is the only
            # honest choice (the alternative — a zero frame — injects a fake blackout).
            self._vis_hist[self._vis_fresh] = image[self._vis_fresh].unsqueeze(1)
            self._vis_fresh[:] = False
        if self._vis_push_needed:
            self._vis_push_needed = False
            self._vis_hist[:, :-1] = self._vis_hist[:, 1:].clone()
            self._vis_hist[:, -1] = image
        if self._vis_delay_steps <= 0:
            return image
        return self._vis_hist[:, 0]

    def _apply_action(self):
        """v7 applies PID (cmd_vel [surge, yaw, heave]) → PWM → thrust/moments."""
        dt = float(self.cfg.sim.dt)
        t = None
        dr_on = bool(getattr(self.cfg, "dr_enable", False))
        if dr_on:
            # Per-episode uniform current, drawn in _reset_idx. This bypasses the
            # OceanCurrentCfg field on purpose: under DR the per-episode draw IS the
            # field, so a fixed direction/strength would just add a second, contradictory
            # source of truth. The ramp is kept — frame 1 is still water, matching the
            # handover story (thrusters already holding as the current builds).
            t = self.episode_length_buf.to(dtype=torch.float32) * (dt * float(self.cfg.decimation))
            v = torch.stack(
                [
                    self._dr_cur_dir[:, 0] * self._dr_cur_speed,
                    self._dr_cur_dir[:, 1] * self._dr_cur_speed,
                    self._dr_cur_vz,
                ],
                dim=-1,
            )
            ramp_s = float(getattr(self.cfg, "current_ramp_s", 0.0))
            if ramp_s > 0.0:
                v = v * (t / ramp_s).clamp(0.0, 1.0).unsqueeze(-1)
            self.set_ocean_currents(None, v)
        elif getattr(self, "_ocean_current_field", None) is not None:
            # episode_length_buf counts policy steps and is still 0 during the first
            # step's physics, so t = 0 at the handover frame. Multiply by the control
            # period, not sim.dt: sim.dt would stall the clock at ~2 s in a 15 s episode.
            t = self.episode_length_buf.to(dtype=torch.float32) * (dt * float(self.cfg.decimation))
            v = self._ocean_current_field(self._robot.data.root_pos_w, t)
            ramp_s = float(getattr(self.cfg, "current_ramp_s", 0.0))
            if ramp_s > 0.0:
                v = v * (t / ramp_s).clamp(0.0, 1.0).unsqueeze(-1)
            self.set_ocean_currents(None, v)

        # PID at physics rate. Depth setpoint is fixed at the spawn depth (= the
        # first-frame depth); the PI branch holds it without any policy authority.
        self.pid.depth_target[:] = float(self.cfg.starting_depth)
        self._pwm[:] = self.pid.compute(
            self._cmd_vel,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_w,
            self._robot.data.root_ang_vel_w,
            dt=dt,
        )
        if dr_on:
            # Command-space thrust gain: motor thrust follows u·|u|, so multiplying the
            # normalised command by g scales delivered thrust by g² — the same order as a
            # pack sagging from 24 V to ~22 V.
            self._pwm.mul_(self._dr_thrust_gain[:, None])
        thrust, moment = self._compute_dynamics(self._pwm)
        if dr_on and float(self.cfg.dr_drag_gain_range[1]) > 0.0:
            # Extra linear drag perturbation on the hull [N per (m/s), per kg-free]. The
            # real added-mass/viscous coefficients are estimates, so perturb the damping
            # instead of rescaling the buoyancy/drag split inside the force model.
            thrust = (
                thrust
                - self._dr_drag_gain[:, None]
                * self._drag_aniso
                * self._robot.data.root_lin_vel_w[:, :3]
            )
        # Payload (tier 1): constant weight at a body-frame CG offset. No-op when the DR
        # ranges are all zero, so default runs stay bit-identical.
        if hasattr(self, "_payload_mass") and float(self._payload_mass.abs().max()) > 0.0:
            f_p, tau_p = self._payload_wrench()
            thrust = thrust + f_p
            moment = moment + tau_p
        self._thrust[:, 0, :] = thrust
        self._moment[:, 0, :] = moment
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

        # USD sync on render iterations.
        interval = max(int(self.cfg.sim.render_interval), 1)
        on_render = self._sim_step_counter % interval == 0
        if self._robot_usd is not None and on_render:
            self._sync_robot_usd_pose()

    def _compute_is_col(self):
        # No collisions in open pool — always False.
        # Kept as a stub for API compatibility with parent class.
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _visodom_m(self) -> torch.Tensor:
        """Visual-odometry displacement estimate, current body frame, metres, (N, 2).

        Column 0 is along the optical axis (from the expansion coefficient), column 1
        is lateral (u_px / f_px · Z — fronto-parallel geometry with the tracked Z).
        Returns zeros when vis_odometry is off, so callers can always concatenate.
        """
        if not bool(getattr(self.cfg, "vis_odometry", False)):
            return torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        y_m = self._visodom[:, 0] * self._visodom_z / self._visodom_f_px
        # The additive bias is applied at READ time, not baked into the integrator: it is
        # what the estimator reports, so the actor's 2 channels, the position reward and
        # visodom_err_m must all see the same number.
        return torch.stack([self._visodom_x, y_m], dim=-1) + self._visodom_bias

    def _ego_vel_state(self):
        _, agent_yaw = self._agent_pose_2d()
        vel_w = self._robot.data.root_lin_vel_w
        cy, sy = torch.cos(agent_yaw), torch.sin(agent_yaw)
        v_surge = cy * vel_w[:, 0] + sy * vel_w[:, 1]
        v_sway = -sy * vel_w[:, 0] + cy * vel_w[:, 1]
        v_heave = vel_w[:, 2]
        yaw_rate = self._robot.data.root_ang_vel_w[:, 2]
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        depth = self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth
        # `present` + texture fraction: reward gates flow_term/alive on present, but the
        # old 9-D state hid it — a blind drifting ROV (present=0, flow_rms≈0) aliased onto
        # a perfect hover (present=1, flow_rms=0) and made Q unresolvable. Indices 0–8 are
        # unchanged so existing debug tooling reading st[..., 8] keeps working.
        present = (self._texture_count >= self.cfg.min_texture_px).to(torch.float32)
        tex_frac = (self._texture_count / float(self.cfg.cam_height * self.cfg.cam_width)).clamp(0.0, 1.0)
        state = torch.stack([
            depth.clamp(-10.0, 10.0),
            roll.clamp(-np.pi, np.pi),
            pitch.clamp(-np.pi, np.pi),
            yaw.clamp(-np.pi, np.pi),
            v_surge.clamp(-3.0, 3.0),
            v_sway.clamp(-3.0, 3.0),
            v_heave.clamp(-3.0, 3.0),
            yaw_rate.clamp(-3.0, 3.0),
            # flow_rms is in flow-res pixels (Farneback output), normalized by cam_width (64)
            # to [0,1] so the actor sees a meaningful, non-saturated value (matches the reward term).
            (self._flow_rms / 64.0).clamp(0.0, 1.0),
            present,
            tex_frac,
            # First-frame anchor: offset (du, dv) normalized to ±1 at the match-window
            # edge, match score sim=(ncc+1)/2, yaw error to the capture heading.
            # Indices 0–10 unchanged so existing tooling reading st[..., 8] keeps working.
            (self._anchor_off_px[:, 0] / self._anchor_margin[0]).clamp(-1.0, 1.0),
            (self._anchor_off_px[:, 1] / self._anchor_margin[1]).clamp(-1.0, 1.0),
            ((self._anchor_ncc + 1.0) * 0.5).clamp(0.0, 1.0),
            torch.atan2(torch.sin(yaw - self._anchor_yaw), torch.cos(yaw - self._anchor_yaw)),
            # Visual odometry displacement (m): the deployable position estimate.
            # Clipped ±10 m like the depth channel; zeros when vis_odometry is off.
            (self._visodom_m()[:, 0]).clamp(-10.0, 10.0),
            (self._visodom_m()[:, 1]).clamp(-10.0, 10.0),
        ], dim=-1)
        # Ablation zeros the channels the networks read. The values above are
        # still computed so the logs keep the real estimates.
        if bool(self.cfg.ablate_flow):
            state[:, 8] = 0.0
        if bool(self.cfg.ablate_template):
            state[:, 11:15] = 0.0
        if bool(self.cfg.ablate_visodom):
            state[:, 15:17] = 0.0
        return state

    def _get_observations(self):
        h, w = self.cfg.cam_height, self.cfg.cam_width

        rgb = self._read_camera_frame().to(torch.float32) / 255.0

        flow_mag = torch.sqrt(self._flow_image[:, 0] ** 2 + self._flow_image[:, 1] ** 2).unsqueeze(1)

        # First-frame anchor offset as 2 broadcast (constant-over-space) channels —
        # replaces the leaky drift integral (it forgot with τ=3 s and was yaw-gamed to
        # 0 by the 287k model). ±1 = scene displaced to the edge of the match window
        # (≈±0.9 m lateral @ Z=2 m): the actor's goal-directed "which way is frame 1"
        # cue. +u = scene moved right in the image = ROV must move left.
        mx, my = self._anchor_margin
        anchor_ch = torch.stack([
            (self._anchor_off_px[:, 0] / mx).clamp(-1.0, 1.0),
            (self._anchor_off_px[:, 1] / my).clamp(-1.0, 1.0),
        ], dim=-1)[:, :, None, None].expand(-1, -1, h, w)
        # Visodom broadcast: integrated consecutive-flow displacement, metres / sigma,
        # ±1 clamped. Unlike the anchor channels this keeps counting past the
        # template-match window; ±1 ≈ the reward basin (sigma_visodom_m).
        sig_vo = float(getattr(self.cfg, "sigma_visodom_obs_m", 1.0)) or 1.0
        vo = self._visodom_m()
        visodom_ch = torch.stack([
            (vo[:, 0] / sig_vo).clamp(-1.0, 1.0),
            (vo[:, 1] / sig_vo).clamp(-1.0, 1.0),
        ], dim=-1)[:, :, None, None].expand(-1, -1, h, w)

        flow_obs = self._flow_image
        mag_obs = flow_mag
        if bool(self.cfg.ablate_flow):
            flow_obs = torch.zeros_like(self._flow_image)
            mag_obs = torch.zeros_like(flow_mag)
        if bool(self.cfg.ablate_template):
            anchor_ch = torch.zeros_like(anchor_ch)
        if bool(self.cfg.ablate_visodom):
            visodom_ch = torch.zeros_like(visodom_ch)

        image = torch.cat([
            rgb,
            flow_obs,
            mag_obs,
            self._texture_mask,
            anchor_ch,
            visodom_ch,
        ], dim=1)

        state = self._ego_vel_state()

        if self._vis_delay_steps > 0:
            image = self._apply_vision_delay(image)

        return {"policy": {"image": image, "state": state}}

    def _compute_step_reward(self):
        present = (self._texture_count >= self.cfg.min_texture_px).to(torch.float32)
        # flow_rms_n: RMS normalized by cam_width (64), ∈ [0,1]
        # Flow is measured against the textured plane, which is fixed to the GROUND, so
        # zero flow == ground-locked hover even under ocean current: the policy learns to
        # push upstream until the net (ground-frame) motion vanishes. This is the intended
        # "hold station against the current" objective — driven entirely by the flow cue.
        flow_rms_n = (self._flow_rms / 64.0).clamp(0.0, 1.0)
        flow_term = present * torch.exp(-(flow_rms_n ** 2) / (self.cfg.sigma_flow ** 2))

        # First-frame visual anchor — the goal-directed hover target. Distance term uses
        # the template-match offset converted to metres (in _update_anchor_state); sim
        # term is the match score itself ("current frame looks like frame 1"). Both
        # present-gated: a blind ROV cannot match frame 1 and must not be paid for a
        # garbage match.
        anchor_vis_term = present * torch.exp(
            -(self._anchor_disp_m ** 2) / (self.cfg.sigma_anchor_vis_m ** 2)
        )
        anchor_vis_fine_term = present * torch.exp(
            -(self._anchor_disp_m ** 2) / (float(self.cfg.sigma_anchor_vis_fine_m) ** 2)
        )
        sim_term = present * ((self._anchor_ncc + 1.0) * 0.5).clamp(0.0, 1.0)
        # Heading to the anchor frame: the template match (and "same image as frame 1")
        # is only well-defined at the capture heading — reward re-aligning yaw.
        yaw_now_r = self._sensed_yaw()
        yaw_err_anchor = torch.atan2(
            torch.sin(yaw_now_r - self._anchor_yaw), torch.cos(yaw_now_r - self._anchor_yaw)
        )
        yaw_anchor_term = torch.exp(-(yaw_err_anchor ** 2) / (self.cfg.sigma_yaw_anchor ** 2))

        # Privileged anchor: horizontal distance from the episode spawn origin (ground
        # truth, sim state). With vis_odometry=True this is LOGGING-ONLY — the reward
        # pays the visodom term instead; the GT stays in TensorBoard (disp_end_m,
        # visodom_err_m) as the untouched evaluation metric.
        disp_xy = self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        d_anchor = torch.norm(disp_xy, dim=-1)
        anchor_term = torch.exp(-(d_anchor ** 2) / (self.cfg.sigma_anchor_m ** 2))
        if bool(self.cfg.ablate_visodom):
            # Cue removed. Do not substitute the GT term: that would measure a
            # different question (privileged position) instead of "no visodom".
            position_term = torch.zeros_like(anchor_term)
            position_fine_term = torch.zeros_like(anchor_term)
        elif bool(getattr(self.cfg, "vis_odometry", False)):
            # Which displacement gets paid. See rew_position_source in the cfg block: "gt"
            # pays the true offset from this episode's spawn point, "visodom" pays the
            # estimator's integral (the pre-2026-11-18 behaviour). The actor sees only the
            # estimate either way, so "gt" does not make the task easier — it makes the
            # reward fitable, which is what the harder DR distribution needs.
            if str(getattr(self.cfg, "rew_position_source", "gt")).lower() == "visodom":
                d_reward = self._visodom_m().norm(dim=-1)
            else:
                d_reward = torch.norm(self._robot.data.root_pos_w[:, :2] - self._spawn_xy, dim=-1)
            position_term = present * torch.exp(
                -(d_reward ** 2) / (float(self.cfg.sigma_visodom_m) ** 2)
            )
            # Fine twin. Only meaningful near zero, and that is the point: the estimator
            # itself degrades far out (measured 0.586 m error at a 0.87 m excursion vs
            # 1-6 mm while hovering), so the broad term owns the basin and this one owns
            # the last millimetre.
            position_fine_term = present * torch.exp(
                -(d_reward ** 2) / (float(self.cfg.sigma_visodom_fine_m) ** 2)
            )
        else:
            position_term = anchor_term
            position_fine_term = anchor_term
        w_flow = 0.0 if bool(self.cfg.ablate_flow) else float(self.cfg.rew_flow)
        w_anchor_vis = 0.0 if bool(self.cfg.ablate_template) else float(self.cfg.rew_anchor_vis)
        w_sim = 0.0 if bool(self.cfg.ablate_template) else float(self.cfg.rew_sim)
        w_yaw = 0.0 if bool(self.cfg.ablate_template) else float(self.cfg.rew_yaw_anchor)
        w_position = 0.0 if bool(self.cfg.ablate_visodom) else float(self.cfg.rew_anchor)
        # Fine twins follow the same ablation switches as their broad counterparts, so a
        # cue-removal run still removes the whole cue rather than half of it.
        w_anchor_vis_fine = 0.0 if bool(self.cfg.ablate_template) else float(self.cfg.rew_anchor_vis_fine)
        w_position_fine = 0.0 if bool(self.cfg.ablate_visodom) else float(self.cfg.rew_position_fine)

        # per-episode diagnostics (flushed in _reset_idx)
        self._ep_flow_rms_sum += flow_rms_n
        self._ep_present_sum += present
        self._ep_flow_term_sum += flow_term
        self._ep_anchor_sum += d_anchor
        self._ep_anchor_vis_sum += self._anchor_disp_m
        self._ep_anchor_sim_sum += sim_term
        self._ep_anchor_valid_sum += self._anchor_valid.to(torch.float32)

        depth = self._robot.data.root_pos_w[:, 2]
        depth_err = depth - self.cfg.starting_depth
        depth_term = torch.exp(-(depth_err ** 2) / (self.cfg.sigma_depth ** 2))

        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        # Attitude error is relative to the WORK attitude (level when level_attitude,
        # free-floating trim otherwise), not to whatever the hull drifts to.
        roll_err = roll - self._att_roll_target
        pitch_err = pitch - self._att_pitch_target
        att_err = torch.sqrt(roll_err * roll_err + pitch_err * pitch_err)
        att_term = torch.exp(-(att_err ** 2) / (self.cfg.sigma_att ** 2))

        alive = present
        # action smoothness on raw action (杆量), not remapped cmd_vel
        da = self._actions - self._prev_actions
        smooth = torch.sum(da ** 2, dim=-1)
        # Both terms below are BOUNDED, so neither can dominate the reward the way the old
        # unbounded hinge did. rate_pen sits in [0, 1] and effort in [0, 1].
        rate_scale = float(getattr(self.cfg, "action_rate_scale", 0.15))
        rate_pen = torch.mean(1.0 - torch.exp(-((da / rate_scale) ** 2)), dim=-1)
        effort = torch.mean(self._actions.abs(), dim=-1)

        rew = (
            w_flow * flow_term
            + w_anchor_vis * anchor_vis_term
            + w_anchor_vis_fine * anchor_vis_fine_term
            + w_sim * sim_term
            + w_yaw * yaw_anchor_term
            + w_position * position_term
            + w_position_fine * position_fine_term
            + self.cfg.rew_alive * alive
            + self.cfg.rew_depth * depth_term
            + self.cfg.rew_att * att_term
            - self.cfg.rew_action * smooth
            - float(getattr(self.cfg, "rew_action_rate", 0.0)) * rate_pen
            + float(getattr(self.cfg, "rew_effort", 0.0)) * torch.exp(-effort)
        )
        self._ep_action_delta_sum += torch.sqrt(torch.mean(da ** 2, dim=-1))
        self._ep_rate_pen_sum += rate_pen
        self._ep_effort_sum += effort
        rew = torch.nan_to_num(rew, nan=0.0, posinf=5.0, neginf=0.0)

        # texture-lost streak
        self._texture_lost_streak = torch.where(
            present > 0.5,
            torch.zeros_like(self._texture_lost_streak),
            self._texture_lost_streak + 1,
        )
        return rew

    def _get_dones(self):
        # Update flow/texture (after physics+render) so reward uses current frame.
        self._update_flow_and_texture()
        self._cached_rew[:] = self._compute_step_reward()

        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        tex_end = self._texture_lost_streak >= self.cfg.texture_lost_limit_steps
        terminated = terminated | tex_end
        depth_err = (self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth).abs()
        depth_end = depth_err > self.cfg.depth_limit
        terminated = terminated | depth_end
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        roll_err = roll - self._att_roll_target
        pitch_err = pitch - self._att_pitch_target
        att_err = torch.sqrt(roll_err * roll_err + pitch_err * pitch_err)
        att_end = att_err > self.cfg.att_limit
        terminated = terminated | att_end
        terminated = terminated | self._compute_is_col()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # record which condition ended this episode (cause order: tex/depth/att/timeout)
        if hasattr(self, "_ep_cause"):
            self._ep_cause[:, 0] |= tex_end
            self._ep_cause[:, 1] |= depth_end
            self._ep_cause[:, 2] |= att_end
            self._ep_cause[:, 3] |= truncated
        return terminated, truncated

    def _get_rewards(self):
        return self._cached_rew

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        # Flush per-episode diagnostics BEFORE the parent reset zeroes episode_length_buf.
        # Sb3VecEnvWrapper (fast_variant) copies extras["log"] into the terminating envs'
        # info["episode"] dict, and EpisodeRewardLogCallback writes it to TensorBoard as
        # Episode_Reward/<key>. Without this, a collapse shows only a bare ep_rew_mean drop.
        if ids.numel() > 0 and hasattr(self, "_ep_flow_rms_sum"):
            ep_len = self.episode_length_buf[ids].clamp(min=1).to(torch.float32)
            # End-of-episode horizontal displacement from spawn (before the parent reset
            # teleports the root back) — the real station-keeping metric.
            # Two distances on purpose — never redefine a logged key in place, or the
            # runs that already emitted it stop being comparable:
            #  - disp_end_m   : from the env ORIGIN. Semantics unchanged since the first
            #                   ablation run, so the pre-DR table stays valid.
            #  - disp_spawn_m : from THIS episode's spawn = the frame-1 template's home.
            #                   This is the honest station-keeping error once
            #                   dr_spawn_xy_range is nonzero: the origin-based value has a
            #                   ~0.1 m floor that is spawn offset, not control error.
            d_origin = torch.norm(
                self._robot.data.root_pos_w[ids, :2] - self.scene.env_origins[ids, :2], dim=-1
            )
            d_end = torch.norm(self._robot.data.root_pos_w[ids, :2] - self._spawn_xy[ids], dim=-1)
            # Visodom estimate at the same instant — logged next to the GT so the
            # estimator's error is on TensorBoard every episode.
            vo_end = self._visodom_m()[ids].norm(dim=-1)
            self.extras["log"] = {
                "flow_rms_n": float((self._ep_flow_rms_sum[ids] / ep_len).mean()),
                "present_frac": float((self._ep_present_sum[ids] / ep_len).mean()),
                "flow_term": float((self._ep_flow_term_sum[ids] / ep_len).mean()),
                # Is the action-rate penalty actually biting? Without these two, "we
                # added a smoothing term" would be an assumption, not a measurement.
                "action_delta_rms": float((self._ep_action_delta_sum[ids] / ep_len).mean()),
                "action_rate_pen": float((self._ep_rate_pen_sum[ids] / ep_len).mean()),
                "effort_mean": float((self._ep_effort_sum[ids] / ep_len).mean()),
                # Domain-randomization draw of this episode: you cannot tell a policy that
                # copes with a 0.45 m/s beam current from one that only saw 0.2 m/s
                # head-on unless the sampled current is on the same curve as disp_end_m.
                "dr_cur_speed": float(self._dr_cur_speed[ids].mean()),
                "dr_cur_deg": float(
                    torch.atan2(self._dr_cur_dir[ids, 1], self._dr_cur_dir[ids, 0]).mean() * 57.2957795
                ),
                "disp_m": float((self._ep_anchor_sum[ids] / ep_len).mean()),
                # Legacy key: from the env origin, unchanged semantics since the first
                # ablation run. Under spawn DR its floor is the spawn offset (~0.1 m).
                "disp_end_m": float(d_origin.mean()),
                # Honest station-keeping error: from this episode's spawn, which is where
                # the frame-1 template was captured. Equals disp_end_m when DR is off.
                "disp_spawn_m": float(d_end.mean()),
                "visodom_end_m": float(vo_end.mean()),
                "visodom_err_m": float((vo_end - d_end).abs().mean()),
                # Calibration actually in force, so a sensitivity point is never guessed
                # from the command line that produced it.
                "visodom_f_err": float(self._visodom_f_err[ids].mean()),
                "visodom_range_err": float(self._visodom_range_err[ids].mean()),
                "cam_tilt_deg": (
                    float(self._cam_tilt_deg[ids].abs().mean()) if hasattr(self, "_cam_tilt_deg") else 0.0
                ),
                "anchor_disp_m": float((self._ep_anchor_vis_sum[ids] / ep_len).mean()),
                "anchor_sim": float((self._ep_anchor_sim_sum[ids] / ep_len).mean()),
                # caps at 1.0 = template captured on time; 0 = capture refused all episode
                # (textureless crop), in which case anchor_sim is 0 by construction.
                "anchor_valid_frac": float((self._ep_anchor_valid_sum[ids] / ep_len).mean()),
                "term_tex_lost": float(self._ep_cause[ids, 0].to(torch.float32).mean()),
                "term_depth": float(self._ep_cause[ids, 1].to(torch.float32).mean()),
                "term_att": float(self._ep_cause[ids, 2].to(torch.float32).mean()),
                "term_timeout": float(self._ep_cause[ids, 3].to(torch.float32).mean()),
            }
            self._ep_flow_rms_sum[ids] = 0.0
            self._ep_present_sum[ids] = 0.0
            self._ep_flow_term_sum[ids] = 0.0
            self._ep_action_delta_sum[ids] = 0.0
            self._ep_rate_pen_sum[ids] = 0.0
            self._ep_effort_sum[ids] = 0.0
            self._ep_anchor_sum[ids] = 0.0
            self._ep_anchor_vis_sum[ids] = 0.0
            self._ep_anchor_sim_sum[ids] = 0.0
            self._ep_anchor_valid_sum[ids] = 0.0
            self._ep_cause[ids] = False
        try:
            DirectRLEnv._reset_idx(self, env_ids)
        except Exception:
            pass
        if not hasattr(self, "_default_root_state"):
            return
        # Per-episode water draw (new current every dive) and spawn offsets: the ROV
        # starts at a hand-flown hover, so the captured frame 1 is not the nominal one.
        self._sample_domain_randomization(ids, per_episode=True)
        # EVALUATION ONLY. The per-run knobs (thrust gain, drag, camera noise/brightness,
        # compass bias, mount angle) are drawn once per env at init, so N episodes spread
        # over 4 envs are really 4 hardware samples x N/4 episodes: the effective sample
        # size is 4, not N, and every confidence interval computed from them is optimistic.
        # Measured 2026-11-19 on the same K150k weights: env1 est_err median 9 mm vs env3
        # 38 mm, end median 21 vs 50 -- that spread is the hardware draw, not the policy.
        # Redrawing per episode turns each episode into an independent sample of the unit
        # distribution, which is the quantity the gate is trying to estimate.
        # Default False, so training behaviour is bit-identical to before.
        if bool(getattr(self.cfg, "eval_redraw_per_run_dr", False)):
            self._sample_domain_randomization(ids, per_episode=False)
        # Hold the actuator delay for the whole episode: a fixed offset per dive, so the
        # policy must compensate for a lag it cannot average away, exactly like the real
        # vehicle whose delay does not change between control ticks.
        _delay_max = int(getattr(self.cfg, "train_action_delay_max_steps", 0))
        _delay_fix = int(getattr(self.cfg, "train_action_delay_fixed_steps", -1))
        # The parent __init__ runs one _reset_idx before this subclass has built its buffers,
        # so skip the draw there — the buffer already starts at zero, which is the same state.
        if hasattr(self, "_act_delay_steps"):
            if _delay_fix >= 0:
                self._act_delay_steps[ids] = _delay_fix
            elif _delay_max > 0:
                self._act_delay_steps[ids] = torch.randint(
                    0, _delay_max + 1, (int(ids.numel()),), device=self.device
                )
            else:
                self._act_delay_steps[ids] = 0
            self._action_hist[ids] = 0.0
            if hasattr(self, "_vis_fresh"):
                # Flag the reset so the vision ring buffer is refilled with the new first
                # frame instead of serving the previous dive's imagery.
                self._vis_fresh[ids] = True
        dr_on = bool(getattr(self.cfg, "dr_enable", False))
        _n = int(ids.numel())
        if dr_on and float(self.cfg.dr_spawn_xy_range) > 0.0:
            _off = (torch.rand((_n, 2), device=self.device) * 2.0 - 1.0) * float(self.cfg.dr_spawn_xy_range)
        else:
            _off = torch.zeros((_n, 2), device=self.device)
        if dr_on and float(self.cfg.dr_spawn_depth_range) > 0.0:
            _dz = (torch.rand(_n, device=self.device) * 2.0 - 1.0) * float(self.cfg.dr_spawn_depth_range)
        else:
            _dz = torch.zeros(_n, device=self.device)
        # robot spawn at env origin (keep multi-env placement correct)
        origins = self.scene.env_origins[ids]
        self._default_root_state[ids, 0] = origins[:, 0] + _off[:, 0]
        self._default_root_state[ids, 1] = origins[:, 1] + _off[:, 1]
        self._default_root_state[ids, 2] = float(self.cfg.starting_depth) + _dz
        # Remember where this episode's "home" is — the anchor is captured from here and
        # disp_end_m is measured against it.
        self._spawn_xy[ids, 0] = self._default_root_state[ids, 0]
        self._spawn_xy[ids, 1] = self._default_root_state[ids, 1]
        self._dr_spawn_depth[ids] = _dz
        # Spawn at the WORK attitude (level when level_attitude; free-floating trim
        # otherwise). Level is not a buoyancy equilibrium — the attitude PID carries
        # the leveling moment from step 0 (integrator pre-charged below).
        zeros_ang = torch.zeros(ids.numel(), device=self.device)
        spawn_pitch = torch.full((ids.numel(),), float(self._att_pitch_target), device=self.device)
        spawn_roll = torch.full((ids.numel(),), float(self._att_roll_target), device=self.device)
        if dr_on and float(self.cfg.dr_spawn_yaw_range_deg) > 0.0:
            spawn_yaw = (
                torch.rand(ids.numel(), device=self.device) * 2.0 - 1.0
            ) * float(np.radians(float(self.cfg.dr_spawn_yaw_range_deg)))
        else:
            spawn_yaw = zeros_ang
        spawn_quat = quat_from_euler_xyz(spawn_roll, spawn_pitch, spawn_yaw)
        self._default_root_state[ids, 3:7] = spawn_quat
        self._robot.write_root_pose_to_sim(self._default_root_state[ids, :7], ids)
        self._robot.write_root_velocity_to_sim(self._default_root_state[ids, 7:], ids)

        self._flow_image[ids] = 0.0
        self._texture_mask[ids] = 0.0
        self._flow_rms[ids] = 0.0
        # First-frame anchor: invalid until the next textured frame is captured (its
        # offset/ncc read 0 until then — one step at most). Seed the capture-reference
        # heading with the spawn heading so pre-capture yaw_err is not garbage.
        self._anchor_valid[ids] = False
        self._anchor_tmpl[ids] = 0
        self._anchor_off_px[ids] = 0.0
        self._anchor_ncc[ids] = 0.0
        self._anchor_disp_m[ids] = 0.0
        self._anchor_yaw[ids] = euler_xyz_from_quat(self._robot.data.root_quat_w[ids])[2]
        # Visual odometry: fresh episode → zero displacement, plane at the spawn
        # distance, de-rotation seeded at the spawn heading.
        self._visodom[ids] = 0.0
        self._visodom_x[ids] = 0.0
        self._visodom_z[ids] = float(self.cfg.plane_distance) * self._visodom_range_err[ids]
        self._visodom_yaw_prev[ids] = euler_xyz_from_quat(self._robot.data.root_quat_w[ids])[2]
        self._texture_count[ids] = 0.0
        self._prev_rgb_ok[ids] = False
        self._texture_lost_streak[ids] = 0
        self._actions[ids] = 0.0
        self._prev_actions[ids] = 0.0
        self._cmd_vel[ids] = 0.0
        self._prev_cmd_vel[ids] = 0.0
        self._pwm[ids] = 0.0
        if hasattr(self, "_cached_rew"):
            self._cached_rew[ids] = 0.0
        if hasattr(self, "_ocean_current_vel"):
            self._ocean_current_vel[ids] = 0.0
        if hasattr(self, "pid"):
            self.pid.reset(ids, depth_target=float(self.cfg.starting_depth))
            # pid.reset just cleared both integrators. Put the solved level-hold
            # wrench back so frame 0 is already the handover state.
            if getattr(self, "_level_pitch_moment", 0.0) != 0.0 or getattr(self, "_level_heave_wrench", 0.0) != 0.0:
                self._apply_level_hold_precharge(ids)
        if hasattr(self, "thruster_dyn") and getattr(self.cfg, "thruster_first_order_lag", False):
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[ids] = True
            self.thruster_dyn.reset(reset_mask)
        if getattr(self, "_robot_usd", None) is not None:
            self._sync_robot_usd_pose()

    def _apply_action_delay(self, actions):
        """Return the command issued ``delay`` steps ago (per env, held for the episode).

        Slot -1 is the newest command, so a delay of d reads slot max-d. Resetting the
        history to zeros on episode reset means each dive starts from a standstill command
        rather than inheriting the previous episode's thrust.
        """
        max_d = self._action_hist.shape[1] - 1
        if max_d <= 0:
            return actions
        self._action_hist[:, :-1] = self._action_hist[:, 1:].clone()
        self._action_hist[:, -1] = actions.to(self._action_hist.dtype)
        idx = (max_d - self._act_delay_steps).view(-1, 1, 1).expand(-1, 1, 3)
        return self._action_hist.gather(1, idx).squeeze(1).to(actions.dtype)

    def _pre_physics_step(self, actions):
        """v7 action: 3-D杆量 [surge, sway, yaw] ∈ [-1, 1].

        PID (BatchPIDController.gsrov8) reads cmd_vel[:, 0] as surge target,
        cmd_vel[:, 1] as sway target and cmd_vel[:, 2] as yaw target. Depth is
        held by the retuned PI depth branch at starting_depth (first-frame depth).
        """
        self._prev_actions[:] = self._actions
        actions = self._apply_action_delay(actions)
        self._actions[:] = torch.clip(actions, -1.0, 1.0)
        # Exactly one vision push per control step; see _apply_vision_delay.
        self._vis_push_needed = True
        # Normalize the raw joystick-like action to a bounded target velocity. The scale
        # maps unit policy actions to cmd_vel targets (m/s per horizontal axis, rad/s yaw)
        # — see cfg.cmd_vel_scale for why it must cover the ocean-current peak.
        scale = float(getattr(self.cfg, "cmd_vel_scale", 0.5))
        self._cmd_vel[:, 0] = self._actions[:, 0] * scale
        self._cmd_vel[:, 1] = self._actions[:, 1] * scale
        self._cmd_vel[:, 2] = self._actions[:, 2] * scale
        limit = float(getattr(self.cfg, "cmd_rate_limit", 0.0))
        if limit > 0.0:
            lo = self._prev_cmd_vel - limit
            hi = self._prev_cmd_vel + limit
            self._cmd_vel[:] = torch.min(torch.max(self._cmd_vel, lo), hi)
        self._prev_cmd_vel[:] = self._cmd_vel

    def close(self):
        sonar = getattr(self, "oceansim_sonar", None)
        if sonar is not None:
            sonar.close()
        super().close()
