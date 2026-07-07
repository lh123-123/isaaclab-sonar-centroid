# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse, sample_uniform, wrap_to_pi

from .lh_2_env_cfg import Lh2EnvCfg
from .oceansim_sonar_sensor import OceanSimSonarSensor


class Lh2Env(DirectRLEnv):
    cfg: Lh2EnvCfg

    #: 旋转跟随：每环境轨道水平半径（m），夹紧在 cfg 圆环带 ``[min, max]`` 内。
    #: 方案 B 基础尾部：本体绕 Z 角速度 ωz、占位标量、本步动作 (1)。
    _POLICY_BASE_TAIL_DIM: int = 3

    def __init__(self, cfg: Lh2EnvCfg, render_mode: str | None = None, **kwargs):
        # 必须在super().__init__()之前初始化这些属性
        self.sonar_image = None
        self.sonar_beam_data = None
        super().__init__(cfg, render_mode, **kwargs)

        self.robot_pos = self.robot.data.root_pos_w
        self.robot_quat = self.robot.data.root_quat_w
        self.robot_vel = self.robot.data.root_lin_vel_w
        self.robot_ang_vel = self.robot.data.root_ang_vel_w
        
        self.target_pos = self.target.data.root_pos_w
        
        self.actions = torch.zeros(self.num_envs, 1, device=self.device)
        self.previous_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self.target_velocities = torch.zeros(self.num_envs, 3, device=self.device)
        # 目标仅沿世界 Y 轴平移（左右）；+1 / -1 表示方向，由 _update_target_motion 偶尔翻转
        self.target_y_sign = torch.ones(self.num_envs, device=self.device)
        # 变速：标量速率向 target_speed_goal 做加速度限制积分
        v0, v1 = self.cfg.target_speed_range_start
        self.target_current_speed = sample_uniform(v0, v1, (self.num_envs,), self.device)
        self.target_speed_goal = self.target_current_speed.clone()
        
        self.sonar_range = torch.zeros(self.num_envs, device=self.device)
        self.sonar_angle = torch.zeros(self.num_envs, device=self.device)
        self.sonar_intensity = torch.zeros(self.num_envs, device=self.device)
        self.ekf_position = torch.zeros(self.num_envs, 2, device=self.device)
        self.ekf_velocity = torch.zeros(self.num_envs, 2, device=self.device)
        self.prev_lateral_error = torch.zeros(self.num_envs, device=self.device)
        self.prev_bearing_reward = torch.zeros(self.num_envs, device=self.device)
        self.out_of_view_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Geometric sonar (_simulate_sonar) cache: invalidated each _reset_idx; same-step reuse across dones/rewards/obs.
        self._geom_sonar_cache_valid_for: int = -1
        self._geom_sonar_cache_invalidated: bool = True
        self.reversal_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reversal_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reversal_start_error = torch.zeros(self.num_envs, device=self.device)
        self.reversal_total = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reversal_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_target_y_sign = self.target_y_sign.clone()
        self.target_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_orbit_phase = torch.zeros(self.num_envs, device=self.device)
        r_lo = float(getattr(self.cfg, "target_orbit_radius_min", 1.0))
        r_hi = float(getattr(self.cfg, "target_orbit_radius_max", 2.5))
        r_nom = float(getattr(self.cfg, "target_orbit_radius", 1.5))
        r0 = min(max(r_nom, r_lo), r_hi)
        self._target_orbit_radius_env = torch.full(
            (self.num_envs,), r0, device=self.device, dtype=torch.float32
        )
        w_lo = float(getattr(self.cfg, "target_orbit_angular_vel_min", -0.30))
        w_hi = float(getattr(self.cfg, "target_orbit_angular_vel_max", 0.30))
        self.target_orbit_w_inst = sample_uniform(w_lo, w_hi, (self.num_envs,), self.device)
        self.prev_body_bearing_to_target = torch.zeros(self.num_envs, device=self.device)
        self._last_body_bearing_rad = torch.zeros(self.num_envs, device=self.device)
        self._last_body_bearing_dtheta_rad = torch.zeros(self.num_envs, device=self.device)
        self.wrong_direction_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Relative target state estimated from sonar image moments (policy input in relative-position mode).
        self.prev_rel_est = torch.zeros(self.num_envs, 2, device=self.device)
        self.prev_rel_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rel_est_state = torch.zeros(self.num_envs, 2, device=self.device)

        ocfg = self.cfg.oceansim_sonar_cfg
        self.sonar_image = torch.zeros(self.num_envs, ocfg.height, ocfg.width, device=self.device)
        self.sonar_beam_data = torch.zeros(self.num_envs, ocfg.beam_count, 3, device=self.device)
        self.beam_history = torch.zeros(
            self.num_envs, self.cfg.beam_history_len, ocfg.beam_count, 3, device=self.device
        )
        self._beam_angle_unit_logged = False
        self._obs_audit_logged = False
        self._beam_softmax_gate_ratio = torch.tensor(0.0, device=self.device)
        self._beam_intensity_raw_mean = torch.tensor(0.0, device=self.device)
        self._beam_intensity_raw_max = torch.tensor(0.0, device=self.device)
        self._beam_intensity_norm_mean = torch.tensor(0.0, device=self.device)
        self._beam_intensity_norm_max = torch.tensor(0.0, device=self.device)
        self._beam_gate_peak_pass_ratio = torch.tensor(0.0, device=self.device)
        self._beam_gate_mean_pass_ratio = torch.tensor(0.0, device=self.device)
        self._beam_peak_angle_deg = torch.tensor(0.0, device=self.device)
        self._beam_peak_range_m = torch.tensor(0.0, device=self.device)
        self._beam_peak_intensity = torch.tensor(0.0, device=self.device)
        # Reward/done visibility in {0,1}, filled in _ensure_geom_sonar_for_step (geom ± beam).
        self._rl_visible_f = torch.zeros(self.num_envs, device=self.device)
        self._diag_visible_mean_geom = torch.tensor(0.0, device=self.device)
        self._diag_visible_mean_beam = torch.tensor(0.0, device=self.device)
        self._last_beam_raw_for_debug: torch.Tensor | None = None
        ox, oy, oz = self.cfg.sonar_mount_offset_local
        self._sonar_mount_offset_b = torch.tensor(
            (ox, oy, oz), device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.num_envs, 1)

    def _standardize_beam_data(self, beam_data: torch.Tensor) -> torch.Tensor:
        """Validate beam tensor shape and normalize angle channel to radians."""
        expected_beams = int(self.cfg.oceansim_sonar_cfg.beam_count)
        if beam_data.dim() != 3 or beam_data.shape[0] != self.num_envs or beam_data.shape[2] != 3:
            raise ValueError(
                "Lh2Env beam_data shape mismatch: "
                f"expected=[{self.num_envs}, {expected_beams}, 3], got={tuple(beam_data.shape)}"
            )
        if beam_data.shape[1] != expected_beams:
            raise ValueError(
                "Lh2Env beam_count mismatch: "
                f"cfg.beam_count={expected_beams}, runtime={beam_data.shape[1]}"
            )

        beam_data_std = beam_data.clone()
        # Sensor extractor stores beam angle in degrees; convert once for downstream consistency.
        beam_data_std[:, :, 0] = torch.deg2rad(beam_data_std[:, :, 0])
        if not self._beam_angle_unit_logged:
            self._beam_angle_unit_logged = True
        return beam_data_std

    def _beam_int_base_from_intensity(self, intensity: torch.Tensor) -> torch.Tensor:
        """Log-compressed intensity in [0, 1] (same as beam policy obs, without softmax)."""
        log_gain = max(float(getattr(self.cfg, "beam_obs_intensity_log_gain", 4.0)), 1.0e-6)
        int_base = torch.log1p(torch.clamp(intensity, min=0.0) * log_gain) / torch.log1p(
            torch.tensor(log_gain, device=self.device, dtype=intensity.dtype)
        )
        return torch.clamp(int_base, 0.0, 1.0)

    def _detected_float_from_beam_gate(self) -> torch.Tensor:
        """Peak/mean gate on beam int_base → float {0,1} per env (matches gated softmax path intent)."""
        beam = self.sonar_beam_data
        n = self.num_envs
        if beam is None or beam.shape[0] != n:
            return torch.zeros(n, device=self.device, dtype=torch.float32)
        intensity = beam[:, :, 2]
        int_base = self._beam_int_base_from_intensity(intensity)
        peak_thr = float(getattr(self.cfg, "beam_obs_gate_peak_threshold", 0.20))
        mean_thr = float(getattr(self.cfg, "beam_obs_gate_mean_threshold", 0.03))
        peak_pass = torch.max(int_base, dim=1).values >= peak_thr
        mean_pass = torch.mean(int_base, dim=1) >= mean_thr
        gate = peak_pass & mean_pass
        return gate.to(dtype=torch.float32)

    def _update_oceansim_sonar_frontend(self) -> None:
        """One OceanSim update per env step; fills sonar_image / sonar_beam_data for obs + reward alignment."""
        if not hasattr(self, "oceansim_sonar"):
            return
        if not bool(getattr(self.oceansim_sonar, "_initialized", False)):
            return
        dt = float(self.cfg.sim.dt * self.cfg.decimation)
        self.oceansim_sonar.update(dt=dt)
        sonar_image = self.oceansim_sonar.get_sonar_image()
        beam_data = self.oceansim_sonar.get_beam_data()
        if sonar_image is not None:
            self.sonar_image = sonar_image
        if beam_data is not None:
            self._last_beam_raw_for_debug = beam_data.clone()
            self.sonar_beam_data = self._standardize_beam_data(beam_data)

    def _cache_rl_visibility_float(self) -> None:
        """Set ``_rl_visible_f`` from geometry and/or OceanSim beam (cfg)."""
        geom_ok = self.sonar_range < (float(self.cfg.sonar_cfg.max_range) * 1.5)
        geom_f = geom_ok.to(dtype=torch.float32)
        self._diag_visible_mean_geom = geom_f.mean()
        use_beam = bool(getattr(self.cfg, "reward_detection_use_oceansim_beam", True))
        oinit = bool(getattr(self.oceansim_sonar, "_initialized", False)) if hasattr(self, "oceansim_sonar") else False
        if use_beam and oinit and self.sonar_beam_data is not None:
            beam_f = self._detected_float_from_beam_gate()
            self._diag_visible_mean_beam = beam_f.mean()
            if bool(getattr(self.cfg, "reward_beam_detection_or_geom", True)):
                how = str(getattr(self.cfg, "reward_visibility_combine", "min")).lower()
                if how == "max":
                    self._rl_visible_f = torch.maximum(beam_f, geom_f)
                else:
                    self._rl_visible_f = torch.minimum(beam_f, geom_f)
            else:
                self._rl_visible_f = beam_f
        else:
            self._diag_visible_mean_beam = torch.tensor(0.0, device=self.device)
            self._rl_visible_f = geom_f

    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot_cfg)
        self.target = RigidObject(self.cfg.target_cfg)
        
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        self.scene.clone_environments(copy_from_source=False)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        self.scene.rigid_objects["robot"] = self.robot
        self.scene.rigid_objects["target"] = self.target

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.oceansim_sonar = OceanSimSonarSensor(self.cfg.oceansim_sonar_cfg, num_envs=self.num_envs, device=self.device)
        
        for env_idx in range(self.num_envs):
            robot_prim_path = f"/World/envs/env_{env_idx}/Robot"
            sonar_mount_path = f"{robot_prim_path}/SonarMount"
            
            try:
                from pxr import Usd
                from pxr import UsdGeom
                import omni.usd as usd_utils
                
                stage = usd_utils.get_context().get_stage()

                xform = UsdGeom.Xform.Define(stage, sonar_mount_path)
                ox, oy, oz = self.cfg.sonar_mount_offset_local
                # Do not blindly AddTranslateOp(): cloned env prims may already contain xformOp:translate.
                # Reuse existing translate op when present to avoid AddXformOp duplicate-op exceptions.
                xformable = UsdGeom.Xformable(xform.GetPrim())
                translate_op = None
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                if translate_op is None:
                    translate_op = xformable.AddTranslateOp()
                translate_op.Set((float(ox), float(oy), float(oz)))
                # Defensive cleanup: remove any stale sonar camera prims under this mount
                # (can appear after repeated UI/non-headless runs).
                sonar_mount_prim = stage.GetPrimAtPath(sonar_mount_path)
                if sonar_mount_prim:
                    for child in sonar_mount_prim.GetChildren():
                        child_name = str(child.GetName())
                        if child_name.startswith("SonarSensor"):
                            stage.RemovePrim(str(child.GetPath()))
                
            except Exception as e:
                print(f"Warning: Could not create sonar mount, using robot prim: {e}")
                sonar_mount_path = robot_prim_path
            
            self.oceansim_sonar.initialize(env_idx, sonar_mount_path)
        
        import time

        for warmup_idx in range(4):
            self.sim.step()
            if warmup_idx == 0:
                time.sleep(0.02)

        # 动态计算观测空间
        self._update_observation_space_from_sonar()

    def _policy_orbit_w_inst_tail_enabled(self) -> bool:
        if not bool(getattr(self.cfg, "policy_append_orbit_w_inst", False)):
            return False
        return str(getattr(self.cfg, "target_motion_mode", "orbit_robot")).lower() == "orbit_robot"

    def _policy_tail_dim(self) -> int:
        n = self._POLICY_BASE_TAIL_DIM
        mode = str(getattr(self.cfg, "policy_obs_mode", "image")).lower()
        if mode == "relative_position":
            return n
        if bool(getattr(self.cfg, "policy_lateral_hint_enabled", False)):
            n += 1
        if self._policy_orbit_w_inst_tail_enabled():
            n += 1
        return n

    def _append_policy_tail_extensions(self, tail_core: torch.Tensor) -> torch.Tensor:
        """Optional tail after [ωz, pad, last_action]: lateral curriculum hint, then orbit ``w_inst`` scalar."""
        mode = str(getattr(self.cfg, "policy_obs_mode", "image")).lower()
        if mode == "relative_position":
            return tail_core
        out = tail_core
        if bool(getattr(self.cfg, "policy_lateral_hint_enabled", False)):
            w = self._lateral_hint_curriculum_weight()
            dy = self._estimate_world_dy_from_sonar_geometry()
            sigma = max(float(getattr(self.cfg, "policy_lateral_hint_sigma", 1.5)), 1e-6)
            hint = torch.tanh((dy / sigma).unsqueeze(1)) * float(w)
            out = torch.cat([out, hint], dim=-1)
        if self._policy_orbit_w_inst_tail_enabled():
            scale = max(float(getattr(self.cfg, "policy_orbit_w_inst_norm_scale", 0.35)), 1e-6)
            ow = torch.tanh((self.target_orbit_w_inst / scale).unsqueeze(1))
            out = torch.cat([out, ow], dim=-1)
        return out

    def _lateral_hint_curriculum_weight(self) -> float:
        if not bool(getattr(self.cfg, "policy_lateral_hint_enabled", False)):
            return 0.0
        stage1_steps = int(getattr(self.cfg, "policy_lateral_hint_curriculum_env_steps", 0))
        min_w = float(getattr(self.cfg, "policy_lateral_hint_min_weight", 0.0))
        min_w = min(max(min_w, 0.0), 1.0)
        stage2_steps = int(getattr(self.cfg, "policy_lateral_hint_tail_env_steps", 0))

        if stage1_steps <= 0:
            return 1.0

        t = float(self.common_step_counter)
        if t <= float(stage1_steps):
            # Fast linear decay: 1.0 -> min_w
            p1 = t / float(stage1_steps)
            return 1.0 - (1.0 - min_w) * p1

        if stage2_steps <= 0:
            return 0.0

        # Tail linear decay: min_w -> 0.0
        t2 = t - float(stage1_steps)
        p2 = min(max(t2 / float(stage2_steps), 0.0), 1.0)
        return min_w * (1.0 - p2)

    def _policy_obs_dim_from_cfg(self) -> int:
        """Policy observation size expected from cfg."""
        mode = str(getattr(self.cfg, "policy_obs_mode", "image")).lower()
        if mode == "relative_position":
            return 5
        if mode == "beam":
            return int(self.cfg.oceansim_sonar_cfg.beam_count) * 3 + self._policy_tail_dim()
        ocfg = self.cfg.oceansim_sonar_cfg
        return ocfg.height * ocfg.width + self._policy_tail_dim()

    def _update_observation_space_from_sonar(self):
        """Log expected policy observation dim from cfg.

        Note:
            ``single_observation_space`` is initialized later by the base env setup pipeline,
            so this hook must not access gym spaces directly.
        """
        dim = self._policy_obs_dim_from_cfg()
        _ = dim

    def _build_beam_policy_obs(self) -> torch.Tensor:
        """Build policy observation from beam triplets with gated softmax enhancement."""
        beam = self.sonar_beam_data
        if beam is None:
            return torch.zeros(self.num_envs, self._policy_obs_dim_from_cfg(), device=self.device)

        angle = beam[:, :, 0]
        beam_range = beam[:, :, 1]
        intensity = beam[:, :, 2]
        peak_idx = torch.argmax(intensity, dim=1)
        env_idx = torch.arange(self.num_envs, device=self.device)
        peak_angle_rad = angle[env_idx, peak_idx]
        self._beam_peak_angle_deg = torch.mean(torch.rad2deg(peak_angle_rad))
        self._beam_peak_range_m = torch.mean(beam_range[env_idx, peak_idx])
        self._beam_peak_intensity = torch.mean(intensity[env_idx, peak_idx])
        self._beam_intensity_raw_mean = intensity.mean()
        self._beam_intensity_raw_max = torch.max(intensity)

        fov_half_rad = max(float(self.cfg.sonar_cfg.hori_fov) * 0.5 * (torch.pi / 180.0), 1.0e-6)
        angle_norm = torch.clamp(angle / fov_half_rad, -1.0, 1.0)

        range_min = float(self.cfg.oceansim_sonar_cfg.min_range)
        range_max = float(self.cfg.oceansim_sonar_cfg.max_range)
        range_span = max(range_max - range_min, 1.0e-6)
        range_norm = torch.clamp((beam_range - range_min) / range_span, 0.0, 1.0)

        int_base = self._beam_int_base_from_intensity(intensity)
        self._beam_intensity_norm_mean = int_base.mean()
        self._beam_intensity_norm_max = torch.max(int_base)

        if bool(getattr(self.cfg, "beam_obs_softmax_enabled", True)):
            temp = max(float(getattr(self.cfg, "beam_obs_softmax_temperature", 0.25)), 1.0e-4)
            int_soft = torch.softmax(int_base / temp, dim=1)
            mix = min(max(float(getattr(self.cfg, "beam_obs_softmax_mix", 0.60)), 0.0), 1.0)
            peak_thr = float(getattr(self.cfg, "beam_obs_gate_peak_threshold", 0.20))
            mean_thr = float(getattr(self.cfg, "beam_obs_gate_mean_threshold", 0.03))
            peak_pass = torch.max(int_base, dim=1).values >= peak_thr
            mean_pass = torch.mean(int_base, dim=1) >= mean_thr
            self._beam_gate_peak_pass_ratio = peak_pass.float().mean()
            self._beam_gate_mean_pass_ratio = mean_pass.float().mean()
            gate = peak_pass & mean_pass
            if bool(getattr(self.cfg, "beam_obs_gate_use_geom_detect", True)):
                geom_detected = self.sonar_range < (float(self.cfg.sonar_cfg.max_range) * 1.5)
                gate = gate | geom_detected
            gate = gate.unsqueeze(1)
            self._beam_softmax_gate_ratio = gate.float().mean()
            int_fused = torch.where(gate, (1.0 - mix) * int_base + mix * int_soft, int_base)
        else:
            self._beam_softmax_gate_ratio = torch.tensor(0.0, device=self.device)
            self._beam_gate_peak_pass_ratio = torch.tensor(0.0, device=self.device)
            self._beam_gate_mean_pass_ratio = torch.tensor(0.0, device=self.device)
            int_fused = int_base

        beam_feat = torch.stack([angle_norm, range_norm, int_fused], dim=-1).reshape(self.num_envs, -1)
        robot_wz = self.robot_ang_vel[:, 2].unsqueeze(1)
        tail_pad = torch.zeros(self.num_envs, 1, device=self.device)
        last_action = self.actions
        tail = self._append_policy_tail_extensions(torch.cat([robot_wz, tail_pad, last_action], dim=-1))
        return torch.cat([beam_feat, tail], dim=-1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions = (
            self.actions.clone() if hasattr(self, "actions") else torch.zeros(self.num_envs, 1, device=self.device)
        )
        self.actions = actions.clone()
        # Update target motion once per environment step (not every physics sub-step).
        self._update_target_motion()

    def _apply_action(self) -> None:
        # 单自由度：绕世界 Z 轴角速度；线速度置零（原地转向）。
        wz_cmd = self._effective_robot_yaw_rate_command()
        lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        ang_vel[:, 2] = wz_cmd
        self.robot.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1))
        
    def _refresh_kinematic_state(self) -> None:
        """Refresh cached state tensors before rewards/dones/observations."""
        self.robot_pos = self.robot.data.root_pos_w
        self.robot_quat = self.robot.data.root_quat_w
        self.robot_vel = self.robot.data.root_lin_vel_w
        self.robot_ang_vel = self.robot.data.root_ang_vel_w
        self.target_pos = self.target.data.root_pos_w

    def _direction_command_deadband_wz(self) -> float:
        """Min |ωz| command for wrong-direction logic (rad/s)."""
        return max(float(self.cfg.direction_action_eps) * float(self.cfg.action_scale_angular), 0.05)

    def _orbit_radius_bounds(self) -> tuple[float, float]:
        r_lo = float(getattr(self.cfg, "target_orbit_radius_min", 1.0))
        r_hi = float(getattr(self.cfg, "target_orbit_radius_max", 2.5))
        if r_hi < r_lo:
            r_lo, r_hi = r_hi, r_lo
        return r_lo, r_hi

    def _effective_orbit_radius_xy(self) -> torch.Tensor:
        """每环境当前轨道半径，夹紧到圆环带。"""
        r_lo, r_hi = self._orbit_radius_bounds()
        return torch.clamp(self._target_orbit_radius_env, r_lo, r_hi)

    def _orbit_phase_from_robot_forward_xy(self, robot_quat: torch.Tensor) -> torch.Tensor:
        """机体 +X（声呐/相机前向）投影到世界水平面，得到轨道角 ``phase`` 使 ``R*(cos,sin)`` 落在该方向上。"""
        n = robot_quat.shape[0]
        ex = torch.zeros(n, 3, device=self.device, dtype=robot_quat.dtype)
        ex[:, 0] = 1.0
        fwd = quat_apply(robot_quat, ex)
        fwd = fwd.clone()
        fwd[:, 2] = 0.0
        nrm = torch.norm(fwd[:, :2], dim=-1, keepdim=True)
        small = (nrm.squeeze(-1) < 1e-5).unsqueeze(-1)
        u = torch.where(small, torch.tensor([[1.0, 0.0]], device=self.device, dtype=fwd.dtype).expand(n, 2), fwd[:, :2] / nrm.clamp(min=1e-6))
        return torch.atan2(u[:, 1], u[:, 0])

    def _ball_world_heading_angle(self) -> torch.Tensor:
        """世界系水平面内，机器人指向目标的方位角 atan2(Δy, Δx)（相对世界 +X，rad）。"""
        d = self.target_pos[:, :2] - self.robot_pos[:, :2]
        return torch.atan2(d[:, 1], d[:, 0])

    def _body_plane_bearing_to_target(self) -> torch.Tensor:
        """水平面内目标相对机体系的方位角：atan2(y_b, x_b)，x_b 为机头前向。"""
        rel_w = self.target.data.root_pos_w - self.robot.data.root_pos_w
        rel_w = rel_w.clone()
        rel_w[:, 2] = 0.0
        rel_b = quat_apply_inverse(self.robot.data.root_quat_w, rel_w)
        return wrap_to_pi(torch.atan2(rel_b[:, 1], rel_b[:, 0]))

    def _print_body_bearing_debug(self, bearing: torch.Tensor) -> None:
        """训练/回放打印 env0：仅机体前向与目标水平方位差（度），不打印各自绝对角。"""
        if not bool(getattr(self.cfg, "print_ball_world_angle", True)):
            return
        interval = max(1, int(getattr(self.cfg, "ball_world_angle_print_interval", 20)))
        step = int(self.episode_length_buf[0].item()) if self.episode_length_buf.numel() > 0 else 0
        if step % interval != 0:
            return
        diff_deg = float(torch.rad2deg(bearing[0]).item())
        mode = str(getattr(self.cfg, "run_mode", "train"))
        print(
            f"[Lh2 track][{mode}] step={step} angle_diff_body_deg={diff_deg:+.2f}",
            flush=True,
        )

    def _effective_robot_yaw_rate_command(self) -> torch.Tensor:
        """绕世界 Z 的角速度指令（rad/s），经饱和与门控。"""
        w = self.actions[:, 0] * float(self.cfg.action_scale_angular)
        w = torch.clamp(w, -float(self.cfg.max_angular_velocity), float(self.cfg.max_angular_velocity))

        freeze_steps = max(int(getattr(self.cfg, "startup_freeze_steps", 0)), 0)
        if freeze_steps > 0:
            freeze_mask = self.episode_length_buf < freeze_steps
            w = torch.where(freeze_mask, torch.zeros_like(w), w)

        if bool(getattr(self.cfg, "dy_sign_guard", False)):
            bearing = self._body_plane_bearing_to_target()
            deadband = float(getattr(self.cfg, "dy_sign_guard_deadband", 0.07))
            valid = torch.abs(bearing) > deadband
            wrong_sign = (bearing * w) < 0.0
            w = torch.where(valid & wrong_sign, torch.zeros_like(w), w)

        return w

    def _scaled_robot_yaw_rate_command(self) -> torch.Tensor:
        """Alias for rewards/dones (matches _apply_action semantics)."""
        return self._effective_robot_yaw_rate_command()

    def _ensure_geom_sonar_for_step(self) -> None:
        """Run _simulate_sonar at most once per env step; always after resets (cache invalidated in _reset_idx)."""
        if (
            self._geom_sonar_cache_valid_for == self.common_step_counter
            and not self._geom_sonar_cache_invalidated
        ):
            return
        self._refresh_kinematic_state()
        _ = self._simulate_sonar()
        self._update_oceansim_sonar_frontend()
        self._cache_rl_visibility_float()
        self._geom_sonar_cache_valid_for = self.common_step_counter
        self._geom_sonar_cache_invalidated = False

    def _sonar_relative_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """目标相对声呐安装点：世界系向量、机体系（与声呐轴向对齐机体）向量、水平距离 ‖Δxy‖。"""
        rq = self.robot.data.root_quat_w
        sonar_w = self.robot_pos + quat_apply(rq, self._sonar_mount_offset_b)
        rel_w = self.target_pos - sonar_w
        rel_local = quat_apply_inverse(rq, rel_w)
        dist_xy = torch.norm(rel_w[:, :2], dim=-1)
        return rel_w, rel_local, dist_xy

    def _update_target_motion(self):
        """更新目标运动：默认 ``orbit_robot``（固定半径、随机周向角速度）；可选 waypoint / random 侧向模式。"""
        freeze_steps = max(int(getattr(self.cfg, "startup_freeze_steps", 0)), 0)
        freeze_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if freeze_steps > 0:
            freeze_mask = self.episode_length_buf < freeze_steps
            if torch.any(freeze_mask):
                self.target_velocities[freeze_mask] = 0.0
                if torch.all(freeze_mask):
                    self.target.write_root_velocity_to_sim(
                        torch.zeros(self.num_envs, 6, device=self.device)
                    )
                    return
                lin = self.target.data.root_lin_vel_w
                ang = self.target.data.root_ang_vel_w
                vel_w = torch.cat([lin, ang], dim=-1).clone()
                vel_w[freeze_mask] = 0.0
                self.target.write_root_velocity_to_sim(vel_w)

        motion_mode = str(getattr(self.cfg, "target_motion_mode", "orbit_robot")).lower()
        if motion_mode == "orbit_robot":
            dt = self.cfg.sim.dt * self.cfg.decimation
            R = self._effective_orbit_radius_xy()
            w_lo = float(getattr(self.cfg, "target_orbit_angular_vel_min", -0.30))
            w_hi = float(getattr(self.cfg, "target_orbit_angular_vel_max", 0.30))
            if w_hi < w_lo:
                w_lo, w_hi = w_hi, w_lo
            noise_std = max(float(getattr(self.cfg, "target_orbit_angular_noise_std", 0.08)), 0.0)
            dw = torch.randn(self.num_envs, device=self.device, dtype=R.dtype) * noise_std
            w_new = torch.clamp(self.target_orbit_w_inst + dw, w_lo, w_hi)
            self.target_orbit_w_inst = torch.where(freeze_mask, self.target_orbit_w_inst, w_new)
            phase_inc = self.target_orbit_w_inst * dt
            self.target_orbit_phase = torch.where(
                freeze_mask,
                self.target_orbit_phase,
                self.target_orbit_phase + phase_inc,
            )
            robot_p = self.robot.data.root_pos_w
            c = torch.cos(self.target_orbit_phase)
            s = torch.sin(self.target_orbit_phase)
            new_x = robot_p[:, 0] + R * c
            new_y = robot_p[:, 1] + R * s
            new_z = robot_p[:, 2]
            current_pos = self.target.data.root_pos_w.clone()
            new_pos = torch.stack([new_x, new_y, new_z], dim=-1)
            new_pos = torch.where(freeze_mask.unsqueeze(1), current_pos, new_pos)

            w_use = self.target_orbit_w_inst
            self.target_velocities[:, 0] = torch.where(
                freeze_mask,
                torch.zeros_like(c),
                -R * w_use * s,
            )
            self.target_velocities[:, 1] = torch.where(
                freeze_mask,
                torch.zeros_like(c),
                R * w_use * c,
            )
            self.target_velocities[:, 2] = 0.0

            target_quat = self.target.data.root_quat_w.clone()
            target_pose = torch.cat([new_pos, target_quat], dim=-1)
            self.target.write_root_pose_to_sim(target_pose)
            target_vel = torch.cat([self.target_velocities, torch.zeros(self.num_envs, 3, device=self.device)], dim=-1)
            self.target.write_root_velocity_to_sim(target_vel)
            return

        if motion_mode == "waypoint_cycle":
            env_origins = self.scene.env_origins
            dt = self.cfg.sim.dt * self.cfg.decimation
            speed = max(float(getattr(self.cfg, "target_waypoint_speed", 0.12)), 1e-6)
            reach_thr = max(float(getattr(self.cfg, "target_waypoint_reach_threshold", 0.01)), 1e-6)
            waypoints = tuple(getattr(self.cfg, "target_waypoint_cycle_y", (0.3, -0.3, 0.5, -0.5)))
            if len(waypoints) == 0:
                waypoints = (0.0,)

            wp_tensor = torch.tensor(waypoints, device=self.device, dtype=self.target_pos.dtype)
            cur_idx = self.target_waypoint_idx.clamp(min=0, max=len(waypoints) - 1)
            cur_wp = wp_tensor[cur_idx]

            current_pos = self.target.data.root_pos_w.clone()
            local_y = current_pos[:, 1] - env_origins[:, 1]
            err = cur_wp - local_y
            reached = torch.abs(err) <= reach_thr
            if torch.any(reached):
                next_idx = (cur_idx[reached] + 1) % len(waypoints)
                self.target_waypoint_idx[reached] = next_idx
                cur_idx = self.target_waypoint_idx.clamp(min=0, max=len(waypoints) - 1)
                cur_wp = wp_tensor[cur_idx]
                err = cur_wp - local_y

            vy_cmd = torch.sign(err) * speed
            max_step = speed * dt
            close_mask = torch.abs(err) < max_step
            vy_cmd = torch.where(close_mask, err / max(dt, 1e-6), vy_cmd)
            if freeze_steps > 0:
                vy_cmd = torch.where(freeze_mask, torch.zeros_like(vy_cmd), vy_cmd)

            new_local_y = local_y + vy_cmd * dt
            new_local_y = torch.clamp(new_local_y, -float(self.cfg.target_lateral_half_range), float(self.cfg.target_lateral_half_range))
            new_pos = current_pos.clone()
            new_pos[:, 0] = env_origins[:, 0] + self.cfg.target_fixed_x
            new_pos[:, 1] = new_local_y + env_origins[:, 1]
            new_pos[:, 2] = 10.0

            self.target_velocities[:, 0] = 0.0
            self.target_velocities[:, 1] = vy_cmd
            self.target_velocities[:, 2] = 0.0
            target_quat = self.target.data.root_quat_w.clone()
            target_pose = torch.cat([new_pos, target_quat], dim=-1)
            self.target.write_root_pose_to_sim(target_pose)
            target_vel = torch.cat([self.target_velocities, torch.zeros(self.num_envs, 3, device=self.device)], dim=-1)
            self.target.write_root_velocity_to_sim(target_vel)
            return

        env_origins = self.scene.env_origins
        change_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.target_direction_change_prob
        self.target_y_sign = torch.where(change_mask, -self.target_y_sign, self.target_y_sign)

        cur_steps = float(self.common_step_counter)
        cur_lim = float(max(1, int(getattr(self.cfg, "target_speed_curriculum_env_steps", 1))))
        alpha = min(max(cur_steps / cur_lim, 0.0), 1.0)
        v_lo = float(self.cfg.target_speed_range_start[0]) + alpha * (
            float(self.cfg.target_speed_range_final[0]) - float(self.cfg.target_speed_range_start[0])
        )
        v_hi = float(self.cfg.target_speed_range_start[1]) + alpha * (
            float(self.cfg.target_speed_range_final[1]) - float(self.cfg.target_speed_range_start[1])
        )
        speed_change_prob = float(self.cfg.target_speed_change_prob_start) + alpha * (
            float(self.cfg.target_speed_change_prob_final) - float(self.cfg.target_speed_change_prob_start)
        )
        speed_goal_change = torch.rand(self.num_envs, device=self.device) < speed_change_prob
        new_goals = sample_uniform(v_lo, v_hi, (self.num_envs,), self.device)
        self.target_speed_goal = torch.where(speed_goal_change, new_goals, self.target_speed_goal)

        dt = self.cfg.sim.dt * self.cfg.decimation
        accel_lim = float(self.cfg.target_speed_accel_limit_start) + alpha * (
            float(self.cfg.target_speed_accel_limit_final) - float(self.cfg.target_speed_accel_limit_start)
        )
        max_dv = accel_lim * dt
        dv = torch.clamp(self.target_speed_goal - self.target_current_speed, -max_dv, max_dv)
        self.target_current_speed = torch.clamp(self.target_current_speed + dv, v_lo, v_hi)

        self.target_velocities[:, 0] = 0.0
        self.target_velocities[:, 1] = self.target_current_speed * self.target_y_sign
        self.target_velocities[:, 2] = 0.0
        if freeze_steps > 0:
            self.target_velocities[freeze_mask] = 0.0

        current_pos = self.target.data.root_pos_w.clone()
        new_pos = current_pos + self.target_velocities * dt

        half = self.cfg.target_lateral_half_range
        local_xy = new_pos[:, :2] - env_origins[:, :2]
        new_y_local = local_xy[:, 1]
        over_high = new_y_local > half
        over_low = new_y_local < -half
        local_xy[:, 1] = torch.clamp(new_y_local, -half, half)
        self.target_y_sign = torch.where(over_high | over_low, -self.target_y_sign, self.target_y_sign)
        self.target_velocities[:, 1] = self.target_current_speed * self.target_y_sign
        new_pos[:, :2] = local_xy + env_origins[:, :2]

        new_pos[:, 0] = env_origins[:, 0] + self.cfg.target_fixed_x

        boundary = self.cfg.target_boundary_range
        local_xy = new_pos[:, :2] - env_origins[:, :2]
        out_of_bounds = torch.norm(local_xy, dim=-1) > boundary
        if out_of_bounds.any():
            direction_to_center = -local_xy[out_of_bounds]
            direction_to_center = direction_to_center / (torch.norm(direction_to_center, dim=-1, keepdim=True) + 1e-6)
            local_xy[out_of_bounds] = direction_to_center * (boundary - 0.5)
            local_xy[:, 1] = torch.clamp(local_xy[:, 1], -half, half)
            new_pos[:, :2] = local_xy + env_origins[:, :2]
            new_pos[:, 0] = env_origins[:, 0] + self.cfg.target_fixed_x

        new_pos[:, 2] = 10.0
        
        target_quat = self.target.data.root_quat_w.clone()
        target_pose = torch.cat([new_pos, target_quat], dim=-1)
        self.target.write_root_pose_to_sim(target_pose)
        
        target_vel = torch.cat([self.target_velocities, torch.zeros(self.num_envs, 3, device=self.device)], dim=-1)
        self.target.write_root_velocity_to_sim(target_vel)

        flipped = self.target_y_sign != self.prev_target_y_sign
        if torch.any(flipped):
            self.reversal_active[flipped] = True
            self.reversal_age[flipped] = 0
            # 与奖励横向一致：|target_y - robot_y|（new_pos 已为本步目标位置）
            ry = self.robot.data.root_pos_w[:, 1]
            self.reversal_start_error[flipped] = torch.abs(new_pos[flipped, 1] - ry[flipped])
            self.reversal_total = self.reversal_total + flipped.long()
        self.prev_target_y_sign = self.target_y_sign.clone()

    def _simulate_sonar(self):
        """模拟声呐探测（距离/水平角相对声呐安装点，与 SonarMount 外参一致）。"""
        relative_pos, relative_pos_local, distance = self._sonar_relative_geometry()
        self._sonar_rel_w = relative_pos
        self._sonar_rel_local = relative_pos_local
        self._sonar_dist_xy = distance

        angle = torch.atan2(relative_pos_local[:, 1], relative_pos_local[:, 0])
        
        in_range = (distance >= self.cfg.sonar_cfg.min_range) & (distance <= self.cfg.sonar_cfg.max_range)
        in_fov = torch.abs(angle) <= torch.deg2rad(torch.tensor(self.cfg.sonar_cfg.hori_fov / 2, device=self.device))
        
        detected = in_range & in_fov
        
        self.sonar_range = torch.where(detected, distance, torch.full_like(distance, self.cfg.sonar_cfg.max_range * 2))
        self.sonar_angle = torch.where(detected, angle, torch.zeros_like(angle))
        
        base_intensity = torch.exp(-distance / (self.cfg.sonar_cfg.max_range * 0.5))
        noise = torch.rand(self.num_envs, device=self.device) * 0.1
        self.sonar_intensity = torch.where(detected, base_intensity + noise, torch.zeros_like(base_intensity))
        
        self.ekf_position = relative_pos_local[:, :2].clone()
        self.ekf_velocity = torch.zeros(self.num_envs, 2, device=self.device)
        
        return detected

    def _estimate_world_dy_from_sonar_geometry(self) -> torch.Tensor:
        """Estimate world-frame ``target_y - robot_y`` from sonar-relative geometry and robot pose.

        This avoids directly reading target world coordinates for hint construction.
        """
        rel_local = self._sonar_rel_local
        # target_w - sonar_w
        rel_w_est = quat_apply(self.robot_quat, rel_local)
        # sonar_w - robot_w (sensor extrinsic translated into world frame)
        sonar_offset_w = quat_apply(self.robot_quat, self._sonar_mount_offset_b)
        # target_w - robot_w along world Y
        return rel_w_est[:, 1] + sonar_offset_w[:, 1]

    def _estimate_relative_from_sonar_image(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate target relative position (x,y) and confidence from sonar image moments.

        Returns:
            rel_xy: Estimated target relative position in sonar/body-aligned local frame, shape (N, 2).
            conf: Confidence in [0, 1], shape (N,).
            detected: Reliable detection mask from image frontend, shape (N,).
        """
        img = torch.clamp(self.sonar_image, min=0.0)
        n, h, w = img.shape
        flat = img.view(n, -1)
        maxv = flat.max(dim=1).values
        meanv = flat.mean(dim=1)
        # Adaptive threshold: keep bright returns while suppressing background speckle.
        thr = torch.maximum(meanv + 0.35 * (maxv - meanv), 0.12 * maxv).view(n, 1, 1)
        weights = torch.where(img >= thr, img, torch.zeros_like(img))
        sumw = weights.sum(dim=(1, 2))
        valid = sumw > 1e-6

        xs = torch.linspace(0.0, float(w - 1), w, device=self.device, dtype=img.dtype).view(1, 1, w)
        ys = torch.linspace(0.0, float(h - 1), h, device=self.device, dtype=img.dtype).view(1, h, 1)
        u = torch.where(valid, (weights * xs).sum(dim=(1, 2)) / (sumw + 1e-6), torch.zeros_like(sumw))
        v = torch.where(valid, (weights * ys).sum(dim=(1, 2)) / (sumw + 1e-6), torch.zeros_like(sumw))

        fov_rad = torch.deg2rad(torch.tensor(float(self.cfg.sonar_cfg.hori_fov), device=self.device))
        bearing = (u / max(float(w - 1), 1.0) - 0.5) * fov_rad
        range_min = float(self.cfg.sonar_cfg.min_range)
        range_max = float(self.cfg.sonar_cfg.max_range)
        frac = v / max(float(h - 1), 1.0)
        # In this sonar view, farther returns are typically toward the top of image.
        est_range = range_max - frac * (range_max - range_min)
        est_range = torch.clamp(est_range, range_min, range_max)

        x_scale = float(getattr(self.cfg, "rel_est_x_scale", 1.0))
        rel_x = est_range * torch.cos(bearing) * x_scale
        # Align sign convention with geometric local y in _sonar_rel_local.
        rel_y_raw = -est_range * torch.sin(bearing)
        y_gain = float(getattr(self.cfg, "rel_est_y_gain", 1.0))
        y_thr = max(float(getattr(self.cfg, "rel_est_y_piecewise_threshold", 0.9)), 0.0)
        y_gain_high = float(getattr(self.cfg, "rel_est_y_gain_high", y_gain))
        rel_y = rel_y_raw * y_gain
        high_mask = torch.abs(rel_y_raw) >= y_thr
        rel_y = torch.where(high_mask, rel_y_raw * y_gain_high, rel_y)
        rel_xy = torch.stack([rel_x, rel_y], dim=-1)
        rel_xy = torch.where(valid.unsqueeze(1), rel_xy, torch.zeros_like(rel_xy))
        # Confidence from four factors:
        # 1) peak intensity, 2) blob area ratio, 3) compactness, 4) temporal consistency.
        peak_score = torch.clamp(maxv, 0.0, 1.0)
        area_pixels = (weights > 0.0).sum(dim=(1, 2)).float()
        area_ratio = area_pixels / float(h * w)
        area_ref = max(float(getattr(self.cfg, "rel_est_area_ref", 0.01)), 1e-6)
        area_score = torch.clamp(area_ratio / area_ref, 0.0, 1.0)
        du2 = (xs - u.view(n, 1, 1)) ** 2
        dv2 = (ys - v.view(n, 1, 1)) ** 2
        second_moment = torch.where(
            valid,
            (weights * (du2 + dv2)).sum(dim=(1, 2)) / (sumw + 1e-6),
            torch.zeros_like(sumw),
        )
        diag2 = float((w - 1) * (w - 1) + (h - 1) * (h - 1))
        spread_norm = torch.clamp(second_moment / (diag2 + 1e-6), 0.0, 1.0)
        compact_sigma = max(float(getattr(self.cfg, "rel_est_compact_sigma", 0.015)), 1e-6)
        compactness_score = torch.exp(-spread_norm / compact_sigma)
        temporal_sigma = max(float(getattr(self.cfg, "rel_est_temporal_sigma", 0.30)), 1e-6)
        rel_prev = self.rel_est_state
        delta = torch.norm(rel_xy - rel_prev, dim=-1)
        temporal_score = torch.exp(-delta / temporal_sigma)
        temporal_score = torch.where(self.prev_rel_valid, temporal_score, torch.ones_like(temporal_score))
        w_peak = float(getattr(self.cfg, "rel_est_conf_w_peak", 0.35))
        w_area = float(getattr(self.cfg, "rel_est_conf_w_area", 0.25))
        w_compact = float(getattr(self.cfg, "rel_est_conf_w_compact", 0.20))
        w_temporal = float(getattr(self.cfg, "rel_est_conf_w_temporal", 0.20))
        w_sum = max(w_peak + w_area + w_compact + w_temporal, 1e-6)
        conf = (
            w_peak * peak_score
            + w_area * area_score
            + w_compact * compactness_score
            + w_temporal * temporal_score
        ) / w_sum
        conf = torch.where(valid, torch.clamp(conf, 0.0, 1.0), torch.zeros_like(conf))
        if bool(getattr(self.cfg, "rel_est_gate_fov_only", True)):
            detected = valid
        else:
            conf_thr = max(float(getattr(self.cfg, "rel_est_conf_threshold", 0.03)), 0.0)
            detected = valid & (conf >= conf_thr)
        if bool(getattr(self.cfg, "rel_est_gate_with_in_fov_gt", False)) and hasattr(self, "_sonar_dist_xy") and hasattr(self, "_sonar_rel_local"):
            dist_xy = self._sonar_dist_xy
            hori_deg = torch.rad2deg(torch.atan2(self._sonar_rel_local[:, 1], self._sonar_rel_local[:, 0]))
            fov_half = float(self.cfg.sonar_cfg.hori_fov) * 0.5
            in_range_gt = (dist_xy >= float(self.cfg.sonar_cfg.min_range)) & (dist_xy <= float(self.cfg.sonar_cfg.max_range))
            in_fov_gt = in_range_gt & (torch.abs(hori_deg) <= fov_half)
            detected = detected | in_fov_gt
        # Do not hard-zero rel_xy on low confidence; keep geometry estimate continuous.
        # ``detected`` is returned as a reliability flag and used downstream for guarded updates.
        return rel_xy, conf, detected

    def _get_observations(self) -> dict:
        self._ensure_geom_sonar_for_step()
        if self._last_beam_raw_for_debug is not None:
            self._debug_print_beam_raw(self._last_beam_raw_for_debug)

        self._debug_print_y()

        expected_obs_dim = self._policy_obs_dim_from_cfg()
        mode = str(getattr(self.cfg, "policy_obs_mode", "image")).lower()
        if mode == "relative_position":
            rel_xy_raw, conf, detected_mask = self._estimate_relative_from_sonar_image()
            alpha = float(getattr(self.cfg, "rel_est_ema_alpha", 0.80))
            alpha = min(max(alpha, 0.0), 0.999)
            lost_decay = float(getattr(self.cfg, "rel_est_lost_decay", 0.95))
            lost_decay = min(max(lost_decay, 0.0), 1.0)
            update_conf_thr = max(float(getattr(self.cfg, "rel_est_update_conf_threshold", 0.004)), 0.0)
            conf_span = max(1.0 - update_conf_thr, 1e-6)
            conf_weight = torch.clamp((conf - update_conf_thr) / conf_span, 0.0, 1.0)
            blend = (1.0 - alpha) * conf_weight
            updated_state = (1.0 - blend).unsqueeze(1) * self.rel_est_state + blend.unsqueeze(1) * rel_xy_raw
            low_conf_mask = conf < update_conf_thr
            self.rel_est_state = torch.where(
                low_conf_mask.unsqueeze(1),
                self.rel_est_state * lost_decay,
                updated_state,
            )
            rel_xy = self.rel_est_state
            dt = max(float(self.cfg.sim.dt * self.cfg.decimation), 1e-6)
            rel_vel = (rel_xy - self.prev_rel_est) / dt
            valid_now = detected_mask
            valid_pair = valid_now & self.prev_rel_valid
            rel_vel = torch.where(valid_pair.unsqueeze(1), rel_vel, torch.zeros_like(rel_vel))
            vy_clip = max(float(getattr(self.cfg, "rel_est_vy_clip", 1.5)), 1e-6)
            rel_vel[:, 1] = torch.clamp(rel_vel[:, 1], -vy_clip, vy_clip)
            vy_warmup_steps = max(int(getattr(self.cfg, "rel_est_vy_warmup_steps", 2)), 0)
            if vy_warmup_steps > 0:
                warmup_mask = self.episode_length_buf < vy_warmup_steps
                rel_vel[warmup_mask, 1] = 0.0
            self.prev_rel_est = rel_xy.clone()
            self.prev_rel_valid = valid_now.clone()

            robot_wz = self.robot_ang_vel[:, 2].unsqueeze(1)
            tail_pad = torch.zeros(self.num_envs, 1, device=self.device)
            last_action = self.actions
            obs = torch.cat(
                [
                    robot_wz,
                    tail_pad,
                    last_action,
                    rel_xy[:, 1:2],
                    rel_vel[:, 1:2],
                ],
                dim=-1,
            )
            assert obs.shape[1] == expected_obs_dim, (
                f"Lh2Env relative obs dim mismatch: runtime={obs.shape[1]}, expected={expected_obs_dim}. "
                "Check observation_space and policy_obs_mode."
            )
        elif mode == "beam":
            obs = self._build_beam_policy_obs()
            assert obs.shape[1] == expected_obs_dim, (
                f"Lh2Env beam obs dim mismatch: runtime={obs.shape[1]}, expected={expected_obs_dim}. "
                "Check beam_count and policy_obs_mode."
            )
        elif self.sonar_image is not None:
            sonar_obs_image = self.sonar_image
            sonar_image_dim = sonar_obs_image.shape[1] * sonar_obs_image.shape[2]
            runtime_obs_dim = sonar_image_dim + self._policy_tail_dim()
            assert runtime_obs_dim == expected_obs_dim, (
                f"Lh2Env runtime obs dim mismatch: runtime={runtime_obs_dim}, expected={expected_obs_dim}. "
                "Check oceansim_sonar_cfg height/width and observation_space in cfg."
            )
            sonar_image_flat = sonar_obs_image.reshape(self.num_envs, -1)
            robot_wz = self.robot_ang_vel[:, 2].unsqueeze(1)
            tail_pad = torch.zeros(self.num_envs, 1, device=self.device)
            last_action = self.actions
            tail = self._append_policy_tail_extensions(torch.cat([robot_wz, tail_pad, last_action], dim=-1))
            obs = torch.cat([sonar_image_flat, tail], dim=-1)
        else:
            obs = torch.zeros(self.num_envs, expected_obs_dim, device=self.device)

        # Keep a rolling beam history for optional debug/ablation (not fed to policy).
        if self.cfg.beam_history_len > 1:
            self.beam_history[:, 1:] = self.beam_history[:, :-1].clone()
        self.beam_history[:, 0] = self.sonar_beam_data
        
        obs = torch.clamp(obs, -100.0, 100.0)
        
        observations = {"policy": obs}
        critic_dim = int(getattr(self.cfg, "state_space", 0) or 0)
        if critic_dim > 0:
            pol_dim = self._policy_obs_dim_from_cfg()
            if critic_dim != pol_dim + 1:
                raise ValueError(
                    f"Lh2Env critic state dim mismatch: cfg.state_space={critic_dim}, "
                    f"expected policy_dim+1={pol_dim + 1} (policy obs + body bearing delta per step)."
                )
            bearing = self._body_plane_bearing_to_target()
            dbearing = wrap_to_pi(bearing - self.prev_body_bearing_to_target)
            self._last_body_bearing_rad = bearing.detach()
            self._last_body_bearing_dtheta_rad = dbearing.detach()
            critic_obs = torch.cat([obs, dbearing.unsqueeze(-1)], dim=-1)
            observations["critic"] = torch.clamp(critic_obs, -100.0, 100.0)
            self._print_body_bearing_debug(bearing)
            self.prev_body_bearing_to_target = bearing.detach()

        if not self._obs_audit_logged:
            self._obs_audit_logged = True
        return observations

    def _debug_print_beam_raw(self, beam_data_raw: torch.Tensor | None) -> None:
        """Print env0 beam debug information.

        Default behavior prints a concise direction summary.
        Set ``cfg.debug_print_beam_full`` to print all beam triplets [angle_deg, range_m, intensity].
        """
        if not getattr(self.cfg, "debug_print_beam_raw", False):
            return
        if bool(getattr(self.cfg, "debug_print_only_in_play", True)) and str(getattr(self.cfg, "run_mode", "train")).lower() != "play":
            return
        if beam_data_raw is None or beam_data_raw.numel() == 0:
            return
        interval = max(1, int(getattr(self.cfg, "debug_print_beam_interval", 20)))
        step = int(self.episode_length_buf[0].item()) if self.episode_length_buf.numel() > 0 else 0
        if step % interval != 0:
            return
        env0 = beam_data_raw[0]
        if bool(getattr(self.cfg, "debug_print_beam_full", False)):
            beam_rows: list[str] = []
            for beam_idx in range(env0.shape[0]):
                angle_deg = float(env0[beam_idx, 0].item())
                range_m = float(env0[beam_idx, 1].item())
                intensity = float(env0[beam_idx, 2].item())
                beam_rows.append(
                    f"#{beam_idx:03d}: angle={angle_deg:+7.2f}deg, range={range_m:6.3f}m, intensity={intensity:7.4f}"
                )
            print(
                f"[Beam full][step={step}] env0 total_beams={env0.shape[0]}\n" + "\n".join(beam_rows),
                flush=True,
            )
            return

        intensities = env0[:, 2]
        peak_idx = int(torch.argmax(intensities).item())
        peak_angle_deg = float(env0[peak_idx, 0].item())
        peak_range_m = float(env0[peak_idx, 1].item())
        peak_intensity = float(env0[peak_idx, 2].item())

        if peak_angle_deg < -5.0:
            direction = "left"
        elif peak_angle_deg > 5.0:
            direction = "right"
        else:
            direction = "center"

        print(
            f"[Beam dir][step={step}] target≈{direction} "
            f"(angle={peak_angle_deg:+.1f}deg, range={peak_range_m:.2f}m, intensity={peak_intensity:.3f})",
            flush=True,
        )

    def _debug_print_y(self) -> None:
        """Print env-0 yaw/bearing debug (angles and angular rates only)."""
        if not getattr(self.cfg, "debug_print_y", False):
            return
        if bool(getattr(self.cfg, "debug_print_only_in_play", True)) and str(getattr(self.cfg, "run_mode", "train")).lower() != "play":
            return
        interval = max(1, int(getattr(self.cfg, "debug_print_interval", 5)))
        step = int(self.episode_length_buf[0].item()) if self.episode_length_buf.numel() > 0 else 0
        if step % interval != 0:
            return

        ridx = 0
        action = float(self.actions[ridx, 0].item()) if self.actions.numel() > 0 else 0.0
        net_wz_cmd = action * float(self.cfg.action_scale_angular)
        effective_wz_cmd = float(self._effective_robot_yaw_rate_command()[ridx].item())
        robot_wz = float(self.robot_ang_vel[ridx, 2].item())
        bearing_rad = float(self._body_plane_bearing_to_target()[ridx].item())
        bearing_deg = math.degrees(bearing_rad)
        robot_wz_deg_s = math.degrees(robot_wz)
        net_wz_deg_s = math.degrees(net_wz_cmd)
        eff_wz_deg_s = math.degrees(effective_wz_cmd)
        print(
            f"[Lh2 rel][step={step}] "
            f"bearing_body={bearing_deg:+.2f}° "
            f"robot_wz={robot_wz_deg_s:+.2f}°/s "
            f"action={action:+.3f} net_wz_cmd={net_wz_deg_s:+.2f}°/s effective_wz_cmd={eff_wz_deg_s:+.2f}°/s",
            flush=True,
        )

    def _get_rewards(self) -> torch.Tensor:
        self._ensure_geom_sonar_for_step()
        distance_xy = self._sonar_dist_xy
        bearing = self._body_plane_bearing_to_target()
        lateral_error = torch.abs(bearing)
        lateral_progress = torch.clamp(self.prev_lateral_error - lateral_error, -1.0, 1.0)
        diverge_penalty = torch.clamp(lateral_error - self.prev_lateral_error, min=0.0)

        # Detection + alive shaping (aligned with OceanSim beam gate when cfg enabled).
        detected_f = self._rl_visible_f
        detected = detected_f > 0.5
        not_detected_f = 1.0 - detected_f
        sonar_detection_reward = detected_f
        alive_progress_max = max(float(getattr(self.cfg, "alive_progress_max", 1.0)), 0.0)
        alive_reward = torch.clamp(
            self.episode_length_buf.float() / float(max(1, self.max_episode_length - 1)),
            0.0,
            1.0,
        ) * alive_progress_max

        # Simplified tracking layer.
        lateral_reward = torch.exp(-torch.square(lateral_error / self.cfg.lateral_tracking_sigma))

        # Angular velocity match: legacy orbit w_inst, or bearing PD (general motion in annulus).
        dt_step = max(float(self.cfg.sim.dt * self.cfg.decimation), 1e-6)
        psi_dot = wrap_to_pi(bearing - self.prev_bearing_reward) / dt_step
        mode = str(getattr(self.cfg, "velocity_match_mode", "bearing_pd")).lower()
        if mode == "legacy_orbit_w":
            w_tgt = self.target_orbit_w_inst
            vel_w_error = torch.abs(self.robot_ang_vel[:, 2] - w_tgt)
        else:
            kp = float(getattr(self.cfg, "velocity_match_kp", 1.2))
            kd = float(getattr(self.cfg, "velocity_match_kd", 0.85))
            omega_des = -kp * bearing - kd * psi_dot
            vel_w_error = torch.abs(self.robot_ang_vel[:, 2] - omega_des)
        sigma_vm = max(float(self.cfg.velocity_match_sigma), 1e-6)
        velocity_match_reward = torch.exp(-torch.square(vel_w_error / sigma_vm))
        vel_match_active = lateral_error < float(getattr(self.cfg, "velocity_match_lateral_threshold", 0.25))
        action_y = self.actions[:, 0]
        w_cmd = self._scaled_robot_yaw_rate_command()
        dir_eps = self._direction_command_deadband_wz()
        direction_active = lateral_error > self.cfg.direction_lateral_threshold
        freeze_steps = max(int(getattr(self.cfg, "startup_freeze_steps", 0)), 0)
        if freeze_steps > 0:
            direction_active = direction_active & (self.episode_length_buf >= freeze_steps)
        action_wrong = (bearing * w_cmd) < -dir_eps
        action_wrong_direction_penalty = torch.where(direction_active, action_wrong.float(), torch.zeros_like(distance_xy))
        turn_bonus_scale = float(getattr(self.cfg, "rew_scale_direction_turn_bonus", 0.0))
        bonus_span = max(float(getattr(self.cfg, "direction_turn_bonus_span", 1.2)), 1e-6)
        turn_ok = (bearing * w_cmd) > dir_eps
        bonus_mag = torch.clamp(lateral_error / bonus_span, 0.0, 1.0)
        direction_turn_bonus = detected_f * turn_bonus_scale * direction_active.float() * turn_ok.float() * bonus_mag

        # Stabilization near alignment.
        near_aligned = lateral_error < self.cfg.lateral_idle_threshold
        idle_speed_penalty = torch.where(
            near_aligned, torch.abs(self.robot_ang_vel[:, 2]), torch.zeros_like(distance_xy)
        )
        action_diff = self.actions - self.previous_actions
        smooth_mask = (lateral_error < self.cfg.smooth_lateral_threshold).float()
        action_smooth_penalty = torch.sum(torch.square(action_diff), dim=-1) * smooth_mask
        action_magnitude_penalty = torch.abs(action_y)

        # Safety penalties.
        collision_penalty = torch.where(
            distance_xy < self.cfg.min_safe_distance,
            torch.ones_like(distance_xy) * self.cfg.rew_scale_collision,
            torch.zeros_like(distance_xy),
        )
        out_of_range_penalty = torch.where(
            distance_xy > self.cfg.max_tracking_distance,
            torch.ones_like(distance_xy) * self.cfg.rew_scale_out_of_range,
            torch.zeros_like(distance_xy),
        )
        # ``out_of_view_steps`` is updated once per env-step in ``_get_dones``.
        # Reuse that counter here to avoid double-incrementing lost-step penalties.
        lost_ramp = torch.clamp(
            self.out_of_view_steps.float() / float(max(1, int(self.cfg.out_of_view_penalty_ramp_steps))),
            0.0,
            1.0,
        )
        out_of_view_penalty = not_detected_f * lost_ramp * self.cfg.rew_scale_out_of_view

        # Simplified total reward (fewer terms, fewer conflicts).
        total_reward = (
            self.cfg.rew_scale_alive * alive_reward
            + self.cfg.rew_scale_sonar_detection * sonar_detection_reward
            + detected_f * self.cfg.rew_scale_lateral * lateral_reward
            + self.cfg.rew_scale_lateral_progress * lateral_progress
            + self.cfg.rew_scale_diverge * diverge_penalty
            + detected_f * vel_match_active.float() * self.cfg.rew_scale_velocity_match * velocity_match_reward
            + self.cfg.rew_scale_action_wrong_direction * action_wrong_direction_penalty
            + direction_turn_bonus
            + detected_f * self.cfg.rew_scale_idle_speed * idle_speed_penalty
            + detected_f * self.cfg.rew_scale_action_smooth * action_smooth_penalty
            + self.cfg.rew_scale_action_magnitude * action_magnitude_penalty
            + out_of_view_penalty
            + collision_penalty
            + out_of_range_penalty
        )
        dir_thr = float(getattr(self.cfg, "wrong_direction_reset_lateral_threshold", 0.10))
        dir_valid = (torch.abs(bearing) > dir_thr) & (torch.abs(w_cmd) > dir_eps)
        sign_match = (bearing * w_cmd) > 0.0
        sign_match_rate = torch.where(
            torch.any(dir_valid),
            sign_match[dir_valid].float().mean(),
            torch.tensor(0.0, device=self.device),
        )
        self.prev_detected = detected.clone()
        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        if str(getattr(self.cfg, "run_mode", "train")).lower() == "train":
            # Keep training console concise: only core health/performance signals.
            self.extras["log"] = {
                "visible_ratio": detected_f.mean(),
                # Split visibility for sanity: ``visible_ratio`` uses beam∩geom or beam∪geom per cfg.reward_visibility_combine.
                "visible_ratio_geom": self._diag_visible_mean_geom,
                "visible_ratio_beam_gate": self._diag_visible_mean_beam,
                # Consecutive env-steps with _rl_visible_f < 0.5 (updated in _get_dones before reward); compare to cfg.out_of_view_reset_steps.
                "mean_out_of_view_steps": self.out_of_view_steps.float().mean(),
                "mean_abs_bearing_deg": torch.rad2deg(lateral_error).mean(),
                "rew_weighted_total": total_reward.mean(),
                "sign_match_rate": sign_match_rate,
                "effective_wz_cmd_abs_deg_s": torch.rad2deg(torch.abs(w_cmd)).mean(),
                "beam_peak_angle_deg": self._beam_peak_angle_deg,
                "beam_peak_range_m": self._beam_peak_range_m,
                "beam_peak_intensity": self._beam_peak_intensity,
            }
        else:
            self.extras["log"] = {
                "visible_ratio": detected_f.mean(),
                "visible_ratio_geom": self._diag_visible_mean_geom,
                "visible_ratio_beam_gate": self._diag_visible_mean_beam,
                "mean_out_of_view_steps": self.out_of_view_steps.float().mean(),
                "mean_abs_bearing_deg": torch.rad2deg(lateral_error).mean(),
                "tracking_phase": torch.tensor(
                    float(str(self.cfg.training_phase).lower() == "tracking"), device=self.device
                ),
                "rew_weighted_visibility": (
                    self.cfg.rew_scale_sonar_detection * sonar_detection_reward
                    + out_of_view_penalty
                ).mean(),
                "rew_weighted_total": total_reward.mean(),
                "lateral_hint_weight": torch.tensor(0.0, device=self.device),
                "sign_match_rate": sign_match_rate,
                "effective_wz_cmd_abs_deg_s": torch.rad2deg(torch.abs(w_cmd)).mean(),
                "beam_softmax_gate_ratio": self._beam_softmax_gate_ratio,
                "beam_intensity_raw_mean": self._beam_intensity_raw_mean,
                "beam_intensity_raw_max": self._beam_intensity_raw_max,
                "beam_intensity_norm_mean": self._beam_intensity_norm_mean,
                "beam_intensity_norm_max": self._beam_intensity_norm_max,
                "beam_gate_peak_pass_ratio": self._beam_gate_peak_pass_ratio,
                "beam_gate_mean_pass_ratio": self._beam_gate_mean_pass_ratio,
                "beam_peak_angle_deg": self._beam_peak_angle_deg,
                "beam_peak_range_m": self._beam_peak_range_m,
                "beam_peak_intensity": self._beam_peak_intensity,
                "ball_bearing_body_deg_env0": torch.rad2deg(self._last_body_bearing_rad[0]),
                "ball_dbearing_body_deg_env0": torch.rad2deg(self._last_body_bearing_dtheta_rad[0]),
            }
        self.prev_lateral_error = lateral_error.detach()
        self.prev_bearing_reward = bearing.detach()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_geom_sonar_for_step()
        distance_xy = self._sonar_dist_xy
        
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        out_of_range = distance_xy > self.cfg.max_tracking_distance
        local_robot_y = self.robot_pos[:, 1] - self.scene.env_origins[:, 1]
        robot_out_of_bounds = torch.abs(local_robot_y) > float(self.cfg.robot_lateral_limit)

        # 与奖励同源的可见度：几何 或 beam 门控（见 _cache_rl_visibility_float）。
        out_of_view = self._rl_visible_f < 0.5
        self.out_of_view_steps = torch.where(
            out_of_view, self.out_of_view_steps + 1, torch.zeros_like(self.out_of_view_steps)
        )
        if self.cfg.reset_when_target_out_of_view:
            out_of_view_timeout = self.out_of_view_steps >= int(self.cfg.out_of_view_reset_steps)
        else:
            out_of_view_timeout = torch.zeros_like(out_of_range)

        # Reset when the policy keeps commanding the wrong action sign for multiple steps.
        bearing = self._body_plane_bearing_to_target()
        w_cmd = self._scaled_robot_yaw_rate_command()
        dir_thr = float(getattr(self.cfg, "wrong_direction_reset_lateral_threshold", 0.10))
        dir_eps = self._direction_command_deadband_wz()
        wrong_act = (torch.abs(bearing) > dir_thr) & ((bearing * w_cmd) < -dir_eps)
        freeze_steps = max(int(getattr(self.cfg, "startup_freeze_steps", 0)), 0)
        if freeze_steps > 0:
            wrong_act = wrong_act & (self.episode_length_buf >= freeze_steps)
        self.wrong_direction_steps = torch.where(
            wrong_act,
            self.wrong_direction_steps + 1,
            torch.zeros_like(self.wrong_direction_steps),
        )
        wrong_direction_timeout = self.wrong_direction_steps >= int(getattr(self.cfg, "wrong_direction_reset_steps", 12))
        if not bool(getattr(self.cfg, "wrong_direction_reset_enabled", False)):
            wrong_direction_timeout = torch.zeros_like(out_of_range)

        terminated = out_of_range | out_of_view_timeout | robot_out_of_bounds | wrong_direction_timeout
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0:
            self._geom_sonar_cache_invalidated = True

        num_resets = len(env_ids)
        
        env_origins = self.scene.env_origins[env_ids]

        robot_pos = torch.zeros(num_resets, 3, device=self.device)
        robot_pos[:, 2] = 10.0
        robot_quat = torch.zeros(num_resets, 4, device=self.device)
        robot_quat[:, 0] = 1.0

        robot_pos_w = robot_pos + env_origins

        target_quat = torch.zeros(num_resets, 4, device=self.device)
        target_quat[:, 0] = 1.0

        motion_mode = str(getattr(self.cfg, "target_motion_mode", "orbit_robot")).lower()
        if motion_mode == "orbit_robot":
            # 小球初始在机体前向水平投影、距离 Rt；周向角速度 w_inst 在 [min,max] 随机（相位沿前向）。
            ph = self._orbit_phase_from_robot_forward_xy(robot_quat)
            self.target_orbit_phase[env_ids] = ph
            r_lo, r_hi = self._orbit_radius_bounds()
            if bool(getattr(self.cfg, "target_orbit_randomize_radius_on_reset", False)):
                Rt = sample_uniform(r_lo, r_hi, (num_resets,), self.device).to(dtype=robot_pos_w.dtype)
            else:
                r_nom = float(getattr(self.cfg, "target_orbit_radius", 1.5))
                Rt = torch.full((num_resets,), min(max(r_nom, r_lo), r_hi), device=self.device, dtype=robot_pos_w.dtype)
            self._target_orbit_radius_env[env_ids] = Rt
            w_lo2 = float(getattr(self.cfg, "target_orbit_angular_vel_min", -0.30))
            w_hi2 = float(getattr(self.cfg, "target_orbit_angular_vel_max", 0.30))
            if w_hi2 < w_lo2:
                w_lo2, w_hi2 = w_hi2, w_lo2
            self.target_orbit_w_inst[env_ids] = sample_uniform(w_lo2, w_hi2, (num_resets,), self.device)
            target_pos_w = torch.zeros(num_resets, 3, device=self.device)
            target_pos_w[:, 0] = robot_pos_w[:, 0] + Rt * torch.cos(ph)
            target_pos_w[:, 1] = robot_pos_w[:, 1] + Rt * torch.sin(ph)
            target_pos_w[:, 2] = robot_pos_w[:, 2]
        else:
            self.target_orbit_phase[env_ids] = 0.0
            self.target_orbit_w_inst[env_ids] = 0.0
            target_pos = torch.zeros(num_resets, 3, device=self.device)
            target_pos[:, 0] = self.cfg.target_fixed_x
            target_pos[:, 1] = 0.0
            target_pos[:, 2] = 10.0
            target_pos_w = target_pos + env_origins

        robot_pose = torch.cat([robot_pos_w, robot_quat], dim=-1)
        target_pose = torch.cat([target_pos_w, target_quat], dim=-1)
        
        self.robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(num_resets, 6, device=self.device), env_ids=env_ids)
        
        self.target.write_root_pose_to_sim(target_pose, env_ids=env_ids)
        self.target.write_root_velocity_to_sim(torch.zeros(num_resets, 6, device=self.device), env_ids=env_ids)
        
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.target_velocities[env_ids] = 0.0
        self.target_y_sign[env_ids] = torch.where(
            torch.rand(num_resets, device=self.device) < 0.5,
            torch.ones(num_resets, device=self.device),
            -torch.ones(num_resets, device=self.device),
        )
        v_lo, v_hi = self.cfg.target_speed_range_start
        self.target_current_speed[env_ids] = sample_uniform(v_lo, v_hi, (num_resets,), self.device)
        self.target_speed_goal[env_ids] = self.target_current_speed[env_ids].clone()
        self.target_waypoint_idx[env_ids] = 0
        
        # 4. 环境重置时正确处理传感器
        # 重置声纳数据缓冲区，但不重新初始化传感器
        self.oceansim_sonar.reset_envs(env_ids)
        
        self.sonar_range[env_ids] = 0.0
        self.sonar_angle[env_ids] = 0.0
        self.sonar_intensity[env_ids] = 0.0
        self.ekf_position[env_ids] = 0.0
        self.ekf_velocity[env_ids] = 0.0
        # Initial |bearing| after reset (matches reward lateral_error).
        rel_w0 = target_pos_w - robot_pos_w
        rel_w0[:, 2] = 0.0
        rel_b0 = quat_apply_inverse(robot_quat, rel_w0)
        b0 = wrap_to_pi(torch.atan2(rel_b0[:, 1], rel_b0[:, 0]))
        self.prev_lateral_error[env_ids] = torch.abs(b0)

        self.prev_body_bearing_to_target[env_ids] = b0
        self.prev_bearing_reward[env_ids] = b0
        self._last_body_bearing_rad[env_ids] = b0
        self._last_body_bearing_dtheta_rad[env_ids] = 0.0
        self.prev_rel_est[env_ids] = 0.0
        self.prev_rel_valid[env_ids] = False
        self.rel_est_state[env_ids] = 0.0
        self.beam_history[env_ids] = 0.0
        self.out_of_view_steps[env_ids] = 0
        self.prev_detected[env_ids] = False
        self.reversal_active[env_ids] = False
        self.reversal_age[env_ids] = 0
        self.reversal_start_error[env_ids] = 0.0
        self.reversal_total[env_ids] = 0
        self.reversal_success[env_ids] = 0
        self.prev_target_y_sign[env_ids] = self.target_y_sign[env_ids]
        self.wrong_direction_steps[env_ids] = 0

    def close(self):
        """清理环境资源，包括声呐传感器"""
        if hasattr(self, 'oceansim_sonar'):
            self.oceansim_sonar.close()
        super().close()
