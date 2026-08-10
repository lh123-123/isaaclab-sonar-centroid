# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sonar centroid tracking on WarpAUV (Phase A0–A2: hover, P-control, residual PPO)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab_tasks.direct.sonar_flow_hover.modules.actions import apply_target_motion

from .modules.auv_decoupled_controller import AUVDecoupledController
from .modules.auv_dynamics import WarpAUVDynamicsHelper
from .modules.auv_image_range_base import compute_image_range_base_setpoints
from .modules.auv_position_lead_base import compute_position_lead_base_setpoints
from .modules.auv_lateral_controller import AUVLateralController
from .modules.rewards import compute_auv_residual_reward
from .sonar_centroid_track_env import SonarCentroidTrackEnv
from .sonar_centroid_track_auv_env_cfg import SonarCentroidTrackAUVEnvCfg


class SonarCentroidTrackAUVEnv(SonarCentroidTrackEnv):
    """WarpAUV + OceanSim sonar; optional compensatory PPO on image-range base law."""

    cfg: SonarCentroidTrackAUVEnvCfg

    def __init__(self, cfg: SonarCentroidTrackAUVEnvCfg, render_mode: str | None = None, **kwargs):
        self._auv_pwm = torch.zeros(cfg.scene.num_envs, 6, device=cfg.sim.device)
        self._range_ref_gt_m = torch.full((cfg.scene.num_envs,), float("nan"), device=cfg.sim.device)
        super().__init__(cfg, render_mode, **kwargs)
        self._range_bias_m = torch.zeros(self.num_envs, device=self.device)
        self._range_bias_sway_at_calib_m = torch.zeros(self.num_envs, device=self.device)
        self._range_ref_hold_m = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._range_meas_ema = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._beam_prior_az_rad_reset = torch.zeros(self.num_envs, device=self.device)
        self._beam_range_cache_step = -1
        self._beam_range_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._auv_dyn = WarpAUVDynamicsHelper(cfg, self.num_envs, self.device, self.robot)
        self._auv_lat = AUVLateralController(cfg, self.num_envs, self.device)
        self._auv_dec = AUVDecoupledController(cfg, self.num_envs, self.device)
        self._auv_ff_vy = torch.zeros(self.num_envs, device=self.device)
        self._auv_ff_vy_active = False
        if bool(getattr(self.cfg, "auv_residual_ppo", False)):
            n_act = int(getattr(self.cfg, "auv_residual_action_dim", 2))
            self.actions = torch.zeros(self.num_envs, n_act, device=self.device)
            self._prev_actions = torch.zeros(self.num_envs, n_act, device=self.device)

    def _auv_residual_mode(self) -> bool:
        return bool(getattr(self.cfg, "auv_physics_mode", False)) and bool(
            getattr(self.cfg, "auv_residual_ppo", False)
        )

    def _auv_decoupled_mode(self) -> bool:
        return str(getattr(self.cfg, "auv_control_mode", "decoupled")).lower() == "decoupled"

    def _compute_auv_pwm(
        self,
        desired_sway: torch.Tensor,
        *,
        desired_surge: torch.Tensor | None = None,
        target_vy: torch.Tensor | None = None,
        desired_heave: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = dict(
            station_origin_w=self._station_origin_w,
            fixed_yaw_w=self._fixed_robot_yaw_w,
            target_vy=target_vy,
            desired_surge_m=desired_surge,
        )
        if self._auv_decoupled_mode():
            return self._auv_dec.compute_pwm(
                desired_sway,
                self.robot,
                desired_heave_m=desired_heave,
                **kwargs,
            )
        return self._auv_lat.compute_pwm(desired_sway, self.robot, **kwargs)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._auv_residual_mode():
            self._prev_actions = self.actions.clone()
            n_act = int(getattr(self.cfg, "auv_residual_action_dim", 2))
            self.actions = actions.view(self.num_envs, n_act).float().clamp(-1.0, 1.0)
            return
        if bool(getattr(self.cfg, "auv_physics_mode", False)):
            if actions.numel() == self.num_envs * int(self.cfg.num_auv_actions):
                self._auv_pwm = actions.view(self.num_envs, -1).float().clamp(-1.0, 1.0)
            else:
                self._auv_pwm.zero_()
            self._prev_actions.zero_()
            return
        super()._pre_physics_step(actions)

    def set_auv_feedforward_vy(self, vy: torch.Tensor | None) -> None:
        """Lateral speed feedforward (m/s) for thruster low-level; image-derived or GT vy FF."""
        if vy is None:
            self._auv_ff_vy.zero_()
            self._auv_ff_vy_active = False
        else:
            self._auv_ff_vy = vy.view(-1).float()
            self._auv_ff_vy_active = True

    def _robot_sway_m(self) -> torch.Tensor:
        """Mission-frame lateral sway (fixed heading), matching ball ``target_dy`` frame."""
        if bool(getattr(self.cfg, "auv_physics_mode", False)):
            from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import robot_lateral_sway_signed

            return robot_lateral_sway_signed(
                self._station_origin_w, self.robot.data.root_pos_w, self._fixed_robot_yaw_w
            )
        return super()._robot_sway_m()

    def _robot_surge_m(self) -> torch.Tensor:
        yaw = self._fixed_robot_yaw_w
        forward_w = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
        delta = self.robot.data.root_pos_w - self._station_origin_w
        return torch.sum(delta * forward_w, dim=-1)

    def gt_horiz_range_now_m(self) -> torch.Tensor:
        """Live GT horizontal range (m) sonar mount → beacon, all envs."""
        from .modules.coordinates import gt_sonar_beacon_horiz_range_batch

        return gt_sonar_beacon_horiz_range_batch(
            self._beacon_pos_w,
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self._sonar_mount_offset_b,
        )

    def range_ref_setpoint_m(self) -> torch.Tensor:
        """Fixed distance setpoint for hold (hold_meas / fixed / reset_gt)."""
        return self.range_ref_for_control_m()

    def _update_range_ref_gt(self, env_ids: torch.Tensor) -> None:
        """GT sonar→beacon horizontal range (m) at reset (debug / reset_gt mode)."""
        from .modules.coordinates import gt_sonar_beacon_horiz_range_m

        for idx in env_ids.detach().cpu().tolist():
            self._range_ref_gt_m[idx] = gt_sonar_beacon_horiz_range_m(
                self._beacon_pos_w,
                self.robot.data.root_pos_w,
                self.robot.data.root_quat_w,
                self._sonar_mount_offset_b,
                env_id=idx,
            )

    def _update_beam_prior_azimuth(self, env_ids: torch.Tensor) -> None:
        """Teach fixed-point column → azimuth prior for beam gating (scheme B)."""
        import math

        from .modules.coordinates import polar_col_to_azimuth_rad

        ocfg = self.cfg.oceansim_sonar_cfg
        fov_rad = math.radians(float(ocfg.hori_fov))
        width = int(ocfg.width)
        for idx in env_ids.detach().cpu().tolist():
            az = 0.0
            p_fixed = self.fixed.p_fixed_abs_env(idx)
            if p_fixed is not None:
                az = polar_col_to_azimuth_rad(p_fixed.col, width, fov_rad)
            else:
                p_abs = self.tracker.detected_abs(env_id=idx)
                if p_abs is not None:
                    az = polar_col_to_azimuth_rad(p_abs.col, width, fov_rad)
            self._beam_prior_az_rad_reset[idx] = az

    def _beam_prior_azimuth_rad(self) -> torch.Tensor | None:
        gate_deg = float(getattr(self.cfg, "sonar_beam_azimuth_gate_deg", 0.0))
        if gate_deg <= 0.0:
            return None
        if bool(getattr(self.cfg, "sonar_beam_prior_use_teach", True)):
            return self._beam_prior_az_rad_reset
        import math

        from .modules.coordinates import polar_col_to_azimuth_rad

        ocfg = self.cfg.oceansim_sonar_cfg
        fov_rad = math.radians(float(ocfg.hori_fov))
        width = int(ocfg.width)
        priors = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            p = self.tracker.detected_abs(env_id=i)
            if p is not None:
                priors[i] = polar_col_to_azimuth_rad(p.col, width, fov_rad)
            else:
                priors[i] = self._beam_prior_az_rad_reset[i]
        return priors

    def _beam_peak_raw_m(
        self,
        *,
        peak_threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gated peak-beam range without reset bias."""
        return super().beam_peak_range_meas_m(peak_threshold=peak_threshold)

    def _beam_peak_valid_min(self) -> float:
        return float(getattr(self.cfg, "sonar_beam_peak_valid_min", 0.5))

    def _calibrated_range_raw(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bias-corrected gated beam range, azimuth, peak intensity, control-valid."""
        range_m, azimuth, peak_val, _ = self._beam_peak_raw_m()
        if bool(getattr(self.cfg, "sonar_range_reset_calib", True)) and bool(
            getattr(self.cfg, "auv_physics_mode", False)
        ):
            range_m = range_m + self._range_bias_m
        valid = (peak_val >= self._beam_peak_valid_min()).float()
        return range_m, azimuth, peak_val, valid

    def _apply_range_ema(
        self,
        range_m: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        alpha = float(getattr(self.cfg, "sonar_range_ema_alpha", 0.85))
        if alpha <= 0.0 or not bool(getattr(self.cfg, "auv_physics_mode", False)):
            return range_m
        out = self._range_meas_ema.clone()
        for i in range(self.num_envs):
            if float(valid[i].item()) < 0.5:
                if out[i] == out[i]:
                    continue
                out[i] = range_m[i]
                self._range_meas_ema[i] = range_m[i]
                continue
            prev = self._range_meas_ema[i]
            if prev != prev:
                out[i] = range_m[i]
                self._range_meas_ema[i] = range_m[i]
            else:
                smoothed = alpha * prev + (1.0 - alpha) * range_m[i]
                out[i] = smoothed
                self._range_meas_ema[i] = smoothed
        return out

    def _calibrate_range_bias(self, env_ids: torch.Tensor) -> None:
        """One-shot calib: bias = gt_horiz − beam_raw (scheme A)."""
        if not bool(getattr(self.cfg, "sonar_range_reset_calib", True)):
            self._range_bias_m[env_ids] = 0.0
            return
        self._ensure_pipeline()
        range_raw, _, peak_val, _ = self._beam_peak_raw_m()
        peak_min = self._beam_peak_valid_min()
        gt_now = self.gt_horiz_range_now_m()
        sway = self._robot_sway_m()
        for idx in env_ids.detach().cpu().tolist():
            gt = float(gt_now[idx].item())
            if float(peak_val[idx].item()) >= peak_min and gt == gt:
                self._range_bias_m[idx] = gt - float(range_raw[idx].item())
                self._range_bias_sway_at_calib_m[idx] = float(sway[idx].item())
            else:
                self._range_bias_m[idx] = 0.0
                self._range_bias_sway_at_calib_m[idx] = float(sway[idx].item())

    def _maybe_recalibrate_range_bias(self) -> None:
        """Re-estimate bias when lateral sway or periodic timer exceeds thresholds."""
        if not bool(getattr(self.cfg, "sonar_range_reset_calib", True)):
            return
        if not bool(getattr(self.cfg, "auv_physics_mode", False)):
            return
        sway_thresh = float(getattr(self.cfg, "sonar_range_bias_recalib_sway_m", 0.0))
        period = int(getattr(self.cfg, "sonar_range_bias_recalib_period_steps", 0))
        if sway_thresh <= 0.0 and period <= 0:
            return

        self._ensure_pipeline()
        sway = self._robot_sway_m()
        step = int(self.common_step_counter)
        env_ids: list[int] = []
        for i in range(self.num_envs):
            if period > 0 and step > 0 and step % period == 0:
                env_ids.append(i)
                continue
            if sway_thresh > 0.0 and abs(float(sway[i].item()) - float(
                self._range_bias_sway_at_calib_m[i].item()
            )) >= sway_thresh:
                env_ids.append(i)
        if env_ids:
            self._calibrate_range_bias(torch.tensor(env_ids, device=self.device, dtype=torch.long))
            self._invalidate_beam_range_cache()

    def _update_range_hold_ref(self, env_ids: torch.Tensor) -> None:
        """Store calibrated sonar range at reset for hold_meas control (scheme C)."""
        range_cal, _, peak_val, valid = self._calibrated_range_raw()
        for idx in env_ids.detach().cpu().tolist():
            if float(valid[idx].item()) >= 0.5:
                val = float(range_cal[idx].item())
                self._range_ref_hold_m[idx] = val
                self._range_meas_ema[idx] = val
            else:
                gt = float(self._range_ref_gt_m[idx].item())
                self._range_ref_hold_m[idx] = gt if gt == gt else float("nan")
                self._range_meas_ema[idx] = self._range_ref_hold_m[idx]

    def recalibrate_sonar_range(self, env_ids: torch.Tensor | None = None) -> None:
        """Re-run A+B+C after fixed-point teach refresh (eval / debug)."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else self.robot._ALL_INDICES
        self._update_range_ref_gt(env_ids_t)
        self._update_beam_prior_azimuth(env_ids_t)
        self._calibrate_range_bias(env_ids_t)
        self._update_range_hold_ref(env_ids_t)

    def range_ref_for_control_m(self) -> torch.Tensor:
        """Distance-loop reference: hold_meas, reset_gt, or fixed setpoint (m)."""
        mode = str(getattr(self.cfg, "sonar_range_ref_mode", "hold_meas")).lower()
        if mode == "fixed":
            fixed = float(getattr(self.cfg, "sonar_range_ref_fixed_m", 3.0))
            return torch.full((self.num_envs,), fixed, device=self.device)
        if mode == "hold_meas" and bool(getattr(self.cfg, "auv_physics_mode", False)):
            return self._range_ref_hold_m
        return self._range_ref_gt_m

    def _invalidate_beam_range_cache(self) -> None:
        self._beam_range_cache_step = -1
        self._beam_range_cache = None

    def beam_peak_range_meas_m(
        self,
        *,
        peak_threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gated peak-beam range + bias calib + EMA + peak-valid gate (cached per env step)."""
        del peak_threshold
        self._maybe_recalibrate_range_bias()
        step = int(self.common_step_counter)
        if self._beam_range_cache_step == step and self._beam_range_cache is not None:
            return self._beam_range_cache
        range_m, azimuth, peak_val, valid = self._calibrated_range_raw()
        range_out = self._apply_range_ema(range_m, valid)
        self._beam_range_cache_step = step
        self._beam_range_cache = (range_out, azimuth, peak_val, valid)
        return self._beam_range_cache

    def _residual_scales(self) -> tuple[float, float]:
        return (
            float(getattr(self.cfg, "auv_max_residual_sway", 0.20)),
            float(getattr(self.cfg, "auv_max_residual_surge", 0.35)),
        )

    def _robot_sway_vy(self) -> torch.Tensor:
        yaw = self._fixed_robot_yaw_w
        lateral_w = torch.stack([-torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)], dim=-1)
        return torch.sum(self.robot.data.root_lin_vel_w * lateral_w, dim=-1)

    def _sonar_range_meas_m(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env peak-beam sonar range (m) and validity for base range-hold."""
        range_meas, _, _, valid = self.beam_peak_range_meas_m()
        return range_meas, valid

    def _gt_lateral_blend_alpha(self) -> float:
        """1=full GT position_lead sway, 0=full image P sway (step C sim→real)."""
        a = float(getattr(self.cfg, "auv_gt_lateral_blend", 1.0))
        return max(0.0, min(1.0, a))

    def _compute_base_setpoints(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed PD base (position_lead / image_range / blend) before PPO residual."""
        step_dt = float(self.cfg.sim.dt * self.cfg.decimation)
        ocfg = self.cfg.oceansim_sonar_cfg
        base_law = str(getattr(self.cfg, "auv_base_law", "position_lead")).lower()
        amp = float(self.cfg.lateral_target_amp_m)
        range_meas, range_valid = self._sonar_range_meas_m()
        range_ref = self.range_ref_for_control_m()
        range_db = float(getattr(self.cfg, "sonar_range_deadband_m", 0.0))
        close_m = float(getattr(self.cfg, "sonar_range_close_margin_m", 0.25))
        kp_fwd = float(getattr(self.cfg, "auv_kp_range_fwd", 0.35))
        kp_back = float(getattr(self.cfg, "auv_kp_range_back", 3.0))
        inhibit_fwd = bool(getattr(self.cfg, "auv_inhibit_surge_forward", False))
        alpha = self._gt_lateral_blend_alpha()

        surge_kw = dict(
            range_meas=range_meas,
            range_valid=range_valid,
            range_deadband_m=range_db,
            kp_range_fwd=kp_fwd,
            kp_range_back=kp_back,
            range_close_margin_m=close_m,
            inhibit_surge_forward=inhibit_fwd,
            kp_range=float(getattr(self.cfg, "auv_kp_range", 0.0)),
            kp_surge_hold=float(getattr(self.cfg, "auv_kp_surge_hold", 2.0)),
            max_surge=float(getattr(self.cfg, "auv_max_surge", 1.0)),
        )

        use_image = base_law == "image_range" or alpha < 1.0 - 1.0e-6
        use_gt = base_law != "image_range" and alpha > 1.0e-6

        img_sway = gt_sway = None
        base_surge = None

        if use_image:
            img_sway, base_surge = compute_image_range_base_setpoints(
                tracker=self.tracker,
                robot_sway_m=self._robot_sway_m(),
                surge_robot_m=self._robot_surge_m(),
                range_ref_gt_m=range_ref,
                polar_width=int(ocfg.width),
                polar_height=int(ocfg.height),
                step_dt=step_dt,
                kp_img=float(getattr(self.cfg, "auv_image_p_kp", 1.0)),
                deadband=float(getattr(self.cfg, "auv_image_p_deadband", 0.003)),
                sway_amp_m=amp,
                **surge_kw,
            )

        if use_gt:
            gt_sway, surge_pl = compute_position_lead_base_setpoints(
                target_dy=self._target_dy_m(),
                target_vy=self._target_dy_vel_m(),
                robot_sway_m=self._robot_sway_m(),
                robot_sway_vy=self._robot_sway_vy(),
                surge_robot_m=self._robot_surge_m(),
                kp_sway=float(getattr(self.cfg, "auv_pl_kp_sway", 6.5)),
                lead_s=float(getattr(self.cfg, "auv_pl_lead_s", 0.65)),
                kp_vy=float(getattr(self.cfg, "auv_pl_kp_vy", 0.6)),
                sway_amp_m=amp,
                range_ref_m=range_ref,
                **surge_kw,
            )
            if base_surge is None:
                base_surge = surge_pl

        if use_gt and use_image:
            base_sway = alpha * gt_sway + (1.0 - alpha) * img_sway
        elif use_gt:
            base_sway = gt_sway
        else:
            base_sway = img_sway

        return base_sway, base_surge

    def _apply_residual_control(self) -> None:
        self._ensure_pipeline()
        base_sway, base_surge = self._compute_base_setpoints()
        max_dsway, max_dsurge = self._residual_scales()
        delta_sway = self.actions[:, 0] * max_dsway
        delta_surge = self.actions[:, 1] * max_dsurge
        desired_sway = base_sway + delta_sway
        desired_surge = base_surge + delta_surge
        amp = float(self.cfg.lateral_target_amp_m)
        desired_sway = torch.clamp(desired_sway, -amp * 1.05, amp * 1.05)

        # GT vy feed-forward scales with blend α (0 when fully on image).
        alpha = self._gt_lateral_blend_alpha()
        base_law = str(getattr(self.cfg, "auv_base_law", "position_lead")).lower()
        ff_vy = None
        if base_law != "image_range" and alpha > 1.0e-6:
            ff_vy = self._target_dy_vel_m() * alpha

        pwm = self._compute_auv_pwm(
            desired_sway,
            desired_surge=desired_surge,
            target_vy=ff_vy,
            desired_heave=self._station_origin_w[:, 2],
        )
        sim_dt = float(self.cfg.sim.dt)
        self._auv_dyn.apply_pwm(pwm, sim_dt=sim_dt, step_count=self.episode_length_buf)

    def _apply_action(self) -> None:
        if not bool(getattr(self.cfg, "auv_physics_mode", False)):
            return super()._apply_action()

        self._update_lateral_target_motion()
        apply_target_motion(
            self.beacon,
            self._beacon_pos_w,
            self._beacon_vel_w,
            target_speed_max=1.0e9 if bool(getattr(self.cfg, "lateral_target_moving", False)) else 0.0,
            dt=float(self.cfg.sim.dt * self.cfg.decimation),
            all_env_ids=self.robot._ALL_INDICES,
        )

        sim_dt = float(self.cfg.sim.dt)
        step_count = self.episode_length_buf
        if bool(getattr(self.cfg, "auv_hover_only", True)):
            self._auv_dyn.apply_hover(sim_dt=sim_dt, step_count=step_count)
            return

        if self._auv_residual_mode():
            self._apply_residual_control()
            return

        if self._lateral_vy_override is not None:
            vy = self._lateral_vy_override.view(-1)
            dt = float(self.cfg.sim.dt * self.cfg.decimation)
            if self._lateral_sway_override is not None:
                desired_sway = self._lateral_sway_override.view(-1)
            else:
                cur_sway = self._robot_sway_m()
                desired_sway = cur_sway + vy * dt
        elif self._lateral_sway_override is not None:
            desired_sway = self._lateral_sway_override.view(-1)
        elif bool(getattr(self.cfg, "auv_direct_pwm", False)):
            self._auv_dyn.apply_pwm(self._auv_pwm, sim_dt=sim_dt, step_count=step_count)
            return
        else:
            from .modules.actions import continuous_to_lateral_velocity

            vy = continuous_to_lateral_velocity(
                self.actions,
                max_lateral_velocity=float(self.cfg.max_lateral_velocity),
                num_envs=self.num_envs,
                device=self.device,
            )
            dt = float(self.cfg.sim.dt * self.cfg.decimation)
            desired_sway = self._robot_sway_m() + vy * dt

        desired_surge = (
            self._lateral_surge_override.view(-1)
            if self._lateral_surge_override is not None
            else None
        )
        desired_heave = (
            self._lateral_heave_override.view(-1)
            if self._lateral_heave_override is not None
            else None
        )
        ff_vy = (
            self._auv_ff_vy
            if self._auv_ff_vy_active or float(getattr(self.cfg, "auv_image_p_kff", 0.0)) > 0.0
            else None
        )
        pwm = self._compute_auv_pwm(
            desired_sway,
            desired_surge=desired_surge,
            target_vy=ff_vy,
            desired_heave=desired_heave,
        )
        self._auv_dyn.apply_pwm(pwm, sim_dt=sim_dt, step_count=step_count)

    def _get_rewards(self) -> torch.Tensor:
        if self._auv_residual_mode():
            self._ensure_pipeline()
            range_meas, _, _, range_valid = self.beam_peak_range_meas_m()
            gt_now = self.gt_horiz_range_now_m()
            return compute_auv_residual_reward(
                self.tracker,
                self.actions,
                self._prev_actions,
                range_ref_gt_m=self.range_ref_setpoint_m(),
                range_actual_m=gt_now,
                sway_err_m=self._target_dy_m() - self._robot_sway_m(),
                sway_amp_m=float(self.cfg.lateral_target_amp_m),
                min_depth=float(self.cfg.min_depth),
                max_depth=float(self.cfg.max_depth),
                depth=self.robot.data.root_pos_w[:, 2],
                min_valid_ratio=float(self.cfg.min_valid_ratio),
                w_u=float(getattr(self.cfg, "reward_w_u", 1.0)),
                w_v=float(getattr(self.cfg, "reward_w_v", 0.3)),
                w_range=float(getattr(self.cfg, "reward_w_range", 0.8)),
                w_too_close=float(getattr(self.cfg, "reward_w_too_close", 2.5)),
                w_residual=float(getattr(self.cfg, "reward_w_residual", 0.05)),
                w_lateral_action=float(getattr(self.cfg, "reward_w_lateral_action", 0.0)),
                w_sway=float(getattr(self.cfg, "reward_w_sway", 0.0)),
                w_safety=float(getattr(self.cfg, "reward_w_safety", 10.0)),
                range_accept_m=float(getattr(self.cfg, "range_accept_m", 0.5)),
                sonar_range_meas_m=range_meas,
                sonar_range_valid=range_valid,
                lost_track_steps=self.term_checker.lost_track_steps,
                lost_track_safety_steps=int(getattr(self.cfg, "lost_track_terminate_steps", 80)),
                immediate_lost_safety=bool(getattr(self.cfg, "immediate_lost_terminate", False)),
            )
        return super()._get_rewards()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        if not self._auv_residual_mode():
            return terminated, time_out
        gt_now = self.gt_horiz_range_now_m()
        min_close = float(getattr(self.cfg, "range_accept_m", getattr(self.cfg, "min_close_range_m", 0.5)))
        range_ref = self.range_ref_setpoint_m()
        too_close = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(self.num_envs):
            rng = float(gt_now[i].item())
            ref = float(range_ref[i].item())
            if rng != rng or ref != ref:
                continue
            if rng < ref - min_close:
                too_close[i] = True
        if bool(getattr(self.cfg, "debug_disable_too_close_terminate", False)):
            too_close[:] = False
        return terminated | too_close, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)
        if not bool(getattr(self.cfg, "auv_physics_mode", False)):
            return
        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else self.robot._ALL_INDICES
        n = len(env_ids_t)
        self._invalidate_beam_range_cache()
        self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids_t)
        self._update_range_ref_gt(env_ids_t)
        self._update_beam_prior_azimuth(env_ids_t)
        self._calibrate_range_bias(env_ids_t)
        self._update_range_hold_ref(env_ids_t)
        if self._auv_residual_mode():
            self.actions[env_ids_t] = 0.0
            self._prev_actions[env_ids_t] = 0.0
        if self._auv_decoupled_mode():
            self._auv_dec.reset(env_ids_t)
