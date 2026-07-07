# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Nonar2: 1-DoF forward-looking sonar tracking with OceanSim + belief pipeline."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch
from torch.utils.tensorboard import SummaryWriter

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, sample_uniform, wrap_to_pi

from .modules import (
    BeliefEstimator,
    SonarDetector,
    compute_action_reward,
    compute_action_reward_breakdown,
    preprocess_sonar,
)
from .modules.sonar_beam import azimuth_rad_to_beam, azimuth_rad_to_beam_tensor, beam_center
from .modules.sonar_column_profile import beam_intensity_column_profile
from .modules.sonar_debug import (
    img_stats,
    peak_pixel_from_polar,
    save_sonar_debug_triplet,
)
from .modules.sonar_encoder import SonarFeatureEncoder
from .nonar2_env_cfg import Nonar2EnvCfg
from .oceansim_sonar_sensor import OceanSimSonarSensor


class Nonar2Env(DirectRLEnv):
    cfg: Nonar2EnvCfg

    def __init__(self, cfg: Nonar2EnvCfg, render_mode: str | None = None, **kwargs):
        self.sonar_image = None
        self.sonar_beam_data = None
        super().__init__(cfg, render_mode, **kwargs)

        ocfg = self.cfg.oceansim_sonar_cfg
        self.sonar_image = torch.zeros(self.num_envs, ocfg.height, ocfg.width, device=self.device)
        self.sonar_beam_data = torch.zeros(self.num_envs, ocfg.beam_count, 3, device=self.device)

        self.actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._net_action = torch.zeros(self.num_envs, 1, device=self.device)
        self._exec_wz = torch.zeros(self.num_envs, device=self.device)
        self._robot_yaw = torch.zeros(self.num_envs, device=self.device)
        self.sonar_yaw = torch.zeros(self.num_envs, device=self.device)
        self.out_of_view_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_in_fov = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_facing_quality = torch.ones(self.num_envs, device=self.device)

        self.det_azimuth = torch.zeros(self.num_envs, device=self.device)
        self.det_range = torch.zeros(self.num_envs, device=self.device)
        self.det_conf = torch.zeros(self.num_envs, device=self.device)
        self.det_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.detector = SonarDetector(
            conf_threshold=self.cfg.detection_conf_threshold,
            peak_threshold=float(getattr(self.cfg, "detection_peak_threshold", 0.20)),
            mean_threshold=float(getattr(self.cfg, "detection_mean_threshold", 0.03)),
        )
        self.belief = BeliefEstimator(
            self.num_envs,
            self.device,
            process_noise=float(getattr(self.cfg, "belief_process_noise", 0.01)),
            lost_inflation=float(getattr(self.cfg, "belief_lost_inflation", 1.02)),
            lost_inflation_speed=float(getattr(self.cfg, "belief_lost_inflation_speed", 0.05)),
        )

        ocfg = self.cfg.oceansim_sonar_cfg
        self.sonar_history = torch.zeros(
            self.num_envs,
            max(int(getattr(self.cfg, "sonar_history_len", 1)), 1),
            ocfg.height,
            ocfg.width,
            device=self.device,
        )
        self.sonar_features = torch.zeros(
            self.num_envs, int(getattr(self.cfg, "sonar_feature_dim", 256)), device=self.device
        )
        self.processed_sonar = torch.zeros(self.num_envs, ocfg.height, ocfg.width, device=self.device)

        # Episode evaluation counters
        self._eval_steps = 0
        self._eval_det_steps = 0
        self._eval_align_ok = 0
        self._eval_bearing_err_sum = 0.0
        self._eval_mu_err_sum = 0.0
        self._eval_reward_sum = 0.0
        self._eval_action_sum = 0.0

        # TensorBoard custom scalar logging
        tb_dir = str(getattr(self.cfg, "tb_log_dir", ""))
        if tb_dir:
            os.makedirs(tb_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=tb_dir)
        else:
            self._tb_writer = None

        obs_mode = str(getattr(self.cfg, "policy_obs_mode", "structured")).lower()
        self._uses_cnn_obs = obs_mode == "belief_sac"
        if self._uses_cnn_obs:
            self.feature_encoder = SonarFeatureEncoder(
                height=ocfg.height,
                width=ocfg.width,
                feature_dim=int(getattr(self.cfg, "sonar_feature_dim", 256)),
                history_len=int(getattr(self.cfg, "sonar_history_len", 1)),
            ).to(self.device)
        else:
            self.feature_encoder = None

        self._target_orbit_radius = torch.full(
            (self.num_envs,), float(self.cfg.target_orbit_radius), device=self.device
        )
        self.target_orbit_w = sample_uniform(
            self.cfg.target_orbit_angular_vel_min,
            self.cfg.target_orbit_angular_vel_max,
            (self.num_envs,),
            self.device,
        )
        self._target_orbit_w_ref = self.target_orbit_w.clone()
        self._target_orbit_r_ref = self._target_orbit_radius.clone()
        self.target_orbit_phase = torch.zeros(self.num_envs, device=self.device)
        self._target_cv_vel = torch.zeros(self.num_envs, 2, device=self.device)
        fixed_r = float(getattr(self.cfg, "target_fixed_radius", 3.0))
        self._target_fixed_radius = torch.full(
            (self.num_envs,), fixed_r, device=self.device
        )
        self._target_fixed_omega = torch.zeros(self.num_envs, device=self.device)

        ox, oy, oz = self.cfg.sonar_mount_offset_local
        self._sonar_mount_offset_b = torch.tensor(
            (ox, oy, oz), device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self._pipeline_cache_step: int = -1

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

        self.oceansim_sonar = OceanSimSonarSensor(
            self.cfg.oceansim_sonar_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        for env_idx in range(self.num_envs):
            robot_prim_path = f"/World/envs/env_{env_idx}/Robot"
            sonar_mount_path = f"{robot_prim_path}/SonarMount"
            try:
                from pxr import UsdGeom
                import omni.usd as usd_utils

                stage = usd_utils.get_context().get_stage()
                xform = UsdGeom.Xform.Define(stage, sonar_mount_path)
                ox, oy, oz = self.cfg.sonar_mount_offset_local
                xformable = UsdGeom.Xformable(xform.GetPrim())
                translate_op = None
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                if translate_op is None:
                    translate_op = xformable.AddTranslateOp()
                translate_op.Set((float(ox), float(oy), float(oz)))
            except Exception as exc:
                print(f"Warning: sonar mount fallback to robot root: {exc}")
                sonar_mount_path = robot_prim_path

            self.oceansim_sonar.initialize(env_idx, sonar_mount_path)

        import time

        for warmup_idx in range(4):
            self.sim.step()
            if warmup_idx == 0:
                time.sleep(0.02)

        self._refresh_viewport_camera()

    def _refresh_viewport_camera(self) -> None:
        """Re-frame the main viewport on the workspace (GUI debugging)."""
        controller = getattr(self, "viewport_camera_controller", None)
        if controller is None:
            return
        viewer = self.cfg.viewer
        origin = str(getattr(viewer, "origin_type", "world"))
        try:
            if origin == "asset_root" and getattr(viewer, "asset_name", None):
                controller.update_view_to_asset_root(viewer.asset_name)
            elif origin == "env":
                controller.set_view_env_index(int(getattr(viewer, "env_index", 0)))
                controller.update_view_to_env()
            else:
                controller.update_view_to_world()
            if self.sim.has_gui():
                self.sim.render()
        except Exception as exc:
            print(f"Warning: viewport camera refresh failed: {exc}")

    def _control_dt(self) -> float:
        return float(self.cfg.control_dt)

    def _target_motion_mode(self) -> str:
        return str(getattr(self.cfg, "target_motion_mode", "orbit")).lower()

    def _robot_yaw_limits_rad(self) -> tuple[float, float]:
        lo = math.radians(float(getattr(self.cfg, "motion_yaw_min_deg", -90.0)))
        hi = math.radians(float(getattr(self.cfg, "motion_yaw_max_deg", 90.0)))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    def _target_phase_limits_rad(self) -> tuple[float, float]:
        lo = math.radians(float(getattr(self.cfg, "target_orbit_world_phase_min_deg", -90.0)))
        hi = math.radians(float(getattr(self.cfg, "target_orbit_world_phase_max_deg", 90.0)))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    def _clamp_yaw_to_motion_limits(self, yaw: torch.Tensor) -> torch.Tensor:
        lo, hi = self._robot_yaw_limits_rad()
        return torch.clamp(yaw, lo, hi)

    def _clamp_target_phase(self, phase: torch.Tensor) -> torch.Tensor:
        lo, hi = self._target_phase_limits_rad()
        return torch.clamp(phase, lo, hi)

    def _zero_yaw_rate_at_motion_limits(self, yaw: torch.Tensor, wz: torch.Tensor) -> torch.Tensor:
        """Stop robot angular rate when yaw is already at ±motion_yaw_limit."""
        lo, hi = self._robot_yaw_limits_rad()
        eps = 1.0e-4
        at_hi = (yaw >= hi - eps) & (wz > 0.0)
        at_lo = (yaw <= lo + eps) & (wz < 0.0)
        return torch.where(at_hi | at_lo, torch.zeros_like(wz), wz)

    def _zero_target_w_at_phase_limits(self, phase: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        lo, hi = self._target_phase_limits_rad()
        eps = 1.0e-4
        at_hi = (phase >= hi - eps) & (w > 0.0)
        at_lo = (phase <= lo + eps) & (w < 0.0)
        return torch.where(at_hi | at_lo, torch.zeros_like(w), w)

    def _target_at_orbit_boundary(self) -> torch.Tensor:
        lo, hi = self._target_phase_limits_rad()
        eps = math.radians(float(getattr(self.cfg, "target_orbit_boundary_eps_deg", 2.0)))
        phase = self.target_orbit_phase
        return (phase >= hi - eps) | (phase <= lo + eps)

    def _soften_orbit_w_near_phase_limits(self, phase: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Scale target world orbit rate down near ±phase limits for smooth reversals."""
        margin_deg = float(getattr(self.cfg, "target_orbit_phase_soft_margin_deg", 15.0))
        margin = math.radians(max(margin_deg, 1.0e-3))
        lo, hi = self._target_phase_limits_rad()
        dist = torch.min(phase - lo, hi - phase)
        scale = torch.clamp(dist / margin, 0.0, 1.0)
        w = w * scale
        return self._zero_target_w_at_phase_limits(phase, w)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Discrete actions: 0=left, 1=stay, 2=right."""
        self._net_action = actions.clone()
        dt = max(self._control_dt(), 1.0e-6)
        max_w = float(getattr(self.cfg, "max_angular_velocity", float(self.cfg.max_dtheta) / dt))
        # Map discrete 0/1/2 to angular velocity in rad/s
        if self.cfg.action_space.__class__.__name__ == "Discrete":
            self.actions = torch.zeros(self.num_envs, 1, device=self.device)
            self.actions[:, 0] = torch.where(actions.squeeze(-1) == 0, -max_w, 0.0)
            self.actions[:, 0] = torch.where(actions.squeeze(-1) == 2, max_w, self.actions[:, 0])
        else:
            self.actions = torch.nan_to_num(actions.clone(), nan=0.0, posinf=1.0, neginf=-1.0)
        wz = self.actions[:, 0]
        wz = self._zero_yaw_rate_at_motion_limits(self._robot_yaw, wz)
        self._exec_wz = wz.clone()
        self._robot_yaw = self._clamp_yaw_to_motion_limits(self._robot_yaw + wz * dt)
        self._update_target_motion()

    def _apply_action(self) -> None:
        # Pose-driven yaw (same mechanism as the orbiting target — reliable in GUI/Fabric).
        zeros = torch.zeros(self.num_envs, device=self.device)
        quat = quat_from_euler_xyz(zeros, zeros, self._robot_yaw)
        pos = self.robot.data.root_pos_w.clone()
        pos[:, 0] = self.scene.env_origins[:, 0]
        pos[:, 1] = self.scene.env_origins[:, 1]
        pos[:, 2] = self.scene.env_origins[:, 2] + float(self.cfg.workspace_height)
        self.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        self.robot.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, device=self.device))

    def _ensure_pipeline_for_step(self) -> None:
        if self._pipeline_cache_step == self.common_step_counter:
            return
        self._update_sonar()
        self._update_belief_pipeline()
        self._pipeline_cache_step = self.common_step_counter

    def _update_sonar(self) -> None:
        """Update OceanSim sonar — must call render() even in headless for fresh data."""
        if not bool(getattr(self.oceansim_sonar, "_initialized", False)):
            return
        # Always render: headless physics still needs a render tick for OceanSim to update.
        self.sim.render()
        dt = float(self.cfg.sim.dt * self.cfg.decimation)
        self.oceansim_sonar.update(dt=dt)
        self.sonar_image = self.oceansim_sonar.get_sonar_image()
        self.sonar_beam_data = self.oceansim_sonar.get_beam_data()
        if bool(getattr(self.cfg.oceansim_sonar_cfg, "sonar_initialize_viewport", False)) and self.sim.has_gui():
            try:
                import omni.kit.app

                omni.kit.app.get_app().update()
            except Exception:
                pass

    def _update_belief_pipeline(self) -> None:
        dt = self._control_dt()

        self.sonar_yaw = wrap_to_pi(self._robot_yaw)

        if self._uses_cnn_obs or bool(getattr(self.cfg, "print_sonar_pipeline_debug", False)):
            processed = preprocess_sonar(self.sonar_image)
            self.processed_sonar = processed
        elif bool(getattr(self.cfg, "sonar_preprocess_for_detection", False)):
            processed = preprocess_sonar(self.sonar_image)
            self.processed_sonar = processed
        else:
            processed = None
        if self._uses_cnn_obs:
            self.sonar_history = torch.roll(self.sonar_history, shifts=-1, dims=1)
            self.sonar_history[:, -1] = processed
            if self.feature_encoder is not None:
                hist_len = self.sonar_history.shape[1]
                with torch.no_grad():
                    self.sonar_features = self.feature_encoder(
                        self.sonar_history[:, -hist_len:],
                        self.sonar_yaw.unsqueeze(-1),
                    )

        beam_for_detect = self.sonar_beam_data
        if bool(getattr(self.cfg, "sonar_preprocess_for_detection", False)) and processed is not None:
            self.oceansim_sonar.refresh_beams_from_images(processed)
            beam_for_detect = self.oceansim_sonar.get_beam_data()
            self.sonar_beam_data = beam_for_detect

        prior_azimuth = None
        if bool(getattr(self.cfg, "detection_use_belief_prior", False)):
            mu_body = self.belief.state_in_body(self._sonar_pos_w()[:, :2], self._robot_yaw)["position"]
            mu_norm = torch.linalg.norm(mu_body, dim=-1)
            mu_min = float(getattr(self.cfg, "detection_prior_mu_norm_min", 0.35))
            if bool((mu_norm >= mu_min).any()):
                prior_azimuth = torch.atan2(mu_body[:, 1], mu_body[:, 0])

        ocfg = self.cfg.oceansim_sonar_cfg
        det_common = dict(
            range_min=float(getattr(self.cfg, "detection_range_min", 1.5)),
            range_max=float(getattr(self.cfg, "detection_range_max", 4.5)),
            prior_azimuth=prior_azimuth,
            prior_gate_deg=float(getattr(self.cfg, "detection_prior_gate_deg", 35.0)),
            col_contrast=bool(getattr(self.cfg, "detection_col_contrast", True)),
        )
        if str(getattr(self.cfg, "detection_mode", "beams")).lower() == "image_roi":
            az, rng, conf, detected = self.detector.detect_from_image_roi(
                self.sonar_image,
                hori_fov_deg=float(ocfg.hori_fov),
                min_range=float(ocfg.min_range),
                max_range=float(ocfg.max_range),
                peak_relative_min=float(getattr(self.cfg, "detection_peak_relative_min", 0.35)),
                range_weight_sigma=float(getattr(self.cfg, "detection_range_weight_sigma", 0.0)),
                azimuth_margin_cols=int(getattr(self.cfg, "detection_azimuth_margin_cols", 0)),
                **det_common,
            )
        else:
            az, rng, conf, detected = self.detector.detect_from_beams(
                beam_for_detect,
                **det_common,
            )
        self.det_azimuth = az
        self.det_range = rng
        self.det_conf = conf
        self.det_detected = detected

        self.target_in_fov = self._compute_target_in_sonar_fov()
        # Gate: FOV 外的"检测"是虚检，强制设为 lost（同 /home/lh/1 做法）
        detected_gated = detected & self.target_in_fov
        # 也更新存储的检测状态，让下游（reward/obs）知道真实检测状态
        self.det_detected = detected_gated

        self.belief.predict(dt)
        agent_pos_xy = self.robot.data.root_pos_w[:, :2]
        self.belief.update(
            az,
            rng,
            conf,
            detected_gated,
            agent_pos_xy=agent_pos_xy,
            agent_yaw=self._robot_yaw,
            conf_threshold=self.cfg.detection_conf_threshold,
            dt=dt,
        )

        self.target_facing_quality = self._compute_target_facing_quality()
        self.out_of_view_steps = torch.where(
            self.target_in_fov,
            torch.zeros_like(self.out_of_view_steps),
            self.out_of_view_steps + 1,
        )
        # Per-step eval tracking
        if self._eval_enabled():
            self._update_eval_stats()

    def _eval_enabled(self) -> bool:
        return bool(getattr(self.cfg, "print_eval_debug", False))

    def _reset_eval_stats(self) -> None:
        self._eval_steps = 0
        self._eval_det_steps = 0
        self._eval_align_ok = 0
        self._eval_bearing_err_sum = 0.0
        self._eval_mu_err_sum = 0.0
        self._eval_reward_sum = 0.0
        self._eval_action_sum = 0.0

    def _update_eval_stats(self) -> None:
        env_idx = 0
        det = bool(self.det_detected[env_idx].item())
        self._eval_steps += 1
        if det:
            self._eval_det_steps += 1
            # Bearing error: |det_az - gt_bearing|
            gt_bearing, _ = self._target_body_bearing_and_range()
            err = float(torch.abs(wrap_to_pi(self.det_azimuth[env_idx] - gt_bearing[env_idx])).item())
            self._eval_bearing_err_sum += err
        # Align check
        body = self._belief_body_state()
        mu = body["position"][env_idx]
        mu_norm = float(torch.linalg.norm(mu).item())
        mu_bearing = float(torch.atan2(mu[1], mu[0]).item()) if mu_norm > 0.35 else 0.0
        ref_az = float(self.det_azimuth[env_idx].item()) if det else mu_bearing
        act = float(self.actions[env_idx, 0].item())
        if abs(ref_az) > 0.05 and abs(act) > 0.02:
            if act * ref_az > 0.0:
                self._eval_align_ok += 1
        # Mu error vs GT
        gt_bearing, gt_dist = self._target_body_bearing_and_range()
        gt_x = float((gt_dist[env_idx] * torch.cos(gt_bearing[env_idx])).item())
        gt_y = float((gt_dist[env_idx] * torch.sin(gt_bearing[env_idx])).item())
        mu_err = float(torch.linalg.norm(mu - torch.tensor([gt_x, gt_y], device=mu.device)).item())
        self._eval_mu_err_sum += mu_err
        self._eval_action_sum += abs(act)

    def _print_eval_summary(self, terminated: bool) -> None:
        """Print episode evaluation summary and log to TensorBoard."""
        if not self._eval_enabled():
            return
        s = self._eval_steps
        if s == 0:
            return
        det_rate = self._eval_det_steps / s * 100.0
        align_rate = self._eval_align_ok / max(s, 1)
        mean_bearing_err = math.degrees(self._eval_bearing_err_sum / max(self._eval_det_steps, 1))
        mean_mu_err = self._eval_mu_err_sum / s
        mean_act = self._eval_action_sum / s
        total_reward = self._eval_reward_sum
        tag = "ep_done" if terminated else "timeout"
        step = int(self.common_step_counter)
        # Console print (may be silenced during training)
        if not bool(getattr(self.cfg, "eval_silent", False)):
            print(
                f"[Nonar2 eval] {tag} "
                f"steps={s} det_rate={det_rate:.1f}% align_ok={align_rate:.1%} "
                f"|bearing_err|={mean_bearing_err:.1f}° |mu_err|={mean_mu_err:.2f}m "
                f"mean|action|={mean_act:.3f} total_reward={total_reward:+.2f}",
                flush=True,
            )
        # TensorBoard custom scalars
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("Eval/Episode length", s, step)
            self._tb_writer.add_scalar("Eval/Detection rate", det_rate, step)
            self._tb_writer.add_scalar("Eval/Align rate", align_rate * 100.0, step)
            self._tb_writer.add_scalar("Eval/Mean bearing error (deg)", mean_bearing_err, step)
            self._tb_writer.add_scalar("Eval/Mean mu error (m)", mean_mu_err, step)
            self._tb_writer.add_scalar("Eval/Mean action", mean_act, step)
            self._tb_writer.add_scalar("Eval/Total reward", total_reward, step)
            self._tb_writer.flush()

    def _sonar_pos_w(self) -> torch.Tensor:
        """Sonar mount world position (robot root + body offset)."""
        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w
        offset_w = quat_apply(robot_quat, self._sonar_mount_offset_b)
        return robot_pos + offset_w

    def _target_body_bearing_and_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Target relative to sonar +X in body frame: (bearing rad, horizontal range m)."""
        sensor_pos = self._sonar_pos_w()
        target_pos = self.target.data.root_pos_w
        rel_xy = target_pos[:, :2] - sensor_pos[:, :2]
        dist = torch.norm(rel_xy, dim=-1)

        yaw = self._robot_yaw
        c = torch.cos(-yaw)
        s = torch.sin(-yaw)
        body_x = c * rel_xy[:, 0] - s * rel_xy[:, 1]
        body_y = s * rel_xy[:, 0] + c * rel_xy[:, 1]
        bearing = torch.atan2(body_y, body_x)
        return bearing, dist

    def _compute_target_in_sonar_fov(self) -> torch.Tensor:
        """True when the target ball lies inside sonar horizontal FOV and range (body frame)."""
        bearing, dist = self._target_body_bearing_and_range()

        hori_fov_deg = float(self.cfg.oceansim_sonar_cfg.hori_fov)
        margin_deg = float(getattr(self.cfg, "sonar_fov_margin_deg", 0.0))
        half_fov = math.radians(max(hori_fov_deg * 0.5 - margin_deg, 1.0))
        in_azimuth = torch.abs(wrap_to_pi(bearing)) <= half_fov

        r_min = float(self.cfg.oceansim_sonar_cfg.min_range)
        r_max = float(self.cfg.oceansim_sonar_cfg.max_range)
        in_range = (dist >= r_min) & (dist <= r_max)
        return in_azimuth & in_range

    def _compute_target_facing_quality(self) -> torch.Tensor:
        """Quality in [0, 1]: target ahead of robot +X only (1 at center, 0 beyond bearing cone)."""
        bearing, _ = self._target_body_bearing_and_range()

        max_deg = float(getattr(self.cfg, "tracking_bearing_max_deg", 25.0))
        half_angle = math.radians(max(max_deg, 1.0e-3))
        bearing_norm = torch.abs(wrap_to_pi(bearing)) / half_angle
        return torch.clamp(1.0 - bearing_norm, 0.0, 1.0)

    @staticmethod
    def _quat_to_euler(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (math.pi / 2.0), torch.asin(sinp))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def _build_structured_obs(self) -> torch.Tensor:
        body = self._belief_body_state()
        return torch.cat(
            [
                body["position"],                     # body frame μx, μy
                self.sonar_yaw.unsqueeze(-1),          # θ
                self.det_conf.unsqueeze(-1),           # conf
                self.det_azimuth.unsqueeze(-1),
                torch.clamp(self.det_range, max=float(self.cfg.oceansim_sonar_cfg.max_range)).unsqueeze(-1),
            ],
            dim=-1,
        )

    def _belief_body_state(self) -> dict[str, torch.Tensor]:
        """Belief state (position, velocity) in sonar body frame.

        When detected: use direct detection (azimuth, range) as position —
        avoids EKF CV model's ~15° systematic error on orbit targets.
        When not detected: fallback to EKF μ (episode will end soon anyway).
        """
        body = self.belief.state_in_body(self._sonar_pos_w()[:, :2], self._robot_yaw)
        # Replace position with raw detection when available
        det_pos_x = self.det_range * torch.cos(self.det_azimuth)
        det_pos_y = self.det_range * torch.sin(self.det_azimuth)
        det_pos = torch.stack([det_pos_x, det_pos_y], dim=-1)
        body["position"] = torch.where(
            self.det_detected.unsqueeze(-1), det_pos, body["position"]
        )
        # Velocity from detection: finite-difference or EKF (keep EKF for now)
        return body

    def _sonar_beam_cfg(self) -> tuple[float, int]:
        ocfg = self.cfg.oceansim_sonar_cfg
        return float(ocfg.hori_fov), int(ocfg.beam_count)

    def _build_belief6_obs(self) -> torch.Tensor:
        """1-d: detected beam index [0, beam_count-1] (Module B → strongest beam)."""
        fov_deg, beam_count = self._sonar_beam_cfg()
        beam = azimuth_rad_to_beam_tensor(
            self.det_azimuth,
            hori_fov_deg=fov_deg,
            beam_count=beam_count,
        )
        beam_float = beam.float().unsqueeze(-1)
        return torch.where(self.det_detected.unsqueeze(-1), beam_float, torch.full_like(beam_float, -1.0))

    def _build_column_profile_obs(self) -> torch.Tensor:
        """21-d: subsampled per-beam peak intensity from raw sonar columns."""
        stride = int(getattr(self.cfg, "column_profile_stride", 3))
        col_contrast = bool(getattr(self.cfg, "column_profile_contrast", True))
        return beam_intensity_column_profile(
            self.sonar_beam_data,
            stride=stride,
            col_contrast=col_contrast,
        )

    def _build_belief_sac_obs(self) -> torch.Tensor:
        """6-d state + CNN features (262-d)."""
        body = self._belief_body_state()
        structured = torch.cat(
            [
                body["position"],
                self.sonar_yaw.unsqueeze(-1),
                self.det_conf.unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.cat([structured, self.sonar_features], dim=-1)

    def _sanitize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(obs, nan=0.0, posinf=1.0e3, neginf=-1.0e3)

    def _debug_interval_hit(self) -> bool:
        interval = max(1, int(getattr(self.cfg, "print_entity_positions_interval", 20)))
        return int(self.common_step_counter) % interval == 0

    def _print_entity_positions_debug(self) -> None:
        """Print env0 robot/target world positions during training."""
        if not bool(getattr(self.cfg, "print_entity_positions", False)) or not self._debug_interval_hit():
            return
        step = int(self.common_step_counter)
        robot_p = self.robot.data.root_pos_w[0]
        target_p = self.target.data.root_pos_w[0]
        cmd_yaw_deg = float(torch.rad2deg(self._robot_yaw[0]).item())
        sim_yaw_deg = float(torch.rad2deg(self.sonar_yaw[0]).item())
        action = float(self.actions[0, 0].item())
        dist_xy = float(torch.norm(target_p[:2] - robot_p[:2]).item())
        if self._target_motion_mode() == "constant_velocity":
            cv = self._target_cv_vel[0]
            cv_spd = float(torch.linalg.norm(cv).item())
            cv_hdg_deg = float(torch.rad2deg(torch.atan2(cv[1], cv[0])).item())
            motion_tag = f"cv_spd={cv_spd:.3f}m/s cv_hdg={cv_hdg_deg:+.1f}°"
        elif self._target_motion_mode() == "fixed_radius_cv":
            r = float(self._target_fixed_radius[0].item())
            w = float(self._target_fixed_omega[0].item())
            phase_deg = float(torch.rad2deg(self.target_orbit_phase[0]).item())
            tangential = abs(w * r)
            motion_tag = (
                f"fixed_R={r:.3f}m ω={w:+.3f}rad/s v_t={tangential:.3f}m/s phase={phase_deg:+.1f}°"
            )
        else:
            orbit_r = float(self._target_orbit_radius[0].item())
            orbit_phase_deg = float(torch.rad2deg(self.target_orbit_phase[0]).item())
            motion_tag = f"orbit_R={orbit_r:.3f}m world_phase_deg={orbit_phase_deg:+.1f}"
        print(
            f"[Nonar2] step={step} "
            f"robot=({robot_p[0]:+.3f},{robot_p[1]:+.3f},{robot_p[2]:+.3f}) "
            f"target=({target_p[0]:+.3f},{target_p[1]:+.3f},{target_p[2]:+.3f}) "
            f"{motion_tag} dist_xy={dist_xy:.3f}m "
            f"cmd_yaw_deg={cmd_yaw_deg:+.1f} sim_yaw_deg={sim_yaw_deg:+.1f} action={action:+.3f}",
            flush=True,
        )

    def _sonar_pipeline_peak_beam(self, env_idx: int = 0) -> tuple[int, float, float, float]:
        """Peak column/beam index and polar coords (matches active SonarDetector mode)."""
        ocfg = self.cfg.oceansim_sonar_cfg
        r_min = float(getattr(self.cfg, "detection_range_min", 1.5))
        r_max = float(getattr(self.cfg, "detection_range_max", 4.5))

        if str(getattr(self.cfg, "detection_mode", "beams")).lower() == "image_roi":
            raw = self.sonar_image[env_idx]
            height, width = raw.shape
            span = max(float(ocfg.max_range) - float(ocfg.min_range), 1.0e-6)
            row_min = int(round((r_min - float(ocfg.min_range)) / span * max(height - 1, 0)))
            row_max = int(round((r_max - float(ocfg.min_range)) / span * max(height - 1, 0)))
            row_min = max(0, min(height - 1, row_min))
            row_max = max(row_min, min(height - 1, row_max))
            roi = raw[row_min : row_max + 1, :]
            col_peak_vals, col_peak_rows = roi.max(dim=0)
            scores = col_peak_vals
            sigma = float(getattr(self.cfg, "detection_range_weight_sigma", 0.0))
            if sigma > 0.0:
                r_center = 0.5 * (r_min + r_max)
                range_m = float(ocfg.min_range) + (
                    (col_peak_rows.float() + float(row_min)) / max(height - 1, 1)
                ) * span
                range_w = torch.exp(-0.5 * ((range_m - r_center) / sigma) ** 2)
                scores = col_peak_vals * range_w
            if bool(getattr(self.cfg, "detection_col_contrast", True)):
                scores = scores - scores.median()
            margin = max(0, int(getattr(self.cfg, "detection_azimuth_margin_cols", 0)))
            if margin > 0 and scores.shape[0] > 2 * margin:
                scores = scores.clone()
                scores[:margin] = -1.0e6
                scores[-margin:] = -1.0e6
            peak_col = int(torch.argmax(scores).item())
            peak_row_local = int(col_peak_rows[peak_col].item())
            peak_val = float(col_peak_vals[peak_col].item())
            fov_rad = math.radians(float(ocfg.hori_fov))
            peak_az = float(-((peak_col - (width - 1) / 2.0) / max(width - 1, 1)) * fov_rad)
            peak_rng = float(ocfg.min_range) + ((peak_row_local + row_min) / max(height - 1, 1)) * span
            return peak_col, peak_az, peak_rng, peak_val

        beam = self.sonar_beam_data[env_idx]
        intensity = beam[:, 2].clone()
        ranges = beam[:, 1]
        r_min = float(getattr(self.cfg, "detection_range_min", 1.5))
        r_max = float(getattr(self.cfg, "detection_range_max", 4.5))
        in_band = (ranges >= r_min) & (ranges <= r_max)
        intensity = torch.where(in_band, intensity, torch.full_like(intensity, -1.0e6))
        peak_beam = int(torch.argmax(intensity).item())
        peak_az = float(beam[peak_beam, 0].item())
        peak_rng = float(beam[peak_beam, 1].item())
        peak_val = float(intensity[peak_beam].item())
        return peak_beam, peak_az, peak_rng, peak_val

    def _print_sonar_pipeline_debug(self) -> None:
        """Print/save raw sonar, preprocessed sonar, and beam peak (Belief6 / play)."""
        if not bool(getattr(self.cfg, "print_sonar_pipeline_debug", False)) or not self._debug_interval_hit():
            return

        step = int(self.common_step_counter)
        env_idx = 0
        raw = self.sonar_image[env_idx] if self.sonar_image is not None else None
        if raw is None:
            return

        proc = self.processed_sonar[env_idx]

        raw_m, raw_x, raw_nz = img_stats(raw)
        proc_m, proc_x, proc_nz = img_stats(proc)

        peak_beam, peak_az, peak_rng, peak_val = self._sonar_pipeline_peak_beam(env_idx)
        det = int(self.det_detected[env_idx].item())
        ocfg = self.cfg.oceansim_sonar_cfg
        body_bearing, body_dist = self._target_body_bearing_and_range()

        peak_row, peak_col = None, None
        gt_row, gt_col = peak_pixel_from_polar(
            float(body_bearing[env_idx].item()),
            float(body_dist[env_idx].item()),
            height=int(ocfg.height),
            width=int(ocfg.width),
            hori_fov_deg=float(ocfg.hori_fov),
            min_range=float(ocfg.min_range),
            max_range=float(ocfg.max_range),
        )

        if det and torch.isfinite(self.det_range[env_idx]):
            peak_row, peak_col = peak_pixel_from_polar(
                float(self.det_azimuth[env_idx].item()),
                float(self.det_range[env_idx].item()),
                height=int(ocfg.height),
                width=int(ocfg.width),
                hori_fov_deg=float(ocfg.hori_fov),
                min_range=float(ocfg.min_range),
                max_range=float(ocfg.max_range),
            )
        elif peak_val >= float(getattr(self.cfg, "detection_peak_threshold", 0.20)):
            peak_row, peak_col = peak_pixel_from_polar(
                peak_az,
                peak_rng,
                height=int(ocfg.height),
                width=int(ocfg.width),
                hori_fov_deg=float(ocfg.hori_fov),
                min_range=float(ocfg.min_range),
                max_range=float(ocfg.max_range),
            )

        gt_bearing_deg = float(torch.rad2deg(body_bearing[env_idx]).item())
        ref_az_deg = float(torch.rad2deg(self.det_azimuth[env_idx]).item()) if det else math.degrees(peak_az)
        bearing_err_deg = ref_az_deg - gt_bearing_deg
        if bearing_err_deg > 180.0:
            bearing_err_deg -= 360.0
        elif bearing_err_deg < -180.0:
            bearing_err_deg += 360.0

        header = (
            f"[Nonar2 sonar pipeline] step={step} "
            f"raw(H={raw.shape[0]},W={raw.shape[1]}, mean={raw_m:.4f}, max={raw_x:.4f}, nz={raw_nz:.1%}) "
            f"proc(mean={proc_m:.4f}, max={proc_x:.4f}, nz={proc_nz:.1%}) "
            f"peak_beam={peak_beam} peak_az_deg={math.degrees(peak_az):+.2f} "
            f"peak_rng={peak_rng:.3f}m peak_int={peak_val:.4f} "
        )
        if det:
            tail = (
                f"det={det} det_az_deg={float(torch.rad2deg(self.det_azimuth[env_idx])):+.2f} "
                f"det_rng={float(self.det_range[env_idx]):.3f}m conf={float(self.det_conf[env_idx]):.4f} "
                f"pix_peak=({peak_row},{peak_col}) gt_bearing_deg={gt_bearing_deg:+.2f} "
                f"bearing_err_deg={bearing_err_deg:+.2f} dist={float(body_dist[env_idx]):.3f}m "
                f"rng_err_m={float(self.det_range[env_idx]) - float(body_dist[env_idx]):+.3f}"
            )
        else:
            thr_abs = float(getattr(self.cfg, "detection_peak_threshold", 0.2))
            thr_rel = float(getattr(self.cfg, "detection_peak_relative_min", 0.35))
            tail = (
                f"det={det} (peak_int={peak_val:.4f} thr_abs={thr_abs:.2f} thr_rel={thr_rel:.0%}×roi_max) "
                f"pix_peak=({peak_row},{peak_col}) gt_bearing_deg={gt_bearing_deg:+.2f} "
                f"peak_vs_gt_az_err={math.degrees(peak_az) - gt_bearing_deg:+.2f} "
                f"dist={float(body_dist[env_idx]):.3f}m peak_rng={peak_rng:.3f}m "
                f"rng_err_m={peak_rng - float(body_dist[env_idx]):+.3f}"
            )
        print(header + tail, flush=True)

        if bool(getattr(self.cfg, "sonar_debug_save_images", False)):
            save_sonar_debug_triplet(
                raw=raw,
                processed=proc,
                peak_row=peak_row,
                peak_col=peak_col,
                gt_row=gt_row,
                gt_col=gt_col,
                save_dir=str(getattr(self.cfg, "sonar_debug_save_dir", "logs/nonar2_sonar_debug")),
                step=step,
                env_idx=env_idx,
            )

    def _print_sonar_obs_debug(self) -> None:
        """Verify sonar image / CNN features reach policy observation (belief_sac 262-d)."""
        if not bool(getattr(self.cfg, "print_sonar_obs_debug", False)) or not self._debug_interval_hit():
            return
        if str(getattr(self.cfg, "policy_obs_mode", "structured")).lower() != "belief_sac":
            return

        step = int(self.common_step_counter)
        raw = self.sonar_image[0] if self.sonar_image is not None else None
        proc = self.processed_sonar[0]
        feat = self.sonar_features[0]

        def _img_stats(t: torch.Tensor) -> tuple[float, float, float]:
            nz = (t > 1.0e-4).float().mean().item()
            return float(t.mean().item()), float(t.max().item()), nz

        if raw is not None:
            raw_m, raw_x, raw_nz = _img_stats(raw)
        else:
            raw_m, raw_x, raw_nz = 0.0, 0.0, 0.0
        proc_m, proc_x, proc_nz = _img_stats(proc)
        feat_norm = float(torch.norm(feat).item())
        feat_m = float(feat.mean().item())
        feat_x = float(feat.max().item())

        obs = self._build_belief_sac_obs()[0]
        obs_norm = float(torch.norm(obs).item())
        struct6 = obs[:6]
        feat_tail = obs[6:]

        det = int(self.det_detected[0].item())
        print(
            f"[Nonar2 sonar] step={step} "
            f"raw(mean={raw_m:.3f},max={raw_x:.3f},nz={raw_nz:.1%}) "
            f"proc(mean={proc_m:.3f},max={proc_x:.3f},nz={proc_nz:.1%}) "
            f"cnn(norm={feat_norm:.2f},mean={feat_m:.4f},max={feat_x:.3f}) "
            f"det={det} conf={float(self.det_conf[0]):.3f} "
            f"az_deg={float(torch.rad2deg(self.det_azimuth[0])):+.1f} "
            f"rng={float(self.det_range[0]):.2f}m | "
            f"obs262(norm={obs_norm:.1f}) struct6={struct6.tolist()} "
            f"feat256(norm={float(torch.norm(feat_tail)):.2f})",
            flush=True,
        )
        if raw_x < 1.0e-4 and proc_x < 1.0e-4:
            print(
                f"[Nonar2 sonar] WARNING step={step}: sonar image nearly empty — "
                "policy CNN input may be invalid.",
                flush=True,
            )

    def _print_belief_pipeline_debug(self) -> None:
        """Print EKF state vs detection and GT (Module C health check)."""
        if not bool(getattr(self.cfg, "print_belief_pipeline_debug", False)) or not self._debug_interval_hit():
            return

        env_idx = 0
        step = int(self.common_step_counter)
        body = self._belief_body_state()
        mu = body["position"][env_idx]
        sig = self.belief.uncertainty_xy[env_idx]
        vel = self.belief.velocity_xy[env_idx]
        mu_norm = float(torch.linalg.norm(mu).item())
        mu_brg_deg = float(torch.rad2deg(torch.atan2(mu[1], mu[0])).item()) if mu_norm > 1.0e-4 else 0.0
        mu_rng = mu_norm

        gt_bearing, gt_dist = self._target_body_bearing_and_range()
        gt_brg_deg = float(torch.rad2deg(gt_bearing[env_idx]).item())
        gt_x = float((gt_dist[env_idx] * torch.cos(gt_bearing[env_idx])).item())
        gt_y = float((gt_dist[env_idx] * torch.sin(gt_bearing[env_idx])).item())
        mu_err_m = float(torch.linalg.norm(mu - torch.tensor([gt_x, gt_y], device=mu.device)).item())

        det = int(self.det_detected[env_idx].item())
        det_az_deg = float(torch.rad2deg(self.det_azimuth[env_idx]).item()) if det else 0.0
        det_rng = float(self.det_range[env_idx].item()) if det and torch.isfinite(self.det_range[env_idx]) else float("nan")
        residual_az_deg = 0.0
        residual_rng_m = 0.0
        if det and mu_norm > 1.0e-4:
            residual_az_deg = float(
                torch.rad2deg(wrap_to_pi(self.det_azimuth[env_idx] - torch.atan2(mu[1], mu[0]))).item()
            )
            residual_rng_m = det_rng - mu_rng
        elif det:
            residual_az_deg = det_az_deg
            residual_rng_m = det_rng

        sigma_trace = float(sig.sum().item())
        cov_det = float(self.belief.position_cov_det[env_idx].item())
        logdet = -math.log(max(cov_det, 1e-30))
        p00 = float(self.belief.p[env_idx, 0, 0].item())
        p01 = float(self.belief.p[env_idx, 0, 1].item())
        p11 = float(self.belief.p[env_idx, 1, 1].item())
        lost = sigma_trace > float(getattr(self.cfg, "belief_lost_threshold", 2.0))
        speed = float(torch.linalg.norm(vel).item())

        print(
            f"[Nonar2 belief/EKF] step={step} det={det} "
            f"μ=({float(mu[0]):+.3f},{float(mu[1]):+.3f}) μ_brg_deg={mu_brg_deg:+.2f} μ_rng={mu_rng:.3f}m "
            f"σ²=({float(sig[0]):.4f},{float(sig[1]):.4f}) det(Σ)={cov_det:.2e} -logdet={logdet:.2f} "
            f"trace={sigma_trace:.4f} lost={int(lost)} |v|={speed:.3f} "
            f"gt=({gt_x:+.3f},{gt_y:+.3f}) gt_brg_deg={gt_brg_deg:+.2f} dist={float(gt_dist[env_idx]):.3f}m "
            f"μ_err_m={mu_err_m:.3f}",
            flush=True,
        )
        # Also print P matrix diagonals and correlation every 50 steps
        if step % 50 == 0:
            print(
                f"[Nonar2 belief/EKF P] step={step} "
                f"P_xx={p00:.4f} P_xy={p01:.4f} P_yy={p11:.4f} "
                f"P_vx={float(self.belief.p[env_idx,2,2].item()):.4f} P_vy={float(self.belief.p[env_idx,3,3].item()):.4f}",
                flush=True,
            )

    def _print_play_step_debug(self) -> None:
        """Compact play log: policy net output vs executed yaw rate."""
        if not bool(getattr(self.cfg, "print_play_debug", False)) or not self._debug_interval_hit():
            return

        env_idx = 0
        step = int(self.common_step_counter)
        gt_bearing, _ = self._target_body_bearing_and_range()
        gt_brg_deg = float(torch.rad2deg(gt_bearing[env_idx]).item())

        mu = self._belief_body_state()["position"][env_idx]
        mu_norm = float(torch.linalg.norm(mu).item())
        mu_brg_deg = float(torch.rad2deg(torch.atan2(mu[1], mu[0])).item()) if mu_norm > 1.0e-4 else float("nan")

        det = int(self.det_detected[env_idx].item())
        det_az_deg = float(torch.rad2deg(self.det_azimuth[env_idx]).item()) if det else float("nan")

        net_a = float(self._net_action[env_idx, 0].item())
        exec_a = float(torch.clamp(self._net_action[env_idx, 0], -1.0, 1.0).item())
        wz = float(self._exec_wz[env_idx].item())
        theta_deg = float(torch.rad2deg(self.sonar_yaw[env_idx]).item())

        ref_az = float(self.det_azimuth[env_idx].item()) if det else (
            float(torch.atan2(mu[1], mu[0]).item()) if mu_norm >= float(getattr(self.cfg, "bearing_align_mu_norm_min", 0.35)) else 0.0
        )
        if abs(ref_az) > 0.05 and abs(net_a) > 0.05:
            turn_ok = (net_a * ref_az) > 0.0
            turn_tag = "ok" if turn_ok else "WRONG"
        else:
            turn_tag = "-"

        print(
            f"[Nonar2 play] step={step} det={det} fov={bool(self.target_in_fov[env_idx].item())} "
            f"gt={gt_brg_deg:+.1f}° det_az={det_az_deg:+.1f}° μ_brg={mu_brg_deg:+.1f}° "
            f"net={net_a:+.3f} exec={exec_a:+.3f} ω={wz:+.3f}rad/s θ={theta_deg:+.1f}° turn={turn_tag}",
            flush=True,
        )

    def _print_policy_obs_debug(self) -> None:
        """Print policy observation vector (Belief6 7-d or structured 10-d)."""
        if not bool(getattr(self.cfg, "print_policy_obs_debug", False)) or not self._debug_interval_hit():
            return

        mode = str(getattr(self.cfg, "policy_obs_mode", "structured")).lower()
        if mode == "belief_sac":
            return

        step = int(self.common_step_counter)
        env_idx = 0
        if mode == "belief6":
            obs = self._build_belief6_obs()[env_idx]
            obs_list = [float(v) for v in obs.tolist()]
            # GT beam
            gt_bearing, _ = self._target_body_bearing_and_range()
            fov_deg, beam_count = self._sonar_beam_cfg()
            gt_beam = azimuth_rad_to_beam(
                float(gt_bearing[env_idx].item()),
                hori_fov_deg=fov_deg,
                beam_count=beam_count,
            )

            det = int(self.det_detected[env_idx].item())
            det_beam = self._sonar_detected_beam(env_idx) if det else None

            print(
                f"[Nonar2 obs] step={step} det={det} "
                f"gt_beam={gt_beam} det_beam={det_beam if det_beam is not None else 'X'} "
                f"obs_beam={obs_list[0]:+.1f}",
                flush=True,
            )
        elif mode == "column_profile":
            obs = self._build_column_profile_obs()[env_idx]
            obs_list = [float(v) for v in obs.tolist()]
            stride = int(getattr(self.cfg, "column_profile_stride", 3))
            peak_bin = int(torch.argmax(obs).item())
            peak_beam = peak_bin * stride
            gt_bearing, _ = self._target_body_bearing_and_range()
            fov_deg, beam_count = self._sonar_beam_cfg()
            gt_beam = azimuth_rad_to_beam(
                float(gt_bearing[env_idx].item()),
                hori_fov_deg=fov_deg,
                beam_count=beam_count,
            )
            det = int(self.det_detected[env_idx].item())
            det_beam = self._sonar_detected_beam(env_idx) if det else None
            print(
                f"[Nonar2 obs] step={step} dim={len(obs_list)} "
                f"peak_bin={peak_bin} peak_beam={peak_beam} "
                f"gt_beam={gt_beam} det_beam={det_beam if det_beam is not None else 'X'} "
                f"profile_max={max(obs_list):+.3f} profile_min={min(obs_list):+.3f}",
                flush=True,
            )
        elif mode == "structured":
            obs = self._build_structured_obs()[env_idx]
            print(
                f"[Nonar2 policy obs] step={step} dim={obs.numel()} structured10={obs.tolist()}",
                flush=True,
            )

    def _action_reward_kwargs(self) -> dict:
        return dict(
            action=self.actions,
            detected=self.det_detected,
            det_azimuth=self.det_azimuth,
            r_detect=float(getattr(self.cfg, "r_detect", 5.0)),
            r_deviation_k=float(getattr(self.cfg, "r_deviation_k", 0.15)),
            r_miss=float(getattr(self.cfg, "r_miss", -5.0)),
            r_smooth=float(getattr(self.cfg, "r_smooth", 0.010)),
        )

    def _compute_discrete_reward(self) -> torch.Tensor:
        """Center-seeking reward: high when GT target is near boresight (beam 30 / 0°)."""
        r_max = float(getattr(self.cfg, "discrete_reward_center", 2.0))
        k = float(getattr(self.cfg, "discrete_reward_center_k", 0.12))
        r_min = float(getattr(self.cfg, "discrete_reward_min", -0.5))
        r_miss = float(getattr(self.cfg, "discrete_reward_miss", -1.0))

        gt_bearing, _ = self._target_body_bearing_and_range()
        err_deg = torch.rad2deg(torch.abs(gt_bearing))

        reward = r_max - k * err_deg
        reward = torch.clamp(reward, min=r_min, max=r_max)
        reward = torch.where(self.det_detected, reward, torch.full_like(reward, r_miss))
        return reward

    def _sonar_detected_beam(self, env_idx: int) -> int | None:
        """Return detected beam index for env_idx, or None."""
        if not self.det_detected[env_idx]:
            return None
        fov_deg, beam_count = self._sonar_beam_cfg()
        return azimuth_rad_to_beam(
            float(self.det_azimuth[env_idx].item()),
            hori_fov_deg=fov_deg,
            beam_count=beam_count,
        )

    def _get_rewards(self) -> torch.Tensor:
        self._ensure_pipeline_for_step()
        if self.cfg.action_space.__class__.__name__ == "Discrete":
            reward = self._compute_discrete_reward()
        else:
            self._print_reward_debug()
            reward = compute_action_reward(**self._action_reward_kwargs())
            scale = float(getattr(self.cfg, "reward_scale", 1.0))
            reward = reward * scale
            clip_lo = float(getattr(self.cfg, "reward_clip_min", -5.0))
            clip_hi = float(getattr(self.cfg, "reward_clip_max", 5.0))
            reward = torch.clamp(reward, min=clip_lo, max=clip_hi)
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        if self._eval_enabled():
            self._eval_reward_sum += float(reward[0].item())
        return reward

    def _get_observations(self) -> dict:
        self._ensure_pipeline_for_step()
        self._print_play_step_debug()
        self._print_entity_positions_debug()
        self._print_sonar_pipeline_debug()
        self._print_belief_pipeline_debug()
        self._print_sonar_obs_debug()
        self._print_policy_obs_debug()
        mode = str(getattr(self.cfg, "policy_obs_mode", "structured")).lower()
        if mode == "belief_sac":
            obs = self._build_belief_sac_obs()
        elif mode == "belief6":
            obs = self._build_belief6_obs()
        elif mode == "column_profile":
            obs = self._build_column_profile_obs()
        else:
            obs = self._build_structured_obs()
        return {"policy": self._sanitize_obs(obs)}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_pipeline_for_step()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if bool(getattr(self.cfg, "terminate_on_target_out_of_fov", True)):
            grace = int(getattr(self.cfg, "target_out_of_fov_grace_steps", 0))
            if grace > 0:
                lost = self.out_of_view_steps >= grace
            else:
                lost = ~self.target_in_fov
        else:
            lost = self.out_of_view_steps >= int(self.cfg.max_out_of_view_steps)
        terminated = lost
        if bool(getattr(self.cfg, "terminate_on_target_orbit_boundary", False)):
            terminated = terminated | self._target_at_orbit_boundary()
        # Episode end eval summary (only print on actual termination, not timeout)
        if terminated.any():
            self._print_eval_summary(terminated=True)
            self._reset_eval_stats()
        elif time_out.any():
            self._print_eval_summary(terminated=False)
            self._reset_eval_stats()
        return terminated, time_out

    def _orbit_radius_bounds(self) -> tuple[float, float]:
        r_lo = float(getattr(self.cfg, "target_orbit_radius_min", 2.0))
        r_hi = float(getattr(self.cfg, "target_orbit_radius_max", 4.0))
        if r_lo > r_hi:
            r_lo, r_hi = r_hi, r_lo
        return r_lo, r_hi

    def _update_target_orbit_radius(self) -> torch.Tensor:
        """Smooth orbit-radius tracking toward a slowly drifting reference."""
        r_lo, r_hi = self._orbit_radius_bounds()
        r_prev = self._target_orbit_radius.clone()
        dt = max(self._control_dt(), 1.0e-6)

        noise_std = max(float(getattr(self.cfg, "target_orbit_radius_noise_std", 0.0)), 0.0)
        if noise_std > 0.0:
            self._target_orbit_r_ref = self._target_orbit_r_ref + torch.randn_like(
                self._target_orbit_r_ref
            ) * noise_std
        self._target_orbit_r_ref = torch.clamp(self._target_orbit_r_ref, r_lo, r_hi)

        alpha = float(getattr(self.cfg, "target_orbit_r_smooth_alpha", 0.10))
        alpha = min(max(alpha, 0.0), 1.0)
        r_goal = (1.0 - alpha) * self._target_orbit_radius + alpha * self._target_orbit_r_ref

        max_speed = max(float(getattr(self.cfg, "target_orbit_r_speed_max", 0.12)), 0.0)
        dr = torch.clamp(r_goal - self._target_orbit_radius, -max_speed * dt, max_speed * dt)
        self._target_orbit_radius = torch.clamp(self._target_orbit_radius + dr, r_lo, r_hi)
        return dr / dt

    def _update_target_orbit_angular_rate(self) -> None:
        """Smooth orbit angular rate toward a slowly drifting reference."""
        w_min = float(self.cfg.target_orbit_angular_vel_min)
        w_max = float(self.cfg.target_orbit_angular_vel_max)
        dt = max(self._control_dt(), 1.0e-6)

        noise_std = max(float(getattr(self.cfg, "target_orbit_angular_noise_std", 0.0)), 0.0)
        if noise_std > 0.0:
            self._target_orbit_w_ref = self._target_orbit_w_ref + torch.randn_like(
                self._target_orbit_w_ref
            ) * noise_std
        self._target_orbit_w_ref = torch.clamp(self._target_orbit_w_ref, w_min, w_max)

        alpha = float(getattr(self.cfg, "target_orbit_w_smooth_alpha", 0.12))
        alpha = min(max(alpha, 0.0), 1.0)
        w_goal = (1.0 - alpha) * self.target_orbit_w + alpha * self._target_orbit_w_ref

        max_accel = max(float(getattr(self.cfg, "target_orbit_w_accel_max", 0.15)), 0.0)
        dw = torch.clamp(w_goal - self.target_orbit_w, -max_accel * dt, max_accel * dt)
        self.target_orbit_w = torch.clamp(self.target_orbit_w + dw, w_min, w_max)

    def _robot_command_quat(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Quaternion from commanded yaw (_robot_yaw), aligned with pose control."""
        if env_ids is None:
            yaw = self._robot_yaw
        else:
            yaw = self._robot_yaw[env_ids]
        n = yaw.shape[0]
        zeros = torch.zeros(n, device=self.device, dtype=yaw.dtype)
        return quat_from_euler_xyz(zeros, zeros, yaw)

    def _world_orbit_position(
        self,
        radius: torch.Tensor,
        phase: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Target in world XY: env_origin + [r cos φ, r sin φ, 0]; φ is world azimuth (rad)."""
        origins = self.scene.env_origins if env_ids is None else self.scene.env_origins[env_ids]
        z = origins[:, 2] + float(self.cfg.workspace_height)
        c = torch.cos(phase)
        s = torch.sin(phase)
        return torch.stack(
            [
                origins[:, 0] + radius * c,
                origins[:, 1] + radius * s,
                z,
            ],
            dim=-1,
        )

    def _body_offset_to_world(
        self,
        radius: torch.Tensor,
        phase: torch.Tensor,
        quat: torch.Tensor,
        origin: torch.Tensor,
    ) -> torch.Tensor:
        """Target position: origin + R_world * [r cos φ, r sin φ, 0] (body-frame orbit)."""
        c = torch.cos(phase)
        s = torch.sin(phase)
        offset_local = torch.stack([radius * c, radius * s, torch.zeros_like(radius)], dim=-1)
        offset_world = quat_apply(quat, offset_local)
        pos = origin + offset_world
        pos[:, 2] = origin[:, 2]
        return pos

    def _update_target_motion(self) -> None:
        mode = self._target_motion_mode()
        if mode == "constant_velocity":
            self._update_target_constant_velocity()
        elif mode == "fixed_radius_cv":
            self._update_target_fixed_radius_cv()
        else:
            self._update_target_orbit_motion()

    def _update_target_fixed_radius_cv(self) -> None:
        """Fixed-radius circular arc with constant tangential speed; ω flips at phase limits."""
        dt = self._control_dt()
        lo, hi = self._target_phase_limits_rad()
        w = self._target_fixed_omega
        phase = self.target_orbit_phase + w * dt

        hit_lo = phase < lo
        hit_hi = phase > hi
        phase = torch.where(hit_lo, 2.0 * lo - phase, phase)
        phase = torch.where(hit_hi, 2.0 * hi - phase, phase)
        w = torch.where(hit_lo | hit_hi, -w, w)

        self.target_orbit_phase = phase
        self._target_fixed_omega = w

        r = self._target_fixed_radius
        new_pos = self._world_orbit_position(r, phase)
        c = torch.cos(phase)
        s = torch.sin(phase)
        vel_xy = (r * w).unsqueeze(-1) * torch.stack([-s, c], dim=-1)
        lin_vel = torch.cat(
            [vel_xy, torch.zeros(self.num_envs, 1, device=self.device)],
            dim=-1,
        )
        target_quat = self.target.data.root_quat_w
        self.target.write_root_pose_to_sim(torch.cat([new_pos, target_quat], dim=-1))
        ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.target.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1))

    def _update_target_constant_velocity(self) -> None:
        """World XY uniform linear motion; reflect velocity at annulus bounds."""
        dt = self._control_dt()
        origins = self.scene.env_origins
        pos_w = self.target.data.root_pos_w
        z = origins[:, 2] + float(self.cfg.workspace_height)

        rel = pos_w[:, :2] - origins[:, :2]
        vel = self._target_cv_vel.clone()
        rel_new = rel + vel * dt

        r_lo = float(getattr(self.cfg, "target_cv_radius_min", 1.8))
        r_hi = float(getattr(self.cfg, "target_cv_radius_max", 4.5))
        r = torch.linalg.norm(rel_new, dim=-1, keepdim=True).clamp(min=1.0e-6)
        n = rel_new / r

        outer = (r.squeeze(-1) > r_hi).unsqueeze(-1)
        rel_new = torch.where(outer, n * r_hi, rel_new)
        vn = (vel * n).sum(dim=-1, keepdim=True)
        vel = torch.where(outer, vel - 2.0 * vn * n, vel)

        r = torch.linalg.norm(rel_new, dim=-1, keepdim=True).clamp(min=1.0e-6)
        n = rel_new / r
        inner = (r.squeeze(-1) < r_lo).unsqueeze(-1)
        rel_new = torch.where(inner, n * r_lo, rel_new)
        vn = (vel * n).sum(dim=-1, keepdim=True)
        vel = torch.where(inner, vel - 2.0 * vn * n, vel)

        self._target_cv_vel = vel
        new_pos = torch.cat([origins[:, :2] + rel_new, z.unsqueeze(-1)], dim=-1)
        lin_vel = torch.cat([vel, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        target_quat = self.target.data.root_quat_w
        self.target.write_root_pose_to_sim(torch.cat([new_pos, target_quat], dim=-1))
        ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.target.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1))

    def _update_target_orbit_motion(self) -> None:
        dt = self._control_dt()
        self._update_target_orbit_angular_rate()
        self.target_orbit_w = self._soften_orbit_w_near_phase_limits(self.target_orbit_phase, self.target_orbit_w)
        self.target_orbit_phase = self._clamp_target_phase(
            self.target_orbit_phase + self.target_orbit_w * dt
        )

        r_dot = self._update_target_orbit_radius()
        r = self._target_orbit_radius
        phase = self.target_orbit_phase
        c = torch.cos(phase)
        s = torch.sin(phase)
        frame = str(getattr(self.cfg, "target_orbit_frame", "world")).lower()

        if frame == "world":
            new_pos = self._world_orbit_position(r, phase)
            vel_xy = r_dot.unsqueeze(-1) * torch.stack([c, s], dim=-1) + (
                (r * self.target_orbit_w).unsqueeze(-1) * torch.stack([-s, c], dim=-1)
            )
            lin_vel = torch.cat(
                [vel_xy, torch.zeros(self.num_envs, 1, device=self.device)],
                dim=-1,
            )
        else:
            quat = self._robot_command_quat()
            robot_pos = self.robot.data.root_pos_w
            new_pos = self._body_offset_to_world(r, phase, quat, robot_pos)
            u_rad = torch.stack([c, s], dim=-1)
            u_tan = torch.stack([-s, c], dim=-1)
            vel_body_xy = r_dot.unsqueeze(-1) * u_rad + (r * self.target_orbit_w).unsqueeze(-1) * u_tan
            vel_body = torch.cat(
                [vel_body_xy, torch.zeros(self.num_envs, 1, device=self.device)],
                dim=-1,
            )
            lin_vel = quat_apply(quat, vel_body)

        target_quat = self.target.data.root_quat_w
        self.target.write_root_pose_to_sim(torch.cat([new_pos, target_quat], dim=-1))
        ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.target.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1))

    def _reset_target_orbit(self, ids: torch.Tensor, robot_pos_w: torch.Tensor) -> None:
        if self.cfg.target_orbit_randomize_radius_on_reset:
            r = sample_uniform(
                self.cfg.target_orbit_radius_min,
                self.cfg.target_orbit_radius_max,
                (len(ids),),
                self.device,
            )
            self._target_orbit_radius[ids] = r
        else:
            self._target_orbit_radius[ids] = float(self.cfg.target_orbit_radius)

        self.target_orbit_w[ids] = sample_uniform(
            self.cfg.target_orbit_angular_vel_min,
            self.cfg.target_orbit_angular_vel_max,
            (len(ids),),
            self.device,
        )
        self._target_orbit_w_ref[ids] = self.target_orbit_w[ids]
        self._target_orbit_r_ref[ids] = self._target_orbit_radius[ids]
        self.target_orbit_phase[ids] = 0.0

        r = self._target_orbit_radius[ids]
        frame = str(getattr(self.cfg, "target_orbit_frame", "world")).lower()
        if frame == "world":
            target_pos_w = self._world_orbit_position(r, self.target_orbit_phase[ids], env_ids=ids)
        else:
            quat = self._robot_command_quat(ids)
            target_pos_w = self._body_offset_to_world(
                r, self.target_orbit_phase[ids], quat, robot_pos_w
            )
        target_quat = torch.zeros(len(ids), 4, device=self.device)
        target_quat[:, 0] = 1.0
        self.target.write_root_pose_to_sim(torch.cat([target_pos_w, target_quat], dim=-1), ids)
        self.target.write_root_velocity_to_sim(
            torch.zeros(len(ids), 6, device=self.device),
            ids,
        )

    def _reset_target_constant_velocity(self, ids: torch.Tensor, robot_pos_w: torch.Tensor) -> None:
        n = len(ids)
        r = sample_uniform(
            float(getattr(self.cfg, "target_cv_spawn_radius_min", 2.0)),
            float(getattr(self.cfg, "target_cv_spawn_radius_max", 4.0)),
            (n,),
            self.device,
        )
        half_fov = math.radians(float(getattr(self.cfg, "target_cv_spawn_half_fov_deg", 75.0)))
        bearing = sample_uniform(-half_fov, half_fov, (n,), self.device)
        rel_xy = torch.stack([r * torch.cos(bearing), r * torch.sin(bearing)], dim=-1)

        speed = sample_uniform(
            float(getattr(self.cfg, "target_cv_speed_min", 0.20)),
            float(getattr(self.cfg, "target_cv_speed_max", 0.50)),
            (n,),
            self.device,
        )
        heading = sample_uniform(-math.pi, math.pi, (n,), self.device)
        vel_xy = torch.stack([speed * torch.cos(heading), speed * torch.sin(heading)], dim=-1)
        self._target_cv_vel[ids] = vel_xy

        origins = self.scene.env_origins[ids]
        z = origins[:, 2] + float(self.cfg.workspace_height)
        target_pos_w = torch.cat([origins[:, :2] + rel_xy, z.unsqueeze(-1)], dim=-1)
        target_quat = torch.zeros(n, 4, device=self.device)
        target_quat[:, 0] = 1.0
        self.target.write_root_pose_to_sim(torch.cat([target_pos_w, target_quat], dim=-1), ids)
        lin_vel = torch.cat([vel_xy, torch.zeros(n, 1, device=self.device)], dim=-1)
        self.target.write_root_velocity_to_sim(
            torch.cat([lin_vel, torch.zeros(n, 3, device=self.device)], dim=-1),
            ids,
        )

    def _reset_target_fixed_radius_cv(self, ids: torch.Tensor) -> None:
        n = len(ids)
        r_val = float(getattr(self.cfg, "target_fixed_radius", 3.0))
        self._target_fixed_radius[ids] = r_val

        speed = sample_uniform(
            float(getattr(self.cfg, "target_fixed_speed_min", 0.25)),
            float(getattr(self.cfg, "target_fixed_speed_max", 0.40)),
            (n,),
            self.device,
        )
        sign = torch.where(
            torch.rand(n, device=self.device) > 0.5,
            torch.ones(n, device=self.device),
            -torch.ones(n, device=self.device),
        )
        omega = sign * speed / max(r_val, 1.0e-6)
        self._target_fixed_omega[ids] = omega

        lo, hi = self._target_phase_limits_rad()
        phase = sample_uniform(lo, hi, (n,), self.device)
        self.target_orbit_phase[ids] = phase

        r = self._target_fixed_radius[ids]
        target_pos_w = self._world_orbit_position(r, phase, env_ids=ids)
        c = torch.cos(phase)
        s = torch.sin(phase)
        vel_xy = (r * omega).unsqueeze(-1) * torch.stack([-s, c], dim=-1)
        target_quat = torch.zeros(n, 4, device=self.device)
        target_quat[:, 0] = 1.0
        self.target.write_root_pose_to_sim(torch.cat([target_pos_w, target_quat], dim=-1), ids)
        lin_vel = torch.cat([vel_xy, torch.zeros(n, 1, device=self.device)], dim=-1)
        self.target.write_root_velocity_to_sim(
            torch.cat([lin_vel, torch.zeros(n, 3, device=self.device)], dim=-1),
            ids,
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        robot_default = self.robot.data.default_root_state[ids].clone()
        robot_default[:, :3] += self.scene.env_origins[ids]
        self.robot.write_root_pose_to_sim(robot_default[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(
            torch.zeros(len(env_ids), 6, device=self.device),
            env_ids,
        )

        robot_quat = robot_default[:, 3:7]
        robot_pos_w = robot_default[:, :3]
        _, _, yaw = self._quat_to_euler(robot_quat)
        self._robot_yaw[ids] = self._clamp_yaw_to_motion_limits(wrap_to_pi(yaw))

        if self._target_motion_mode() == "constant_velocity":
            self._reset_target_constant_velocity(ids, robot_pos_w)
        elif self._target_motion_mode() == "fixed_radius_cv":
            self._reset_target_fixed_radius_cv(ids)
        else:
            self._reset_target_orbit(ids, robot_pos_w)

        self.actions[ids] = 0.0
        self._net_action[ids] = 0.0
        self._exec_wz[ids] = 0.0
        self.sonar_yaw[ids] = self._robot_yaw[ids]
        self.out_of_view_steps[ids] = 0
        self.target_in_fov[ids] = True
        self.target_facing_quality[ids] = 1.0
        self.belief.reset(ids)
        self.sonar_history[ids] = 0.0
        self.sonar_features[ids] = 0.0
        self.processed_sonar[ids] = 0.0
        self._pipeline_cache_step = -1

        if hasattr(self, "oceansim_sonar"):
            self.oceansim_sonar.reset_envs(env_ids)

        self._refresh_viewport_camera()

    def close(self):
        if hasattr(self, "oceansim_sonar"):
            self.oceansim_sonar.close()
        if self._tb_writer is not None:
            self._tb_writer.close()
        super().close()
