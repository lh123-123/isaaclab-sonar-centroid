# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sonar centroid tracking environment (orchestrator over M0–M11 modules)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from isaaclab_tasks.direct.sonar_flow_hover.modules.actions import (
    apply_target_motion,
    apply_velocity_action,
    enforce_lateral_1d_kinematics,
    hold_robot_at_station,
)
from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import (
    euler_xyz_from_quat,
    robot_lateral_sway_signed,
    yaw_toward_beacon,
)
from isaaclab_tasks.direct.sonar_flow_hover.modules.reset_logic import (
    reset_beacon_and_station,
    sample_lateral_spawn_offset,
)

from .modules.actions import continuous_to_lateral_velocity, discrete_to_lateral_velocity
from .modules.centroid_bus import make_centroid_frame
from .modules.constants import ACTION_STAY
from .modules.fixed_point import FixedPointStore
from .modules.geometry_perception import GeometryPerception
from .modules.obs_correspondence import correspondence_report, format_correspondence_line
from .modules.observation import DiscreteObservationBuilder
from .modules.rewards import compute_reward
from .modules.sonar_centroid_tracker import SonarCentroidTracker
from .modules.sonar_gt import SonarGtDetector
from .modules.sonar_pipeline import PolarSonarPipeline
from .modules.terminations import CentroidTerminationChecker
from .modules.target_motion import round_trip_dy, round_trip_vy, sine_dy_and_vy, triangle_dy, triangle_vy
from .shared_platform import OceanSimSonarSensor, mount_oceansim_sonar_on_robots
from .sonar_centroid_track_env_cfg import SonarCentroidTrackEnvCfg


class SonarCentroidTrackEnv(DirectRLEnv):
    """AUV lateral tracking via polar bright-spot centroid (fixed-point frame) + discrete or continuous PPO."""

    cfg: SonarCentroidTrackEnvCfg

    def __init__(self, cfg: SonarCentroidTrackEnvCfg, render_mode: str | None = None, **kwargs):
        n_envs = cfg.scene.num_envs
        dev = cfg.sim.device
        self._continuous_actions = bool(getattr(cfg, "continuous_actions", False))

        self.pipeline = PolarSonarPipeline(cfg, n_envs, dev)
        self.gt = SonarGtDetector(
            threshold_ratio=float(cfg.feature_polar_threshold_ratio),
            polar_width=int(cfg.oceansim_sonar_cfg.width),
        )
        self.fixed = FixedPointStore(
            n_envs,
            dev,
            polar_width=int(cfg.oceansim_sonar_cfg.width),
            sonar_gt=self.gt,
        )
        self.tracker = SonarCentroidTracker(
            self.gt,
            self.fixed,
            polar_width=int(cfg.oceansim_sonar_cfg.width),
            polar_height=int(cfg.oceansim_sonar_cfg.height),
        )
        self.geometry = GeometryPerception(self.fixed)
        self.obs_builder = DiscreteObservationBuilder(
            n_envs,
            dev,
            lateral_amp_m=float(cfg.lateral_target_amp_m),
            target_period_s=float(cfg.lateral_target_period_s),
            polar_width=int(cfg.oceansim_sonar_cfg.width),
            polar_height=int(cfg.oceansim_sonar_cfg.height),
            image_only=bool(getattr(cfg, "image_only_obs", False)),
            include_range_obs=bool(getattr(cfg, "include_range_obs", False)),
            continuous_actions=self._continuous_actions,
        )
        self.term_checker = CentroidTerminationChecker(n_envs, dev)

        self._station_origin_w = torch.zeros(n_envs, 3, device=dev)
        self._beacon_pos_w = torch.zeros(n_envs, 3, device=dev)
        self._beacon_vel_w = torch.zeros(n_envs, 3, device=dev)
        self._kinematic_offset_w = torch.zeros(n_envs, 3, device=dev)
        self._target_kinematic_offset_b = torch.zeros(n_envs, 3, device=dev)
        self._fixed_robot_yaw_w = torch.zeros(n_envs, device=dev)
        self._lateral_target_phase = torch.zeros(n_envs, device=dev)
        if self._continuous_actions:
            self.actions = torch.zeros(n_envs, dtype=torch.float32, device=dev)
            self._prev_actions = torch.zeros(n_envs, dtype=torch.float32, device=dev)
        else:
            self.actions = torch.full((n_envs,), ACTION_STAY, dtype=torch.long, device=dev)
            self._prev_actions = torch.full((n_envs,), ACTION_STAY, dtype=torch.long, device=dev)
        self._lateral_vy_override: torch.Tensor | None = None
        self._lateral_sway_override: torch.Tensor | None = None
        self._lateral_surge_override: torch.Tensor | None = None
        self._lateral_heave_override: torch.Tensor | None = None

        ox, oy, oz = cfg.sonar_mount_offset_local
        self._sonar_mount_offset_b = (
            torch.tensor((ox, oy, oz), device=dev, dtype=torch.float32).unsqueeze(0).repeat(n_envs, 1)
        )

        super().__init__(cfg, render_mode, **kwargs)

    @property
    def polar_image(self) -> torch.Tensor:
        return self.pipeline.polar_image

    def _ensure_pipeline(self) -> None:
        self.pipeline.ensure_updated(self.common_step_counter, self.oceansim_sonar, self.sim)
        self.tracker.bind_polar(self.pipeline.polar_image)

    def _beam_prior_azimuth_rad(self) -> torch.Tensor | None:
        """Optional azimuth gate center per env; override in AUV env for teach/beacon prior."""
        return None

    def beam_peak_range_meas_m(
        self,
        *,
        peak_threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Peak-intensity beam range (m) from native OceanSim beams, not image centroids."""
        self._ensure_pipeline()
        if peak_threshold is None:
            peak_threshold = float(getattr(self.cfg, "sonar_beam_peak_threshold", 0.08))
        from .modules.sonar_beam_range import peak_beam_range_from_beams

        gate_deg = float(getattr(self.cfg, "sonar_beam_azimuth_gate_deg", 0.0))
        return peak_beam_range_from_beams(
            self.oceansim_sonar.get_beam_data(),
            peak_threshold=peak_threshold,
            prior_azimuth=self._beam_prior_azimuth_rad(),
            prior_gate_deg=gate_deg,
        )

    def get_centroid_frame(self, env_id: int = 0):
        self._ensure_pipeline()
        polar_np = self.pipeline.polar_image[env_id].detach().cpu().numpy()
        p_fixed = self.fixed.p_fixed_abs_env(env_id)
        if p_fixed is not None:
            p_sonar = self.gt.detect_abs_roi(polar_np, (p_fixed.col, p_fixed.row))
            if p_sonar is None:
                p_sonar = self.gt.detect_abs(polar_np)
        else:
            p_sonar = self.gt.detect_abs(polar_np)
        ocfg = self.cfg.oceansim_sonar_cfg
        p_perc_list = self.geometry.project_abs(
            beacon_pos_w=self._beacon_pos_w,
            robot_pos_w=self.robot.data.root_pos_w,
            robot_quat_w=self.robot.data.root_quat_w,
            sonar_mount_offset_b=self._sonar_mount_offset_b,
            env_ids=torch.tensor([env_id], device=self.device),
            min_range=float(ocfg.min_range),
            max_range=float(ocfg.max_range),
            polar_height=int(ocfg.height),
            polar_width=int(ocfg.width),
            hori_fov_deg=float(ocfg.hori_fov),
        )
        p_perc_abs = p_perc_list[0]
        valid = 1.0 if p_sonar is not None else 0.0
        return make_centroid_frame(
            p_sonar_abs=p_sonar,
            p_perc_abs=p_perc_abs,
            p_fixed_abs=self.fixed.p_fixed_abs_env(env_id),
            valid_ratio=valid,
        )

    def set_lateral_velocity_override(self, vy: torch.Tensor | None) -> None:
        """Debug/eval: bypass discrete actions and apply continuous body-Y velocity (m/s)."""
        self._lateral_vy_override = vy

    def set_lateral_sway_override(self, sway_m: torch.Tensor | None) -> None:
        """Debug/eval: command target lateral sway position directly (m)."""
        self._lateral_sway_override = sway_m

    def set_lateral_surge_override(self, surge_m: torch.Tensor | None) -> None:
        """Debug/eval: command target surge (forward) offset from station (m)."""
        self._lateral_surge_override = surge_m

    def set_lateral_heave_override(self, heave_z_m: torch.Tensor | None) -> None:
        """Debug/eval: command target world Z depth (m); None = hold station depth."""
        self._lateral_heave_override = heave_z_m

    def set_displacement_offset(
        self,
        dx: float,
        dy: float,
        dz: float = 0.0,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        n = len(env_ids)
        zeros = torch.zeros(n, device=self.device)
        yaw = self._fixed_robot_yaw_w[env_ids]
        quat = quat_from_euler_xyz(zeros, zeros, yaw)
        offset_b = torch.zeros(n, 3, device=self.device)
        offset_b[:, 1] = dy
        self._kinematic_offset_w[env_ids] = quat_apply(quat, offset_b)
        self.pipeline.invalidate_cache()
        if bool(getattr(self.cfg, "freeze_robot_pose", False)):
            self._hold_robot(env_ids)

    def set_target_offset(
        self,
        dx: float,
        dy: float,
        dz: float = 0.0,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self._target_kinematic_offset_b[env_ids, 0] = 0.0
        self._target_kinematic_offset_b[env_ids, 1] = dy
        self._target_kinematic_offset_b[env_ids, 2] = 0.0
        self._write_beacon_from_target_offset(env_ids)
        self._beacon_vel_w[env_ids] = 0.0
        self.pipeline.invalidate_cache()
        if bool(getattr(self.cfg, "freeze_robot_pose", False)):
            self._hold_robot(env_ids)

    def set_target_velocity(self, vx: float, vy: float, vz: float = 0.0, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self._beacon_vel_w[env_ids, 0] = 0.0
        self._beacon_vel_w[env_ids, 1] = vy
        self._beacon_vel_w[env_ids, 2] = 0.0

    def _hold_robot(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        hold_robot_at_station(
            self.robot,
            self._station_origin_w,
            self._kinematic_offset_w,
            self._beacon_pos_w,
            env_ids,
            fixed_yaw_w=self._fixed_robot_yaw_w,
        )

    def _write_beacon_from_target_offset(self, env_ids: torch.Tensor) -> None:
        beacon_local = torch.tensor(self.cfg.beacon_offset_local, device=self.device, dtype=torch.float32)
        n = len(env_ids)
        zeros = torch.zeros(n, device=self.device)
        yaw = self._fixed_robot_yaw_w[env_ids]
        quat = quat_from_euler_xyz(zeros, zeros, yaw)
        offset_w = quat_apply(quat, self._target_kinematic_offset_b[env_ids])
        self._beacon_pos_w[env_ids] = self._station_origin_w[env_ids] + beacon_local.unsqueeze(0) + offset_w
        pose = torch.zeros(n, 7, device=self.device)
        pose[:, :3] = self._beacon_pos_w[env_ids]
        pose[:, 3] = 1.0
        self.beacon.write_root_pose_to_sim(pose.clone(), env_ids)
        self.beacon.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

    def _update_lateral_target_motion(self) -> None:
        if not bool(getattr(self.cfg, "lateral_target_moving", False)):
            return
        mode = str(getattr(self.cfg, "lateral_target_mode", "sine"))
        amp = float(self.cfg.lateral_target_amp_m)
        dt = float(self.cfg.sim.dt * self.cfg.decimation)
        step = self.episode_length_buf
        env_ids = self.robot._ALL_INDICES

        if mode == "round_trip":
            vy = float(getattr(self.cfg, "lateral_target_vy", 0.28))
            dy = torch.zeros(self.num_envs, device=self.device)
            for i in range(self.num_envs):
                s = int(step[i].item())
                dy[i] = round_trip_dy(s + 1, sway_amp=amp, vy=vy, step_dt=dt)
            self._target_kinematic_offset_b[env_ids, 0] = 0.0
            self._target_kinematic_offset_b[env_ids, 1] = dy
            self._target_kinematic_offset_b[env_ids, 2] = 0.0
            self._write_beacon_from_target_offset(env_ids)
            return

        if mode == "triangle":
            vy = float(getattr(self.cfg, "lateral_target_vy", 0.28))
            dy = torch.zeros(self.num_envs, device=self.device)
            for i in range(self.num_envs):
                s = int(step[i].item())
                dy[i] = triangle_dy(s + 1, sway_amp=amp, vy=vy, step_dt=dt)
            self._target_kinematic_offset_b[env_ids, 0] = 0.0
            self._target_kinematic_offset_b[env_ids, 1] = dy
            self._target_kinematic_offset_b[env_ids, 2] = 0.0
            self._write_beacon_from_target_offset(env_ids)
            return

        period = max(float(self.cfg.lateral_target_period_s), 1.0e-3)
        t = step.to(dtype=torch.float32, device=self.device) * dt
        omega = 2.0 * math.pi / period
        dy = amp * torch.sin(omega * t + self._lateral_target_phase)
        self._target_kinematic_offset_b[env_ids, 0] = 0.0
        self._target_kinematic_offset_b[env_ids, 1] = dy
        self._target_kinematic_offset_b[env_ids, 2] = 0.0
        self._write_beacon_from_target_offset(env_ids)

    def _robot_sway_m(self) -> torch.Tensor:
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        return robot_lateral_sway_signed(self._station_origin_w, self.robot.data.root_pos_w, yaw)

    def _target_dy_m(self) -> torch.Tensor:
        return self._target_kinematic_offset_b[:, 1].clone()

    def _target_dy_vel_m(self) -> torch.Tensor:
        if not bool(getattr(self.cfg, "lateral_target_moving", False)):
            return torch.zeros(self.num_envs, device=self.device)
        mode = str(getattr(self.cfg, "lateral_target_mode", "sine"))
        amp = float(self.cfg.lateral_target_amp_m)
        dt = float(self.cfg.sim.dt * self.cfg.decimation)
        step = self.episode_length_buf

        if mode == "round_trip":
            vy = float(getattr(self.cfg, "lateral_target_vy", 0.28))
            out = torch.zeros(self.num_envs, device=self.device)
            for i in range(self.num_envs):
                out[i] = round_trip_vy(int(step[i].item()), sway_amp=amp, vy=vy, step_dt=dt)
            return out

        if mode == "triangle":
            vy = float(getattr(self.cfg, "lateral_target_vy", 0.28))
            out = torch.zeros(self.num_envs, device=self.device)
            for i in range(self.num_envs):
                out[i] = triangle_vy(int(step[i].item()), sway_amp=amp, vy=vy, step_dt=dt)
            return out

        period = max(float(self.cfg.lateral_target_period_s), 1.0e-3)
        out = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            _, vy = sine_dy_and_vy(
                int(step[i].item()),
                amp=amp,
                period_s=period,
                step_dt=dt,
                phase=float(self._lateral_target_phase[i].item()),
            )
            out[i] = vy
        return out

    def _calibrate_fixed_point(self, env_ids: torch.Tensor) -> None:
        n_settle = int(getattr(self.cfg, "fixed_point_calib_settle_steps", 8))
        n_warmup = int(self.cfg.sonar_warmup_frames) + max(n_settle, 0)
        self.pipeline.warmup(self.oceansim_sonar, self.sim, n_frames=n_warmup)
        self.tracker.bind_polar(self.pipeline.polar_image)

        snap_path = getattr(self.cfg, "fixed_point_snapshot_path", None)

        for idx in env_ids.detach().cpu().tolist():
            polar_np = self.pipeline.polar_image[idx].detach().cpu().numpy()
            try:
                if bool(getattr(self.cfg, "auto_calibrate_fixed_point", True)):
                    p_fixed = self.fixed.calibrate_from_polar(polar_np, env_id=idx)
                    p_sonar = self.gt.detect_abs(polar_np)
                    gap = float("nan")
                    if p_sonar is not None:
                        gap = math.hypot(p_fixed.col - p_sonar.col, p_fixed.row - p_sonar.row)
                    if bool(getattr(self.cfg, "print_centroid_debug", False)) or (
                        p_sonar is not None and gap > 1.5
                    ):
                        sonar_s = (
                            f"({p_sonar.col:.1f},{p_sonar.row:.1f})"
                            if p_sonar is not None
                            else "None"
                        )
                        print(
                            f"[SonarCentroidTrack] env={idx}: fixed=({p_fixed.col:.1f},{p_fixed.row:.1f}) "
                            f"sonar={sonar_s} gap={gap:.2f}px",
                            flush=True,
                        )
                    if self.num_envs == 1 and snap_path:
                        from pathlib import Path

                        out = Path(snap_path)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        self.fixed.save(out, env_id=idx)
            except RuntimeError as exc:
                print(f"[SonarCentroidTrack] fixed-point calib failed env={idx}: {exc}", flush=True)

    def recalibrate_fixed_point(self, env_ids: torch.Tensor | None = None) -> None:
        """Re-run fixed-point teach (e.g. after reset settle in debug scripts)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._calibrate_fixed_point(env_ids)

    def _setup_scene(self) -> None:
        self.robot = RigidObject(self.cfg.robot_cfg)
        self.beacon = RigidObject(self.cfg.beacon_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.rigid_objects["robot"] = self.robot
        self.scene.rigid_objects["target"] = self.beacon

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.oceansim_sonar = OceanSimSonarSensor(
            self.cfg.oceansim_sonar_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )
        mount_oceansim_sonar_on_robots(
            self.oceansim_sonar,
            num_envs=self.num_envs,
            sonar_mount_offset_local=self.cfg.sonar_mount_offset_local,
            sim=self.sim,
            sonar_mount_rpy_deg=getattr(self.cfg, "sonar_mount_rpy_deg", None),
            sonar_mount_visual=bool(getattr(self.cfg, "sonar_mount_visual", False)),
            sonar_housing_offset_local=getattr(self.cfg, "sonar_housing_offset_local", None),
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self.actions.clone()
        if self._continuous_actions:
            self.actions = actions.view(-1).float().clamp(-1.0, 1.0)
        else:
            self.actions = actions.view(-1).long().clamp(0, 2)

    def _apply_action(self) -> None:
        if bool(getattr(self.cfg, "freeze_robot_pose", False)):
            self._update_lateral_target_motion()
            apply_target_motion(
                self.beacon,
                self._beacon_pos_w,
                self._beacon_vel_w,
                target_speed_max=1.0e9,
                dt=float(self.cfg.sim.dt * self.cfg.decimation),
                all_env_ids=self.robot._ALL_INDICES,
            )
            self._hold_robot()
            return

        self._update_lateral_target_motion()
        if self._lateral_vy_override is not None:
            vy = self._lateral_vy_override.view(-1)
        elif self._continuous_actions:
            vy = continuous_to_lateral_velocity(
                self.actions,
                max_lateral_velocity=float(self.cfg.max_lateral_velocity),
                num_envs=self.num_envs,
                device=self.device,
            )
        else:
            vy = discrete_to_lateral_velocity(
                self.actions,
                max_lateral_velocity=float(self.cfg.max_lateral_velocity),
                num_envs=self.num_envs,
                device=self.device,
            )
        max_v = max(float(self.cfg.max_lateral_velocity), 1.0e-6)
        cmd = torch.zeros(self.num_envs, 4, device=self.device)
        cmd[:, 1] = vy / max_v
        apply_velocity_action(
            self.robot,
            cmd,
            max_lin_vel=max_v,
            max_heave_vel=0.0,
            max_ang_vel=0.0,
            lateral_1d_only=True,
        )
        enforce_lateral_1d_kinematics(
            self.robot,
            self._station_origin_w,
            self._fixed_robot_yaw_w,
            self.robot._ALL_INDICES,
        )
        apply_target_motion(
            self.beacon,
            self._beacon_pos_w,
            self._beacon_vel_w,
            target_speed_max=1.0e9 if bool(getattr(self.cfg, "lateral_target_moving", False)) else 0.0,
            dt=float(self.cfg.sim.dt * self.cfg.decimation),
            all_env_ids=self.robot._ALL_INDICES,
        )

    def ensure_tracking(self, env_ids: torch.Tensor | None = None) -> None:
        """Refresh polar binding (detection runs on read)."""
        del env_ids
        self._ensure_pipeline()

    def _get_observations(self) -> dict:
        self._ensure_pipeline()
        self.obs_builder.set_prev_action(self._prev_actions)
        range_meas = range_valid = None
        if bool(getattr(self.cfg, "include_range_obs", False)):
            range_meas, _, _, range_valid = self.beam_peak_range_meas_m()
        obs = self.obs_builder.build(
            self.tracker,
            robot_sway_m=self._robot_sway_m(),
            target_dy_m=self._target_dy_m(),
            target_dy_vel_m=self._target_dy_vel_m(),
            range_ref_gt_m=(
                self.range_ref_setpoint_m()
                if hasattr(self, "range_ref_setpoint_m")
                else (
                    self.range_ref_for_control_m()
                    if hasattr(self, "range_ref_for_control_m")
                    else getattr(self, "_range_ref_gt_m", None)
                )
            ),
            min_range=float(self.cfg.oceansim_sonar_cfg.min_range),
            max_range=float(self.cfg.oceansim_sonar_cfg.max_range),
            range_accept_m=float(getattr(self.cfg, "range_accept_m", 0.5)),
            sonar_range_meas_m=range_meas,
            sonar_range_valid=range_valid,
        )
        if bool(getattr(self.cfg, "print_centroid_debug", False)):
            interval = int(getattr(self.cfg, "centroid_debug_interval", 20))
            if int(self.common_step_counter) % interval == 0:
                frame = self.get_centroid_frame(0)
                print(f"[centroid] step={self.common_step_counter} {frame.summary()}", flush=True)
        if bool(getattr(self.cfg, "print_policy_obs_debug", False)):
            interval = max(int(getattr(self.cfg, "policy_obs_debug_interval", 10)), 1)
            if int(self.common_step_counter) % interval == 0:
                w = int(self.cfg.oceansim_sonar_cfg.width)
                h = int(self.cfg.oceansim_sonar_cfg.height)
                for i in range(self.num_envs):
                    prev = self._prev_actions[i]
                    prev_arg = float(prev.item()) if self._continuous_actions else int(prev.item())
                    rep = correspondence_report(
                        self.tracker,
                        env_id=i,
                        prev_action=prev_arg,
                        polar_width=w,
                        polar_height=h,
                        continuous_actions=self._continuous_actions,
                    )
                    o = obs[i].detach().cpu().tolist()
                    ref = rep["policy_obs_4d"]
                    ok = rep["u_match"] and rep["v_match"] and all(
                        abs(o[k] - ref[k]) < 1.0e-5 for k in range(4)
                    )
                    tag = "OK" if ok else "OBS_MISMATCH"
                    print(
                        f"[ppo_obs] env={i} {format_correspondence_line(rep, step=int(self.common_step_counter))}",
                        flush=True,
                    )
                    print(
                        f"[ppo_obs] env={i} obs={[f'{x:.6f}' for x in o]} "
                        f"ref={[f'{x:.6f}' for x in ref]} [{tag}]",
                        flush=True,
                    )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._ensure_pipeline()
        return compute_reward(
            self.tracker,
            self.actions,
            self._prev_actions,
            min_depth=float(self.cfg.min_depth),
            max_depth=float(self.cfg.max_depth),
            depth=self.robot.data.root_pos_w[:, 2],
            min_valid_ratio=float(self.cfg.min_valid_ratio),
            continuous_actions=self._continuous_actions,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_pipeline()
        return self.term_checker.evaluate(
            cfg=self.cfg,
            tracker=self.tracker,
            robot=self.robot,
            scene=self.scene,
            station_origin_w=self._station_origin_w,
            episode_length_buf=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
        )

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids_t = (
            env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        )

        self._target_kinematic_offset_b[env_ids_t] = 0.0
        self._kinematic_offset_w[env_ids_t] = 0.0
        mode = str(getattr(self.cfg, "lateral_target_mode", "sine"))
        if mode == "sine":
            # Start at dy=0 for 0 → +A → 0 → −A → 0 round trip (matches triangle eval).
            self._lateral_target_phase[env_ids_t] = 0.0
        else:
            self._lateral_target_phase[env_ids_t] = torch.rand(len(env_ids_t), device=self.device) * 2.0 * math.pi

        reset_beacon_and_station(
            self.cfg,
            env_ids_t,
            self.scene,
            self.beacon,
            None,
            station_origin_w=self._station_origin_w,
            beacon_pos_w=self._beacon_pos_w,
            beacon2_pos_w=self._beacon_pos_w,
            beacon_vel_w=self._beacon_vel_w,
            use_second_beacon=False,
        )

        n = len(env_ids_t)
        radius = float(getattr(self.cfg, "eval_spawn_radius", 0.0) if self.cfg.eval_mode else self.cfg.spawn_radius)
        yaw = yaw_toward_beacon(self._beacon_pos_w[env_ids_t], self._station_origin_w[env_ids_t])
        yaw_range = math.radians(float(getattr(self.cfg, "spawn_yaw_range_deg", 0.0)))
        if yaw_range > 0.0:
            yaw = yaw + (torch.rand(n, device=self.device) * 2.0 - 1.0) * yaw_range
        self._fixed_robot_yaw_w[env_ids_t] = yaw

        if radius > 0.0:
            offset = sample_lateral_spawn_offset(n, radius, yaw, self.device)
            spawn_pos = self._station_origin_w[env_ids_t] + offset
        else:
            spawn_pos = self._station_origin_w[env_ids_t].clone()

        zeros = torch.zeros(n, device=self.device)
        quat = quat_from_euler_xyz(zeros, zeros, yaw)
        pose = torch.cat([spawn_pos, quat], dim=-1)
        self.robot.write_root_pose_to_sim(pose.clone(), env_ids_t)
        self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids_t)

        self.fixed.reset_envs(env_ids_t)
        self.tracker.reset(env_ids_t)
        self.term_checker.reset(env_ids_t)
        if self._continuous_actions:
            self.actions[env_ids_t] = 0.0
            self._prev_actions[env_ids_t] = 0.0
        else:
            self.actions[env_ids_t] = ACTION_STAY
            self._prev_actions[env_ids_t] = ACTION_STAY
        self.pipeline.invalidate_cache()

        if self.oceansim_sonar is not None:
            self.oceansim_sonar.reset_envs(env_ids_t)

        if not bool(getattr(self.cfg, "auv_physics_mode", False)):
            enforce_lateral_1d_kinematics(
                self.robot,
                self._station_origin_w,
                self._fixed_robot_yaw_w,
                env_ids_t,
            )
            if bool(getattr(self.cfg, "freeze_robot_pose", False)):
                self._hold_robot(env_ids_t)

        self._calibrate_fixed_point(env_ids_t)

    def close(self) -> None:
        if hasattr(self, "oceansim_sonar"):
            self.oceansim_sonar.close()
        super().close()
