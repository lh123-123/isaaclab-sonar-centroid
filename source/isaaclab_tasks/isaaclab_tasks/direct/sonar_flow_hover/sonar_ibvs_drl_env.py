# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sonar sparse-feature IBVS + DRL environment (thin orchestrator over modules)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from .modules.actions import (
    apply_target_motion,
    apply_velocity_action,
    enforce_lateral_1d_kinematics,
    hold_robot_at_station,
)
from .modules.debug import collect_metrics, maybe_print_ibvs_debug
from .modules.geometry import (
    euler_xyz_from_quat,
    geometric_centroid_errors_px,
    robot_lateral_sway_abs,
    station_pos_error_norm,
)
from .modules.observation import Lateral1DObservationBuilder, ObservationBuilder
from .modules.reference_manager import ReferenceManager
from .modules.reset_logic import reset_beacon_and_station, reset_robot_episode, write_beacon_pose
from .modules.rewards import compute_reward, compute_reward_lateral_1d
from .modules.sonar_pipeline import SonarPerceptionPipeline
from .modules.terminations import TerminationChecker
from .reference_snapshot import ReferenceSnapshot
from .shared_platform import OceanSimSonarSensor, mount_oceansim_sonar_on_robots
from .sonar_ibvs_drl_env_cfg import SonarIbvsDrlEnvCfg


class SonarIbvsDrlEnv(DirectRLEnv):
    """AUV station keeping / tracking via sparse sonar features + DRL."""

    cfg: SonarIbvsDrlEnvCfg

    def __init__(self, cfg: SonarIbvsDrlEnvCfg, render_mode: str | None = None, **kwargs):
        n_envs = cfg.scene.num_envs
        dev = cfg.sim.device
        step_dt = float(cfg.sim.dt * cfg.decimation)

        self.pipeline = SonarPerceptionPipeline(cfg, n_envs, dev)
        self.ref_mgr = ReferenceManager(cfg, n_envs, dev)
        self._lateral_compact_obs = bool(getattr(cfg, "lateral_1d_mode", False)) and str(
            getattr(cfg, "lateral_1d_obs_mode", "full")
        ).lower() == "compact"
        if self._lateral_compact_obs:
            self.obs_builder = Lateral1DObservationBuilder(
                n_envs,
                dev,
                max_track_points=int(cfg.max_track_features),
            )
        else:
            self.obs_builder = ObservationBuilder(
                n_envs,
                dev,
                ang_vel_scale=float(cfg.obs_ang_vel_scale),
                depth_scale=float(cfg.obs_depth_scale),
                depth_rate_scale=float(cfg.obs_depth_rate_scale),
                step_dt=step_dt,
            )
        self.term_checker = TerminationChecker(n_envs, dev)

        self._station_origin_w = torch.zeros(n_envs, 3, device=dev)
        self._beacon_pos_w = torch.zeros(n_envs, 3, device=dev)
        self._beacon2_pos_w = torch.zeros(n_envs, 3, device=dev)
        self._beacon_vel_w = torch.zeros(n_envs, 3, device=dev)
        self._use_second_beacon = bool(getattr(cfg, "use_second_beacon", True))
        self._kinematic_offset_w = torch.zeros(n_envs, 3, device=dev)
        self._target_kinematic_offset_b = torch.zeros(n_envs, 3, device=dev)
        self._fixed_robot_yaw_w = torch.zeros(n_envs, device=dev)
        self._current_vel_w = torch.zeros(n_envs, 3, device=dev)
        action_dim = int(cfg.action_space.shape[0])
        self.actions = torch.zeros(n_envs, action_dim, device=dev)
        self._lateral_target_phase = torch.zeros(n_envs, device=dev)
        self._geometric_centroid_bias_px = torch.zeros(n_envs, 2, device=dev)

        ox, oy, oz = cfg.sonar_mount_offset_local
        self._sonar_mount_offset_b = torch.tensor(
            (ox, oy, oz), device=dev, dtype=torch.float32
        ).unsqueeze(0).repeat(n_envs, 1)

        super().__init__(cfg, render_mode, **kwargs)

        if self.ref_mgr.use_file:
            self.ref_mgr.load_from_disk(self.pipeline)

    # --- Backward-compatible accessors for debug scripts ---
    @property
    def _hover_pos_w(self) -> torch.Tensor:
        return self._station_origin_w

    @property
    def feature_tracker(self):
        return self.pipeline.feature_tracker

    @property
    def sonar_image(self):
        return self.pipeline.sonar_image

    @property
    def cart_image(self):
        return self.pipeline.cart_image

    @property
    def polar_image(self):
        return self.pipeline.polar_image

    @property
    def track_image(self):
        return self.pipeline.track_image

    @property
    def _reference_mode(self) -> str:
        return self.ref_mgr.mode

    def get_ibvs_metrics(self, env_id: int = 0) -> dict:
        """Public debug API: snapshot pipeline + robot metrics."""
        centroid_err_px = None
        if self._lateral_compact_obs:
            centroid_err_px = self._geometric_centroid_errors_px()
        return collect_metrics(
            pipeline=self.pipeline,
            robot=self.robot,
            station_origin_w=self._station_origin_w,
            actions=self.actions,
            reference_mode=self.ref_mgr.mode,
            kinematic_offset_w=self._kinematic_offset_w,
            target_kinematic_offset_b=self._target_kinematic_offset_b,
            centroid_err_px=centroid_err_px,
            lateral_1d=self._lateral_1d(),
            env_id=env_id,
        )

    def capture_reference_snapshot(self, path: str | None = None) -> ReferenceSnapshot:
        self._ensure_pipeline()
        return self.ref_mgr.capture_snapshot(
            self.pipeline,
            robot=self.robot,
            station_origin_w=self._station_origin_w,
            beacon_pos_w=self._beacon_pos_w,
            sonar_mount_offset_b=self._sonar_mount_offset_b,
            max_range=self.pipeline.max_range,
            path=path,
        )

    def _hold_robot(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        fixed_yaw = self._fixed_robot_yaw_w if bool(getattr(self.cfg, "freeze_robot_pose", False)) else None
        hold_robot_at_station(
            self.robot,
            self._station_origin_w,
            self._kinematic_offset_w,
            self._beacon_pos_w,
            env_ids,
            fixed_yaw_w=fixed_yaw,
        )

    def _actions_for_physics(self) -> torch.Tensor:
        """Expand 1-d sway policy output to 4-d velocity command layout."""
        if self.actions.shape[-1] == 1:
            full = torch.zeros(self.num_envs, 4, device=self.device)
            full[:, 1] = self.actions[:, 0]
            return full
        if self._lateral_1d():
            full = self.actions.clone()
            full[:, 0] = 0.0
            full[:, 2] = 0.0
            full[:, 3] = 0.0
            return full
        return self.actions

    def _update_lateral_target_motion(self) -> None:
        if not self._lateral_1d() or not bool(getattr(self.cfg, "lateral_target_moving", False)):
            return
        amp = float(self.cfg.lateral_target_amp_m)
        period = max(float(self.cfg.lateral_target_period_s), 1.0e-3)
        omega = 2.0 * math.pi / period
        dt = float(self.cfg.sim.dt * self.cfg.decimation)
        t = self.episode_length_buf.to(dtype=torch.float32, device=self.device) * dt
        dy = amp * torch.sin(omega * t + self._lateral_target_phase)
        env_ids = self.robot._ALL_INDICES
        self._target_kinematic_offset_b[env_ids, 0] = 0.0
        self._target_kinematic_offset_b[env_ids, 1] = dy
        self._target_kinematic_offset_b[env_ids, 2] = 0.0
        self._write_beacon_from_target_offset(env_ids)

    def _lateral_1d(self) -> bool:
        return bool(getattr(self.cfg, "lateral_1d_mode", False))

    def _clamp_target_offset_b(self, dx: float, dy: float, dz: float) -> tuple[float, float, float]:
        if self._lateral_1d():
            return 0.0, dy, 0.0
        return dx, dy, dz

    def _clamp_displacement_w(self, dx: float, dy: float, dz: float) -> tuple[float, float, float]:
        if self._lateral_1d():
            return 0.0, dy, 0.0
        return dx, dy, dz

    def _clamp_target_velocity_w(self, vx: float, vy: float, vz: float) -> tuple[float, float, float]:
        if self._lateral_1d():
            return 0.0, vy, 0.0
        return vx, vy, vz

    def _target_offset_world(self, env_ids: torch.Tensor) -> torch.Tensor:
        offset_b = self._target_kinematic_offset_b[env_ids]
        n = len(env_ids)
        zeros = torch.zeros(n, device=self.device)
        if self._lateral_1d() or bool(getattr(self.cfg, "freeze_robot_pose", False)):
            yaw = self._fixed_robot_yaw_w[env_ids]
        else:
            _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w[env_ids])
        quat = quat_from_euler_xyz(zeros, zeros, yaw)
        return quat_apply(quat, offset_b)

    def _write_beacon_from_target_offset(self, env_ids: torch.Tensor) -> None:
        beacon_local = torch.tensor(self.cfg.beacon_offset_local, device=self.device, dtype=torch.float32)
        offset_w = self._target_offset_world(env_ids)
        self._beacon_pos_w[env_ids] = self._station_origin_w[env_ids] + beacon_local.unsqueeze(0) + offset_w
        write_beacon_pose(self.beacon, self._beacon_pos_w[env_ids], env_ids)

    def set_displacement_offset(
        self,
        dx: float,
        dy: float,
        dz: float = 0.0,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Debug: move robot. Lateral1D uses body-Y (sway), same frame as ``set_target_offset``."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if self._lateral_1d():
            dx, dy, dz = self._clamp_target_offset_b(dx, dy, dz)
            n = len(env_ids)
            zeros = torch.zeros(n, device=self.device)
            yaw = self._fixed_robot_yaw_w[env_ids]
            quat = quat_from_euler_xyz(zeros, zeros, yaw)
            offset_b = torch.zeros(n, 3, device=self.device)
            offset_b[:, 0] = dx
            offset_b[:, 1] = dy
            offset_b[:, 2] = dz
            self._kinematic_offset_w[env_ids] = quat_apply(quat, offset_b)
        else:
            dx, dy, dz = self._clamp_displacement_w(dx, dy, dz)
            self._kinematic_offset_w[env_ids, 0] = dx
            self._kinematic_offset_w[env_ids, 1] = dy
            self._kinematic_offset_w[env_ids, 2] = dz
        self.pipeline.feature_tracker.reset_temporal(env_ids)
        self.pipeline.invalidate_cache()
        self._hold_robot()

    def set_target_velocity(
        self,
        vx: float,
        vy: float,
        vz: float = 0.0,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Debug: constant world-frame target velocity (m/s)."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        vx, vy, vz = self._clamp_target_velocity_w(vx, vy, vz)
        self._beacon_vel_w[env_ids, 0] = vx
        self._beacon_vel_w[env_ids, 1] = vy
        self._beacon_vel_w[env_ids, 2] = vz

    def set_target_offset(
        self,
        dx: float,
        dy: float,
        dz: float = 0.0,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Debug: move target in **AUV body frame** (x=fwd, y=left, z=up) relative to teach pose."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        dx, dy, dz = self._clamp_target_offset_b(dx, dy, dz)
        self._target_kinematic_offset_b[env_ids, 0] = dx
        self._target_kinematic_offset_b[env_ids, 1] = dy
        self._target_kinematic_offset_b[env_ids, 2] = dz

        self._write_beacon_from_target_offset(env_ids)
        self._beacon_vel_w[env_ids] = 0.0
        self.pipeline.invalidate_cache()

        if bool(getattr(self.cfg, "freeze_robot_pose", False)):
            self._hold_robot()

    def set_smooth_target_offset(
        self,
        dx: float,
        dy: float,
        dz: float = 0.0,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Move target in body frame without zeroing velocity (smooth trajectories)."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        dx, dy, dz = self._clamp_target_offset_b(dx, dy, dz)
        self._target_kinematic_offset_b[env_ids, 0] = dx
        self._target_kinematic_offset_b[env_ids, 1] = dy
        self._target_kinematic_offset_b[env_ids, 2] = dz

        self._write_beacon_from_target_offset(env_ids)
        self.pipeline.invalidate_cache()

        if bool(getattr(self.cfg, "freeze_robot_pose", False)):
            self._hold_robot()

    def _station_pos_error_norm(self) -> torch.Tensor:
        return station_pos_error_norm(self._station_origin_w, self.robot.data.root_pos_w)

    def _ensure_pipeline(self) -> None:
        self.pipeline.ensure_updated(self.common_step_counter, self.oceansim_sonar, self.sim)

    def _setup_scene(self) -> None:
        self.robot = RigidObject(self.cfg.robot_cfg)
        self.beacon = RigidObject(self.cfg.beacon_cfg)
        self.beacon2 = RigidObject(self.cfg.beacon2_cfg) if self._use_second_beacon else None

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.rigid_objects["robot"] = self.robot
        self.scene.rigid_objects["target"] = self.beacon
        if self._use_second_beacon and self.beacon2 is not None:
            self.scene.rigid_objects["beacon2"] = self.beacon2

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
        )
        self._refresh_viewport_camera()

    def _refresh_viewport_camera(self) -> None:
        controller = getattr(self, "viewport_camera_controller", None)
        if controller is None:
            return
        viewer = self.cfg.viewer
        origin = str(getattr(viewer, "origin_type", "world"))
        try:
            if origin == "env":
                controller.set_view_env_index(int(getattr(viewer, "env_index", 0)))
                controller.update_view_to_env()
            else:
                controller.update_view_to_world()
            if self.sim.has_gui():
                self.sim.render()
        except Exception as exc:
            print(f"Warning: viewport camera refresh failed: {exc}")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)

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
        cmd = self._actions_for_physics()
        apply_velocity_action(
            self.robot,
            cmd,
            max_lin_vel=float(self.cfg.max_lin_vel),
            max_heave_vel=float(self.cfg.max_heave_vel),
            max_ang_vel=float(self.cfg.max_ang_vel),
            lateral_1d_only=self._lateral_1d(),
        )
        if self._lateral_1d():
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
            target_speed_max=float(getattr(self.cfg, "target_speed_max", 0.0)),
            dt=float(self.cfg.sim.dt * self.cfg.decimation),
            all_env_ids=self.robot._ALL_INDICES,
        )

    def _geometric_centroid_errors_px(self, *, corrected: bool = True) -> torch.Tensor:
        ocfg = self.cfg.oceansim_sonar_cfg
        raw = geometric_centroid_errors_px(
            self.pipeline.feature_tracker,
            beacon_pos_w=self._beacon_pos_w,
            robot_pos_w=self.robot.data.root_pos_w,
            robot_quat_w=self.robot.data.root_quat_w,
            sonar_mount_offset_b=self._sonar_mount_offset_b,
            min_range=float(ocfg.min_range),
            max_range=float(ocfg.max_range),
            polar_height=int(ocfg.height),
            polar_width=int(ocfg.width),
            hori_fov_deg=float(ocfg.hori_fov),
        )
        if (
            corrected
            and self._lateral_compact_obs
            and bool(getattr(self.cfg, "geometric_centroid_bias_correct", True))
        ):
            return raw - self._geometric_centroid_bias_px
        return raw

    def _calibrate_geometric_centroid_bias(self, env_ids: torch.Tensor) -> None:
        """Teach-pose bias: sparse teach centroid vs geometry at station origin."""
        if not self._lateral_compact_obs or not bool(
            getattr(self.cfg, "geometric_centroid_bias_correct", True)
        ):
            self._geometric_centroid_bias_px[env_ids] = 0.0
            return
        ocfg = self.cfg.oceansim_sonar_cfg
        n = len(env_ids)
        zeros = torch.zeros(n, device=self.device)
        yaw = self._fixed_robot_yaw_w[env_ids]
        quat = quat_from_euler_xyz(zeros, zeros, yaw)
        beacon_local = torch.tensor(
            self.cfg.beacon_offset_local, device=self.device, dtype=torch.float32
        )
        beacon_teach = self._station_origin_w[env_ids] + beacon_local.unsqueeze(0)
        self._geometric_centroid_bias_px[env_ids] = geometric_centroid_errors_px(
            self.pipeline.feature_tracker,
            beacon_pos_w=beacon_teach,
            robot_pos_w=self._station_origin_w[env_ids],
            robot_quat_w=quat,
            sonar_mount_offset_b=self._sonar_mount_offset_b[env_ids],
            min_range=float(ocfg.min_range),
            max_range=float(ocfg.max_range),
            polar_height=int(ocfg.height),
            polar_width=int(ocfg.width),
            hori_fov_deg=float(ocfg.hori_fov),
        )

    def _get_observations(self) -> dict:
        self._ensure_pipeline()
        if self._lateral_compact_obs:
            geom_err = self._geometric_centroid_errors_px()
            obs = self.obs_builder.build(
                self.pipeline.feature_tracker,
                self.robot,
                self.actions,
                centroid_err=geom_err,
            )
        else:
            obs = self.obs_builder.build(self.pipeline.feature_tracker, self.robot, self.actions)
        if bool(getattr(self.cfg, "print_ibvs_debug", False)) or bool(
            getattr(self.cfg, "print_flow_debug", False)
        ):
            maybe_print_ibvs_debug(
                step_counter=self.common_step_counter,
                interval=int(
                    getattr(self.cfg, "ibvs_debug_interval", None)
                    or getattr(self.cfg, "flow_debug_interval", 50)
                ),
                pipeline=self.pipeline,
                robot=self.robot,
                station_origin_w=self._station_origin_w,
                actions=self.actions,
                reference_mode=self.ref_mgr.mode,
                max_track_features=int(self.cfg.max_track_features),
                kinematic_offset_w=self._kinematic_offset_w,
                target_kinematic_offset_b=self._target_kinematic_offset_b,
                centroid_err_px=self._geometric_centroid_errors_px() if self._lateral_compact_obs else None,
                lateral_1d=self._lateral_1d(),
            )
        if not self._lateral_compact_obs:
            self.obs_builder.commit_depth(self.robot)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._ensure_pipeline()
        geom_err = self._geometric_centroid_errors_px() if self._lateral_compact_obs else None
        reward_fn = compute_reward_lateral_1d if self._lateral_1d() else compute_reward
        if self._lateral_1d():
            return compute_reward_lateral_1d(
                feature_tracker=self.pipeline.feature_tracker,
                actions=self.actions,
                depth=self.robot.data.root_pos_w[:, 2],
                w1_pixel=float(self.cfg.rew_w1_pixel),
                w2_action=float(self.cfg.rew_w2_action),
                w3_safety=float(self.cfg.rew_w3_safety),
                min_depth=float(self.cfg.min_depth),
                max_depth=float(self.cfg.max_depth),
                min_valid_features=int(self.cfg.min_valid_features),
                centroid_err=geom_err,
                lateral_reward_pixel_sq=bool(getattr(self.cfg, "lateral_reward_pixel_sq", False)),
            )
        return reward_fn(
            feature_tracker=self.pipeline.feature_tracker,
            actions=self.actions,
            depth=self.robot.data.root_pos_w[:, 2],
            w1_pixel=float(self.cfg.rew_w1_pixel),
            w2_action=float(self.cfg.rew_w2_action),
            w3_safety=float(self.cfg.rew_w3_safety),
            min_depth=float(self.cfg.min_depth),
            max_depth=float(self.cfg.max_depth),
            min_valid_features=int(self.cfg.min_valid_features),
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_pipeline()
        return self.term_checker.evaluate(
            cfg=self.cfg,
            feature_tracker=self.pipeline.feature_tracker,
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

        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

        self._target_kinematic_offset_b[env_ids_t] = 0.0
        self._lateral_target_phase[env_ids_t] = torch.rand(len(env_ids_t), device=self.device) * 2.0 * math.pi
        reset_beacon_and_station(
            self.cfg,
            env_ids_t,
            self.scene,
            self.beacon,
            self.beacon2,
            station_origin_w=self._station_origin_w,
            beacon_pos_w=self._beacon_pos_w,
            beacon2_pos_w=self._beacon2_pos_w,
            beacon_vel_w=self._beacon_vel_w,
            use_second_beacon=self._use_second_beacon,
        )
        reset_robot_episode(
            self.cfg,
            env_ids_t,
            self.robot,
            self.ref_mgr,
            self.pipeline,
            self.term_checker,
            self.obs_builder,
            station_origin_w=self._station_origin_w,
            beacon_pos_w=self._beacon_pos_w,
            kinematic_offset_w=self._kinematic_offset_w,
            current_vel_w=self._current_vel_w,
            actions=self.actions,
            oceansim_sonar=self.oceansim_sonar,
        )
        self.pipeline.invalidate_cache()
        self._ensure_pipeline()
        self.ref_mgr.apply_on_reset(
            env_ids_t,
            self.pipeline,
            robot=self.robot,
            beacon_pos_w=self._beacon_pos_w,
            sonar_mount_offset_b=self._sonar_mount_offset_b,
            max_range=self.pipeline.max_range,
        )
        if self.ref_mgr.file_ref_yaw_valid:
            self._fixed_robot_yaw_w[env_ids_t] = self.ref_mgr._file_ref_yaw[0].expand(len(env_ids_t))
        else:
            _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w[env_ids_t])
            self._fixed_robot_yaw_w[env_ids_t] = yaw
        if self._lateral_1d():
            enforce_lateral_1d_kinematics(
                self.robot,
                self._station_origin_w,
                self._fixed_robot_yaw_w,
                env_ids_t,
            )
        if bool(getattr(self.cfg, "freeze_robot_pose", False)):
            self._hold_robot(env_ids_t)
        if self._lateral_compact_obs:
            self._calibrate_geometric_centroid_bias(env_ids_t)

    def close(self) -> None:
        if hasattr(self, "oceansim_sonar"):
            self.oceansim_sonar.close()
        super().close()


SonarFlowHoverEnv = SonarIbvsDrlEnv
