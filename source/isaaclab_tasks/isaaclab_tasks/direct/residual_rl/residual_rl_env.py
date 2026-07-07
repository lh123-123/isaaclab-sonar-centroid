# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""WarpAUV environment with PD base controller + SAC residual compensation."""

from __future__ import annotations

import importlib
from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_apply, quat_conjugate

from .base_controller import BaseController
from .residual_rl_env_cfg import ResidualRLEnvCfg
from .reward import compute_residual_rewards

_warpauv = importlib.import_module("isaaclab_tasks.direct.isaac-auv-env.warpauv_env")
WarpAUVEnv = _warpauv.WarpAUVEnv


class ResidualRLEnv(WarpAUVEnv):
    """Residual RL on imperfect PD.

    wrench mode (default): ``w_total = clip(w_pd + clip(Δw_rl, ±δ_w))``, then allocate PWM.
    thruster mode (legacy): ``u_total = clip(u_base + clip(u_rl, ±δ))``.
    """

    cfg: ResidualRLEnvCfg

    def __init__(self, cfg: ResidualRLEnvCfg, render_mode: str | None = None, **kwargs):
        # Buffers must exist before super().__init__() — parent reset calls _reset_domain().
        n_envs = cfg.scene.num_envs
        dev = cfg.sim.device
        n_act = cfg.num_actions
        self._u_rl = torch.zeros(n_envs, n_act, device=dev)
        self._u_base = torch.zeros(n_envs, n_act, device=dev)
        self._prev_u_rl = torch.zeros(n_envs, n_act, device=dev)
        self._w_base = torch.zeros(n_envs, 6, device=dev)
        self._dw_rl = torch.zeros(n_envs, 6, device=dev)
        self._prev_dw_rl = torch.zeros(n_envs, 6, device=dev)
        self._thruster_gain = torch.ones(n_envs, n_act, device=dev)
        self._thruster_bias = torch.zeros(n_envs, n_act, device=dev)
        self._current_vel_w = torch.zeros(n_envs, 3, device=dev)
        self._added_mass_scale = torch.ones(n_envs, 1, device=dev)
        self._target_pos_w = torch.zeros(n_envs, 3, device=dev)
        self._eval_steps = 0
        self._eval_pos_err_sum = 0.0
        self._eval_ang_err_sum = 0.0
        self._eval_reward_sum = 0.0
        self._eval_u_rl_sum = 0.0
        self._eval_success = False
        self._eval_last_terminated = False
        self._success_bonus_pending = torch.ones(n_envs, dtype=torch.bool, device=dev)
        self._prev_pos_err_norm = torch.zeros(n_envs, device=dev)

        super().__init__(cfg, render_mode, **kwargs)

        self.base_controller = BaseController(
            self.num_envs,
            self.device,
            imperfect=cfg.base_controller_imperfect,
        )

    def _eval_enabled(self) -> bool:
        return bool(getattr(self.cfg, "print_eval_debug", False))

    def _reset_eval_stats(self) -> None:
        self._eval_steps = 0
        self._eval_pos_err_sum = 0.0
        self._eval_ang_err_sum = 0.0
        self._eval_reward_sum = 0.0
        self._eval_u_rl_sum = 0.0
        self._eval_success = False

    def _position_error_norm(self) -> torch.Tensor:
        pos_error_w = self._target_pos_w - self._robot.data.root_pos_w
        return torch.norm(pos_error_w, dim=1)

    def _pos_error_body(self) -> torch.Tensor:
        return quat_apply(
            quat_conjugate(self._robot.data.root_quat_w),
            self._target_pos_w - self._robot.data.root_pos_w,
        )

    def _apply_point_reach_reset(self, env_ids: Sequence[int]) -> None:
        """Fix spawn and target for Test 1 point reaching."""
        if isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids
        else:
            env_ids_t = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

        n = len(env_ids_t)
        start_local = torch.tensor(self.cfg.test1_start_pos, device=self.device, dtype=torch.float32)
        target_local = torch.tensor(self.cfg.test1_target_pos, device=self.device, dtype=torch.float32)
        noise_r = float(self.cfg.test1_start_pos_noise)

        roll, pitch, yaw = self.cfg.test1_target_rpy
        rs = torch.full((n,), roll, device=self.device)
        ps = torch.full((n,), pitch, device=self.device)
        ys = torch.full((n,), yaw, device=self.device)
        target_quat = math_utils.quat_from_euler_xyz(rs, ps, ys)

        origins = self.scene.env_origins[env_ids_t]
        if noise_r > 0.0:
            start_offset = self._sample_from_sphere(n, noise_r)
        else:
            start_offset = torch.zeros(n, 3, device=self.device)

        spawn_pos = origins + start_local.unsqueeze(0) + start_offset
        target_pos = origins + target_local.unsqueeze(0)

        self._target_pos_w[env_ids_t] = target_pos
        self._default_env_origins[env_ids_t] = origins + start_local.unsqueeze(0)
        self._goal[env_ids_t, 0:4] = target_quat
        self._goal_pos_w[env_ids_t] = target_pos

        self._default_root_state[env_ids_t, :3] = spawn_pos
        self._default_root_state[env_ids_t, 3:7] = target_quat
        self._default_root_state[env_ids_t, 7:] = 0.0

        self._robot.write_root_pose_to_sim(self._default_root_state[env_ids_t, :7], env_ids_t)
        self._robot.write_root_velocity_to_sim(self._default_root_state[env_ids_t, 7:], env_ids_t)

    def _sync_target_pos(self, env_ids: Sequence[int]) -> None:
        if getattr(self.cfg, "point_reach_mode", False):
            self._apply_point_reach_reset(env_ids)
        else:
            self._target_pos_w[env_ids] = self._default_env_origins[env_ids]

    def _update_eval_stats(self, reward: torch.Tensor) -> None:
        if not self._eval_enabled():
            return
        env_idx = 0
        pos_err = float(self._position_error_norm()[env_idx].item())
        ang_err = float(
            math_utils.quat_error_magnitude(
                self._goal[env_idx : env_idx + 1],
                self._robot.data.root_quat_w[env_idx : env_idx + 1],
            ).item()
        )
        self._eval_steps += 1
        self._eval_pos_err_sum += pos_err
        self._eval_ang_err_sum += ang_err
        self._eval_reward_sum += float(reward[env_idx].item())
        self._eval_u_rl_sum += float(self._residual_norm(env_idx).item())
        if pos_err < self.cfg.success_pos_threshold:
            self._eval_success = True

    def _print_eval_summary(self, terminated: bool) -> dict[str, float]:
        if not self._eval_enabled() or self._eval_steps == 0:
            return {}
        steps = self._eval_steps
        summary = {
            "steps": float(steps),
            "mean_pos_err_m": self._eval_pos_err_sum / steps,
            "mean_ang_err_rad": self._eval_ang_err_sum / steps,
            "final_pos_err_m": float(self._position_error_norm()[0].item()),
            "total_reward": self._eval_reward_sum,
            "mean_u_rl_norm": self._eval_u_rl_sum / steps,
            "success": float(self._eval_success),
            "terminated": float(terminated),
        }
        if not bool(getattr(self.cfg, "eval_silent", False)):
            tag = "terminated" if terminated else "timeout"
            print(
                f"[ResidualRL eval] {tag} "
                f"steps={steps:.0f} mean_pos_err={summary['mean_pos_err_m']:.3f}m "
                f"final_pos_err={summary['final_pos_err_m']:.3f}m "
                f"mean_ang_err={summary['mean_ang_err_rad']:.3f}rad "
                f"mean|u_rl|={summary['mean_u_rl_norm']:.3f} "
                f"success={bool(self._eval_success)} reward={summary['total_reward']:+.2f}",
                flush=True,
            )
        return summary

    def _wrench_mode(self) -> bool:
        return getattr(self.cfg, "residual_mode", "wrench") == "wrench"

    def _residual_for_reward(self) -> torch.Tensor:
        # Wrench mode: penalize Δw (action space). Thruster mode: penalize u_rl directly.
        return self._dw_rl if self._wrench_mode() else self._u_rl

    def _wrench_opposition(self) -> torch.Tensor:
        """Sum of per-component relu(-Δw * w_pd); zero in thruster mode."""
        if not self._wrench_mode():
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(torch.clamp(-self._dw_rl * self._w_base, min=0.0), dim=1)

    def _residual_norm(self, env_idx: int) -> torch.Tensor:
        if self._wrench_mode():
            return torch.norm(self._dw_rl[env_idx])
        return torch.norm(self._u_rl[env_idx])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply bounded residual on top of base PD controller."""
        pd_args = (
            self._target_pos_w,
            self._robot.data.root_pos_w,
            self._robot.data.root_lin_vel_b,
            self._goal,
            self._robot.data.root_quat_w,
            self._robot.data.root_ang_vel_b,
        )
        gain, bias = self._thruster_gain, self._thruster_bias

        if self._wrench_mode():
            self._dw_rl = torch.clamp(actions, -self.cfg.delta_wrench, self.cfg.delta_wrench)
            self._w_base = self.base_controller.compute_wrench(*pd_args)
            w_total = self._w_base + self._dw_rl
            self._u_base = self.base_controller.allocate_thrusters_from_wrench(self._w_base, gain, bias)
            self._actions[:] = self.base_controller.allocate_thrusters_from_wrench(w_total, gain, bias)
            self._u_rl = self._actions - self._u_base
            self._prev_dw_rl = self._dw_rl.clone()
            self._prev_u_rl = self._u_rl.clone()
        else:
            self._u_rl = torch.clamp(actions, -self.cfg.delta, self.cfg.delta)
            self._u_base = self.base_controller.compute(*pd_args, thruster_gain=gain, thruster_bias=bias)
            self._actions[:] = torch.clamp(self._u_base + self._u_rl, -1.0, 1.0)
            self._prev_u_rl = self._u_rl.clone()
            self._dw_rl.zero_()
            self._w_base.zero_()

    def _apply_action(self) -> None:
        """Apply thruster forces plus optional water-current disturbance."""
        self._thrust[:, 0, :], self._moment[:, 0, :] = self._compute_dynamics(self._actions)

        # Water current drag: approximate as added force opposing relative velocity.
        rel_vel = self._robot.data.root_lin_vel_w - self._current_vel_w
        drag_coeff = 5.0 * self._added_mass_scale
        current_force = -drag_coeff * rel_vel
        self._thrust[:, 0, :] = self._thrust[:, 0, :] + current_force

        self._robot.set_external_force_and_torque(self._thrust, self._moment)

    def _get_observations(self) -> dict:
        offset_to_target_b = self._pos_error_body()
        pos_scale = float(getattr(self.cfg, "obs_pos_scale", 5.0))
        vel_scale = float(getattr(self.cfg, "obs_vel_scale", 2.0))

        if self._wrench_mode():
            prev_residual = self._prev_dw_rl / float(getattr(self.cfg, "obs_wrench_scale", 2.0))
        else:
            prev_residual = self._prev_u_rl

        obs = torch.cat(
            [
                self._goal,
                offset_to_target_b / pos_scale,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_b / vel_scale,
                self._robot.data.root_ang_vel_b / vel_scale,
                prev_residual,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _overshoot_x(self) -> torch.Tensor:
        """World-frame +X overshoot past the Test1 target (meters)."""
        if not getattr(self.cfg, "point_reach_mode", False):
            return torch.zeros(self.num_envs, device=self.device)
        robot_x = self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]
        target_x = float(self.cfg.test1_target_pos[0])
        return torch.clamp(robot_x - target_x, min=0.0)

    def _get_rewards(self) -> torch.Tensor:
        pos_error_b = self._pos_error_body()
        pos_err_norm = torch.norm(pos_error_b, dim=1)
        pos_err_norm_sq = pos_err_norm * pos_err_norm
        lin_vel_norm_sq = torch.sum(self._robot.data.root_lin_vel_b * self._robot.data.root_lin_vel_b, dim=1)
        ang_err = math_utils.quat_error_magnitude(self._goal, self._robot.data.root_quat_w)
        is_timeout = self.episode_length_buf >= self.max_episode_length - 1

        reward = compute_residual_rewards(
            self.cfg.rew_scale_pos,
            self.cfg.rew_scale_pos_lin,
            self.cfg.rew_scale_ang,
            self.cfg.rew_scale_u_rl,
            self.cfg.rew_scale_progress,
            self.cfg.rew_scale_vel_near,
            self.cfg.rew_scale_overshoot,
            self.cfg.rew_scale_success,
            self.cfg.rew_scale_terminal,
            self.cfg.success_pos_threshold,
            self.cfg.near_goal_dist,
            is_timeout,
            self._prev_pos_err_norm,
            pos_err_norm,
            pos_err_norm_sq,
            self._overshoot_x(),
            lin_vel_norm_sq,
            ang_err,
            self._residual_for_reward(),
            self._wrench_opposition(),
            float(getattr(self.cfg, "rew_scale_wrench_opp", 0.0)),
            self._success_bonus_pending,
        )

        success_now = (pos_err_norm < self.cfg.success_pos_threshold) & self._success_bonus_pending
        self._success_bonus_pending[success_now] = False
        self._prev_pos_err_norm = pos_err_norm.clone()
        reward = torch.clamp(reward, self.cfg.rew_clip_min, self.cfg.rew_clip_max)
        self._update_eval_stats(reward)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        out_of_bounds, time_out = super()._get_dones()
        if self._eval_enabled():
            self._eval_last_terminated = bool(out_of_bounds[0].item())
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        if self._eval_enabled() and 0 in env_ids and self._eval_steps > 0:
            self._print_eval_summary(self._eval_last_terminated)
        if 0 in env_ids and getattr(self.cfg, "point_reach_mode", False):
            pos_w = self._robot.data.root_pos_w[0]
            vel_w = self._robot.data.root_lin_vel_w[0]
            origin = self.scene.env_origins[0]
            local = pos_w - origin
            self.extras["episode_end"] = {
                "x": float(local[0]),
                "y": float(local[1]),
                "z": float(local[2]),
                "vx": float(vel_w[0]),
                "vy": float(vel_w[1]),
                "vz": float(vel_w[2]),
                "pos_err_m": float(self._position_error_norm()[0].item()),
                "ang_err_rad": float(
                    math_utils.quat_error_magnitude(
                        self._goal[0:1], self._robot.data.root_quat_w[0:1]
                    ).item()
                ),
                "overshoot_x_m": float(self._overshoot_x()[0].item()),
                "residual_mode": self.cfg.residual_mode,
                "residual_norm": float(self._residual_norm(0).item()),
                "u_rl_norm": float(torch.norm(self._u_rl[0]).item()),
                "dw_rl_norm": float(torch.norm(self._dw_rl[0]).item()),
                "w_base_norm": float(torch.norm(self._w_base[0]).item()),
                "u_base_norm": float(torch.norm(self._u_base[0]).item()),
                "u_total_norm": float(torch.norm(self._actions[0]).item()),
                **{f"u_rl_{i}": float(self._u_rl[0, i]) for i in range(6)},
                **{f"u_base_{i}": float(self._u_base[0, i]) for i in range(6)},
                **{f"dw_rl_{i}": float(self._dw_rl[0, i]) for i in range(6)},
            }
        super()._reset_idx(env_ids)
        self._sync_target_pos(env_ids if env_ids is not None else self._robot._ALL_INDICES)
        if self._eval_enabled() and 0 in env_ids:
            self._reset_eval_stats()
        self._prev_u_rl[env_ids] = 0.0
        self._prev_dw_rl[env_ids] = 0.0
        self._u_rl[env_ids] = 0.0
        self._dw_rl[env_ids] = 0.0
        self._w_base[env_ids] = 0.0
        self._u_base[env_ids] = 0.0
        self._success_bonus_pending[env_ids] = True
        self._prev_pos_err_norm[env_ids] = self._position_error_norm()[env_ids]

    def _reset_domain(self, env_ids: Sequence[int]):
        """Apply WarpAUV DR plus curriculum-gated residual randomization."""
        super()._reset_domain(env_ids)

        stage = self.cfg.curriculum_stage
        dr = self.cfg.residual_domain_randomization

        self._thruster_gain[env_ids] = 1.0
        self._thruster_bias[env_ids] = 0.0
        self._current_vel_w[env_ids] = 0.0
        self._added_mass_scale[env_ids] = 1.0

        if stage >= 2:
            bias_lo, bias_hi = dr.thruster_bias_range
            gain_lo, gain_hi = dr.thruster_gain_range
            self._thruster_bias[env_ids] = math_utils.sample_uniform(
                bias_lo, bias_hi, (len(env_ids), self.cfg.num_actions), self.device
            )
            self._thruster_gain[env_ids] = math_utils.sample_uniform(
                gain_lo, gain_hi, (len(env_ids), self.cfg.num_actions), self.device
            )

        if stage >= 3:
            speed_lo, speed_hi = dr.current_speed_range
            speeds = math_utils.sample_uniform(speed_lo, speed_hi, (len(env_ids), 1), self.device)
            thetas = math_utils.sample_uniform(0.0, 2.0 * 3.14159, (len(env_ids), 1), self.device)
            self._current_vel_w[env_ids, 0] = (speeds * torch.cos(thetas)).squeeze(-1)
            self._current_vel_w[env_ids, 1] = (speeds * torch.sin(thetas)).squeeze(-1)
            self._current_vel_w[env_ids, 2] = 0.0

        if stage >= 4:
            mass_lo, mass_hi = dr.added_mass_scale_range
            self._added_mass_scale[env_ids] = math_utils.sample_uniform(
                mass_lo, mass_hi, (len(env_ids), 1), self.device
            )

            vol_lower, vol_upper = self.cfg.domain_randomization.volume_range
            self.volumes[env_ids] = math_utils.sample_uniform(
                vol_lower, vol_upper, self.volumes[env_ids].shape, self.device
            )

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "target_frame_visualizer"):
                s = float(getattr(self.cfg, "target_frame_scale", 0.25))
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.markers["frame"].scale = (s, s, s)
                marker_cfg.prim_path = "/Visuals/Command/target_frame"
                self.target_frame_visualizer = VisualizationMarkers(marker_cfg)
            self.target_frame_visualizer.set_visibility(True)
        elif hasattr(self, "target_frame_visualizer"):
            self.target_frame_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        super()._debug_vis_callback(event)
        if not hasattr(self, "target_frame_visualizer"):
            return
        if not self._robot.is_initialized:
            return
        # RGB triad at target pose (same style as Isaac Lab frame marker on the robot).
        self.target_frame_visualizer.visualize(self._target_pos_w, self._goal)
        # Keep goal cuboid on target (parent draws at _goal_pos_w; sync each frame).
        if hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.visualize(translations=self._target_pos_w)
