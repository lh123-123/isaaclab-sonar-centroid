# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""P (+ optional feedforward) sway controller for 1D lateral sonar IBVS."""

from __future__ import annotations

from typing import Literal

import torch

from ..sparse_feature_tracker import SparseFeatureTracker
from .beacon_cart import (
    beacon_lateral_error_norm,
    estimate_beacon_x,
    teach_beacon_centroid,
)

ControlMode = Literal["hybrid", "beacon_cart", "beacon_ibvs", "ibvs_only"]


def mean_horizontal_pixel_error(feature_tracker: SparseFeatureTracker, env_id: int = 0) -> float:
    """Mean signed lateral IBVS error: cart ex or polar Δaz (normalized)."""
    return float(feature_tracker.mean_lateral_error_signed()[env_id].item())


class Lateral1DPController:
    """Map sonar IBVS / beacon-cart error (+ optional geo FF) to sway in [-1, 1]."""

    def __init__(
        self,
        *,
        kp: float = 3.0,
        kp_geo: float = 3.0,
        kp_beacon: float = 4.0,
        ki_beacon: float = 0.8,
        kff: float = 0.5,
        action_smooth: float = 0.25,
        control_mode: ControlMode = "hybrid",
        peak_threshold: float = 0.10,
        min_valid_gate: int = 2,
        integral_clip: float = 0.5,
        step_dt: float = 0.1,
        device: str | torch.device = "cpu",
    ):
        self.kp = float(kp)
        self.kp_geo = float(kp_geo)
        self.kp_beacon = float(kp_beacon)
        self.ki_beacon = float(ki_beacon)
        self.kff = float(kff)
        self.action_smooth = float(max(min(action_smooth, 1.0), 0.0))
        self.control_mode: ControlMode = control_mode
        self.peak_threshold = float(peak_threshold)
        self.min_valid_gate = max(int(min_valid_gate), 1)
        self.integral_clip = max(float(integral_clip), 1.0e-3)
        self.step_dt = max(float(step_dt), 1.0e-6)
        self.device = device
        self._prev_sway = torch.zeros(1, device=device)
        self._ref_beacon_x = 0.0
        self._ref_beacon_y = 0.0
        self._last_beacon_err_norm = 0.0
        self._integral_err = 0.0
        self._last_debug: dict[str, float] = {}

    @property
    def last_debug(self) -> dict[str, float]:
        return dict(self._last_debug)

    def reset(self) -> None:
        self._prev_sway.zero_()
        self._ref_beacon_x = 0.0
        self._ref_beacon_y = 0.0
        self._last_beacon_err_norm = 0.0
        self._integral_err = 0.0
        self._last_debug = {}

    @staticmethod
    def _sensor_trust(valid_pts: int, min_valid: int, beacon_src: str) -> tuple[float, float]:
        """Scale IBVS / beacon P terms when features are weak (no pose GT fallback)."""
        if beacon_src == "lost":
            return 0.0, 0.25
        if valid_pts >= min_valid:
            return 1.0, 1.0
        ratio = max(float(valid_pts) / float(min_valid), 0.0)
        return ratio, max(ratio, 0.35)

    def latch_teach_setpoint(self, feature_tracker: SparseFeatureTracker, env_id: int = 0) -> None:
        """Call after env reset: fix cart-pixel setpoint from teach features."""
        self._ref_beacon_x, self._ref_beacon_y = teach_beacon_centroid(feature_tracker, env_id)

    def compute(
        self,
        feature_tracker: SparseFeatureTracker,
        *,
        cart_image: torch.Tensor | None = None,
        robot_lat_m: float = 0.0,
        target_dy_b: float = 0.0,
        target_dy_vel_b: float = 0.0,
        lateral_amp_m: float = 1.5,
        env_id: int = 0,
        action_shape: torch.Size | tuple[int, ...] = (1, 1),
    ) -> torch.Tensor:
        actions, _ = self.compute_with_debug(
            feature_tracker,
            cart_image=cart_image,
            robot_lat_m=robot_lat_m,
            target_dy_b=target_dy_b,
            target_dy_vel_b=target_dy_vel_b,
            lateral_amp_m=lateral_amp_m,
            env_id=env_id,
            action_shape=action_shape,
        )
        return actions

    def compute_with_debug(
        self,
        feature_tracker: SparseFeatureTracker,
        *,
        cart_image: torch.Tensor | None = None,
        robot_lat_m: float = 0.0,
        target_dy_b: float = 0.0,
        target_dy_vel_b: float = 0.0,
        lateral_amp_m: float = 1.5,
        env_id: int = 0,
        action_shape: torch.Size | tuple[int, ...] = (1, 1),
    ) -> tuple[torch.Tensor, dict[str, float]]:
        actions = torch.zeros(action_shape, device=self.device)
        mean_ex = mean_horizontal_pixel_error(feature_tracker, env_id)
        amp = max(float(lateral_amp_m), 1.0e-3)
        diagonal = float(feature_tracker.diagonal)

        err = feature_tracker.feature_errors()[env_id]
        valid = feature_tracker.valid[env_id]
        ex_vals: list[float] = []
        if bool(valid.any().item()):
            ex_norm = err[valid, 0] / diagonal
            ex_vals = [float(v.item()) for v in ex_norm]

        beacon_x = self._ref_beacon_x
        beacon_y = self._ref_beacon_y
        beacon_src = "teach"
        beacon_err_norm = mean_ex
        if self.control_mode in ("beacon_cart", "beacon_ibvs") and cart_image is not None:
            beacon_x, beacon_y, beacon_src = estimate_beacon_x(
                feature_tracker,
                cart_image[env_id] if cart_image.ndim == 3 else cart_image,
                env_id,
                ref_beacon_xy=(self._ref_beacon_x, self._ref_beacon_y),
                peak_threshold=self.peak_threshold,
            )
            beacon_err_norm = beacon_lateral_error_norm(beacon_x, self._ref_beacon_x, diagonal)
            if beacon_src == "lost":
                beacon_err_norm = self._last_beacon_err_norm
            else:
                self._last_beacon_err_norm = beacon_err_norm

        lat_track_err = float(target_dy_b) - float(robot_lat_m)
        use_geo = self.control_mode == "hybrid" and abs(self.kp_geo) > 1.0e-8
        use_ibvs = self.control_mode in ("hybrid", "ibvs_only", "beacon_ibvs") and abs(self.kp) > 1.0e-8
        use_beacon = self.control_mode in ("beacon_cart", "beacon_ibvs") and abs(self.kp_beacon) > 1.0e-8
        use_integral = self.control_mode in ("beacon_cart", "beacon_ibvs") and abs(self.ki_beacon) > 1.0e-8

        valid_pts = int(valid.sum().item())
        trust_ibvs, trust_beacon = self._sensor_trust(valid_pts, self.min_valid_gate, str(beacon_src))

        p_ibvs = self.kp * mean_ex * trust_ibvs if use_ibvs else 0.0
        p_beacon = self.kp_beacon * beacon_err_norm * trust_beacon if use_beacon else 0.0
        p_geo = self.kp_geo * lat_track_err / amp if use_geo else 0.0

        i_beacon = 0.0
        if use_integral:
            if self.control_mode == "beacon_ibvs" and valid_pts > 0:
                servo_err = 0.5 * beacon_err_norm + 0.5 * mean_ex
            else:
                servo_err = beacon_err_norm
            if beacon_src != "lost":
                self._integral_err += servo_err * self.step_dt
                self._integral_err = max(
                    min(self._integral_err, self.integral_clip),
                    -self.integral_clip,
                )
            i_beacon = self.ki_beacon * self._integral_err

        ff_pos = 0.0
        ff_vel = 0.0
        ff_scale = 1.0 if valid_pts >= self.min_valid_gate else 1.35
        if abs(self.kff) > 1.0e-8:
            if self.control_mode == "ibvs_only" and abs(self.kp_geo) <= 1.0e-8:
                ff_pos = self.kff * float(target_dy_b) / amp
            ff_vel = ff_scale * self.kff * 0.15 * float(target_dy_vel_b) / amp
        cmd_raw = p_ibvs + p_beacon + p_geo + i_beacon + ff_pos + ff_vel

        sway = max(min(cmd_raw, 1.0), -1.0)
        sway_pre_smooth = sway
        if self.action_smooth > 0.0:
            sway = self.action_smooth * sway + (1.0 - self.action_smooth) * float(self._prev_sway.item())
            sway = max(min(sway, 1.0), -1.0)
        self._prev_sway[0] = sway

        if actions.shape[-1] == 1:
            actions[..., 0] = sway
        else:
            actions[..., 1] = sway

        debug = {
            "mean_ex": mean_ex,
            "beacon_err_norm": beacon_err_norm,
            "beacon_x": beacon_x,
            "beacon_y": beacon_y,
            "ref_beacon_x": self._ref_beacon_x,
            "beacon_src": beacon_src,
            "valid_pts": float(valid.sum().item()),
            "ex_vals": ex_vals,
            "robot_lat_m": float(robot_lat_m),
            "target_dy_m": float(target_dy_b),
            "lat_track_err": lat_track_err,
            "target_dy_vel_mps": float(target_dy_vel_b),
            "p_ibvs": p_ibvs,
            "p_beacon": p_beacon,
            "p_geo": p_geo,
            "i_beacon": i_beacon,
            "integral_err": self._integral_err,
            "trust_ibvs": trust_ibvs,
            "trust_beacon": trust_beacon,
            "ff_scale": ff_scale,
            "p_term": p_ibvs + p_beacon + p_geo + i_beacon,
            "ff_pos": ff_pos,
            "ff_vel": ff_vel,
            "cmd_raw": cmd_raw,
            "sway_pre_smooth": sway_pre_smooth,
            "sway": sway,
            "control_mode": self.control_mode,
        }
        self._last_debug = debug
        return actions, debug


def format_p_control_debug(
    *,
    step: int,
    ctrl_dbg: dict[str, float],
    robot_lat_m: float,
    target_dy_m: float,
    pixel_err: float,
    pos_err_m: float,
    vel_body_y_mps: float,
    valid_pts: int,
) -> str:
    """One-line + multi-field P-control debug for lateral IBVS."""
    lat_err = robot_lat_m - target_dy_m
    ex_str = ",".join(f"{v:+.3f}" for v in ctrl_dbg.get("ex_vals", [])) or "none"
    mode = str(ctrl_dbg.get("control_mode", "hybrid"))
    if mode in ("beacon_cart", "beacon_ibvs"):
        ibvs_str = f" mean_ex={ctrl_dbg['mean_ex']:+.4f}" if mode == "beacon_ibvs" else ""
        return (
            f"[p-dbg] step={step:4d} mode={mode} "
            f"beacon_x={ctrl_dbg['beacon_x']:.1f} ref_x={ctrl_dbg['ref_beacon_x']:.1f} "
            f"err_norm={ctrl_dbg['beacon_err_norm']:+.4f} src={ctrl_dbg['beacon_src']} "
            f"{ibvs_str} lat_err={lat_err:+.3f} pixel_err={pixel_err:.4f} valid={valid_pts} "
            f"P_bcn={ctrl_dbg['p_beacon']:+.3f} P_ibvs={ctrl_dbg['p_ibvs']:+.3f} "
            f"I={ctrl_dbg.get('i_beacon', 0.0):+.3f} tr={ctrl_dbg.get('trust_ibvs', 1.0):.2f}/"
            f"{ctrl_dbg.get('trust_beacon', 1.0):.2f} "
            f"FFv={ctrl_dbg['ff_vel']:+.3f} cmd={ctrl_dbg['cmd_raw']:+.3f} sway={ctrl_dbg['sway']:+.3f}"
        )
    return (
        f"[p-dbg] step={step:4d} "
        f"target_dy={target_dy_m:+.3f} robot_lat={robot_lat_m:+.3f} lat_err={lat_err:+.3f} "
        f"mean_ex={ctrl_dbg['mean_ex']:+.4f} ex=[{ex_str}] valid={valid_pts} "
        f"pixel_err={pixel_err:.4f} pos_err={pos_err_m:.3f}m vy_b={vel_body_y_mps:+.3f} "
        f"P_ibvs={ctrl_dbg['p_ibvs']:+.3f} P_geo={ctrl_dbg['p_geo']:+.3f} "
        f"FFv={ctrl_dbg['ff_vel']:+.3f} cmd={ctrl_dbg['cmd_raw']:+.3f} sway={ctrl_dbg['sway']:+.3f}"
    )
