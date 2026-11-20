#!/usr/bin/env python3
"""Yaw feedback fallback for IMUs whose absolute Euler yaw can stall."""

from __future__ import annotations

import math


YAW_EULER_STALE_SECONDS = 0.30
YAW_EULER_STALE_RATE_THRESHOLD_DPS = 0.50
YAW_EULER_SLOW_RATE_THRESHOLD_DPS = 0.12
YAW_EULER_REANCHOR_DEG = 0.50
YAW_INTEGRATION_MAX_DT_SECONDS = 0.20


class YawFeedbackEstimator:
    """Bridge a stalled absolute yaw with short-term body-rate integration."""

    def __init__(
        self,
        stale_seconds=YAW_EULER_STALE_SECONDS,
        rate_threshold_dps=YAW_EULER_STALE_RATE_THRESHOLD_DPS,
        slow_rate_threshold_dps=YAW_EULER_SLOW_RATE_THRESHOLD_DPS,
        reanchor_deg=YAW_EULER_REANCHOR_DEG,
    ):
        self.stale_seconds = float(stale_seconds)
        self.rate_threshold_dps = float(rate_threshold_dps)
        self.slow_rate_threshold_dps = float(slow_rate_threshold_dps)
        self.reanchor_deg = float(reanchor_deg)
        self.yaw_deg = None
        self._last_absolute_yaw_deg = None
        self._last_absolute_change_at = None
        self._last_update_at = None
        self._fallback_anchor_deg = None
        self._rate_motion_since = None
        self._rate_motion_sign = 0
        self.fallback_active = False

    @staticmethod
    def _wrap(angle):
        return (float(angle) + 180.0) % 360.0 - 180.0

    def reset(self, absolute_yaw_deg, now):
        yaw = float(absolute_yaw_deg)
        timestamp = float(now)
        if not math.isfinite(yaw) or not math.isfinite(timestamp):
            raise ValueError("yaw feedback reset values must be finite")
        self.yaw_deg = self._wrap(yaw)
        self._last_absolute_yaw_deg = self.yaw_deg
        self._last_absolute_change_at = timestamp
        self._last_update_at = timestamp
        self._fallback_anchor_deg = None
        self._rate_motion_since = None
        self._rate_motion_sign = 0
        self.fallback_active = False
        return self.yaw_deg, self.fallback_active

    def update(self, absolute_yaw_deg, gyro_z_dps, now):
        absolute_yaw = float(absolute_yaw_deg)
        gyro_z = float(gyro_z_dps)
        timestamp = float(now)
        if not all(math.isfinite(value) for value in (
            absolute_yaw, gyro_z, timestamp
        )):
            raise ValueError("yaw feedback values must be finite")
        absolute_yaw = self._wrap(absolute_yaw)
        if self.yaw_deg is None:
            return self.reset(absolute_yaw, timestamp)

        previous_update = self._last_update_at
        dt = 0.0 if previous_update is None else max(
            0.0,
            min(timestamp - previous_update, YAW_INTEGRATION_MAX_DT_SECONDS),
        )
        previous_absolute = self._last_absolute_yaw_deg
        absolute_delta = (
            0.0
            if previous_absolute is None
            else self._wrap(absolute_yaw - previous_absolute)
        )
        if self._last_absolute_change_at is None:
            self._last_absolute_change_at = timestamp

        # Very slow yaw can be quantized away by the IMU Euler report. Require
        # the small gyro rate to persist for the stale window so static gyro
        # noise does not create a false turn, while still making slow motion
        # visible in the operator telemetry.
        if abs(gyro_z) >= self.slow_rate_threshold_dps:
            sign = 1 if gyro_z > 0.0 else -1
            if sign != self._rate_motion_sign:
                self._rate_motion_sign = sign
                self._rate_motion_since = timestamp
        else:
            self._rate_motion_sign = 0
            self._rate_motion_since = None
        slow_rate_sustained = bool(
            self._rate_motion_since is not None
            and timestamp - self._rate_motion_since >= self.stale_seconds
        )

        if not self.fallback_active:
            if abs(absolute_delta) >= self.reanchor_deg * 0.2:
                self.yaw_deg = absolute_yaw
                self._last_absolute_change_at = timestamp
            elif (
                (
                    abs(gyro_z) >= self.rate_threshold_dps
                    and timestamp - self._last_absolute_change_at
                    >= self.stale_seconds
                )
                or slow_rate_sustained
            ):
                self.fallback_active = True
                self._fallback_anchor_deg = absolute_yaw
                self.yaw_deg = self._wrap(self.yaw_deg + gyro_z * dt)
            else:
                self.yaw_deg = absolute_yaw
        elif abs(self._wrap(absolute_yaw - self._fallback_anchor_deg)) >= self.reanchor_deg:
            self.yaw_deg = absolute_yaw
            self._last_absolute_change_at = timestamp
            self._fallback_anchor_deg = None
            self.fallback_active = False
        else:
            self.yaw_deg = self._wrap(self.yaw_deg + gyro_z * dt)

        self._last_absolute_yaw_deg = absolute_yaw
        self._last_update_at = timestamp
        return self.yaw_deg, self.fallback_active
