#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed heading PID controller that outputs yaw moment M_z."""

import math
import time


class HeadingControl:
    """Heading PID using the shortest signed angular error."""

    def __init__(self, Kp=0.5, Ki=0.0, Kd=0.0, sample_time=0.1,
                 output_limit=5.0):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.sample_time = float(sample_time)
        self.output_limit = abs(float(output_limit))
        if not 0.0 < self.output_limit <= 15.0:
            raise ValueError("heading output_limit must be in (0, 15]")
        self.windup_guard = self.output_limit
        self.clear()

    def clear(self):
        self.exp_course = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.output = 0.0
        self.current_time = time.monotonic()
        self.last_time = self.current_time

    @staticmethod
    def shortest_error(target, feedback):
        error = (float(target) - float(feedback) + 180.0) % 360.0 - 180.0
        return error

    @staticmethod
    def wrap_180(angle):
        """Normalize an angle to [-180, 180) without changing its direction."""
        value = float(angle)
        if not math.isfinite(value):
            raise ValueError("heading angle must be finite")
        return (value + 180.0) % 360.0 - 180.0

    def update(self, feedback_course):
        error = self.shortest_error(self.exp_course, feedback_course)
        self.current_time = time.monotonic()
        delta_time = self.current_time - self.last_time
        # A normal 2 degree movement through the -180/180 display boundary
        # must never look like a 358 degree jump to the derivative term.
        delta_error = self.shortest_error(error, self.last_error)

        if delta_time >= self.sample_time:
            self.PTerm = self.Kp * error
            if abs(error) <= 10.0:
                self.ITerm += self.Ki * error * delta_time
                self.ITerm = max(
                    -self.windup_guard, min(self.ITerm, self.windup_guard)
                )
            self.DTerm = (
                self.Kd * delta_error / delta_time if delta_time > 0.0 else 0.0
            )
            self.last_time = self.current_time
            self.last_error = error

            # Plant evidence from the wet test: +Mz increases heading and -Mz
            # decreases it. Therefore Mz must have the same sign as error.
            requested = self.PTerm + self.ITerm + self.DTerm
            if not math.isfinite(requested):
                self.clear()
                raise ValueError("heading PID produced a non-finite output")
            self.output = max(
                -self.output_limit, min(requested, self.output_limit)
            )

        return self.output

    def set_pid(self, pid):
        values = tuple(map(float, pid))
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError("heading PID must contain three finite values")
        self.Kp, self.Ki, self.Kd = values
        self.clear()

    def set_course(self, exp_course):
        course = float(exp_course)
        if not math.isfinite(course):
            raise ValueError("heading target must be finite")
        self.exp_course = self.wrap_180(course)
