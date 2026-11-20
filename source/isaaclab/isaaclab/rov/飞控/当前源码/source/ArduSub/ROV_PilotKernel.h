// SPDX-License-Identifier: GPL-3.0-or-later
// Pilot target generator only. ArduSub retains ownership of all feedback loops.
#pragma once
#include <cmath>

namespace ROV_PilotKernel {
inline float clamp(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }
inline float wrap(float v) {
    v = ::fmodf(v + 180.0f, 360.0f);
    return (v < 0 ? v + 360.0f : v) - 180.0f;
}
inline float approach(float v, float target, float step) {
    return v + clamp(target - v, -step, step);
}
// Give the native angle/rate controller the closest point in the allowed
// interval. Inside the interval this removes angle correction while retaining
// angular-rate damping. The nominal target itself remains fixed.
inline float tolerant_angle_target(float target, float measured, float tolerance) {
    if (!std::isfinite(target) || !std::isfinite(measured) ||
        !std::isfinite(tolerance) || tolerance < 0.0f || tolerance > 5.0f) {
        return target;
    }
    return clamp(measured, target - tolerance, target + tolerance);
}
struct Config {
    float level_roll = 0;
    float level_pitch = 0;
    float pitch_rate = 30.0f;
    float pitch_accel = 90.0f;
    float pitch_limit = 30.0f;
    float target_lead = 12.0f;
    float yaw_rate = 45.0f;
    float yaw_accel = 120.0f;
};
struct Output {
    float roll = 0;
    float pitch = 0;
    float yaw = 0;
    float pitch_rate = 0;
    float roll_rate = 0;
    float yaw_rate = 0;
    bool valid = false;
};
class Targets {
public:
    void reset() { _out = Output{}; _yaw_turning = false; }
    void enter(bool preserve, float roll, float pitch, float yaw) {
        if (preserve && _out.valid) { return; }
        reset();
        if (std::isfinite(roll) && std::isfinite(pitch) && std::isfinite(yaw)) {
            _out.roll = roll;
            _out.pitch = pitch;
            _out.yaw = wrap(yaw);
            _out.valid = true;
        }
    }
    void shift_yaw(float delta) {
        if (_out.valid && std::isfinite(delta)) { _out.yaw = wrap(_out.yaw + delta); }
    }
    const Output &output() const { return _out; }
    Output update(float dt, float measured_roll, float measured_pitch, float measured_yaw,
                  float roll_input, float pitch_input, float yaw_input, bool pilot_fresh, const Config &c) {
        if (!valid_config(c) || !std::isfinite(dt) || dt <= 0 || dt > 0.1f ||
            !std::isfinite(measured_roll) || !std::isfinite(measured_pitch) || !std::isfinite(measured_yaw) ||
            !std::isfinite(roll_input) || !std::isfinite(pitch_input) || !std::isfinite(yaw_input)) {
            // Never integrate a scheduler gap or bad data into a large target jump.
            reset();
            return _out;
        }
        if (!_out.valid) { enter(false, measured_roll, measured_pitch, measured_yaw); }
        if (!pilot_fresh) {
            roll_input = pitch_input = yaw_input = 0;
            _out.yaw_rate = 0;
            if (_yaw_turning) { _out.yaw = wrap(measured_yaw); }
            _yaw_turning = false;
        }
        roll_input = clamp(roll_input, -1, 1);
        pitch_input = clamp(pitch_input, -1, 1);
        yaw_input = clamp(yaw_input, -1, 1);
        const float tilt_input = ::hypotf(roll_input, pitch_input);
        if (tilt_input > 1.0f) { roll_input /= tilt_input; pitch_input /= tilt_input; }
        // Angle targets: centered pitch/roll ALWAYS return to level. Do not
        // capture and hold their last angle when the operator releases a stick.
        const float tilt_envelope = ::fmaxf(c.pitch_limit, ::hypotf(_out.roll, _out.pitch));
        float desired_roll = c.level_roll + roll_input * c.pitch_limit;
        float desired_pitch = c.level_pitch + pitch_input * c.pitch_limit;
        const float desired_tilt = ::hypotf(desired_roll, desired_pitch);
        if (desired_tilt > c.pitch_limit) {
            desired_roll *= c.pitch_limit / desired_tilt;
            desired_pitch *= c.pitch_limit / desired_tilt;
        }
        move_angle(_out.roll, _out.roll_rate, desired_roll, dt, c);
        move_angle(_out.pitch, _out.pitch_rate, desired_pitch, dt, c);
        const float tilt = ::hypotf(_out.roll, _out.pitch);
        if (tilt > tilt_envelope) {
            _out.roll *= tilt_envelope / tilt;
            _out.pitch *= tilt_envelope / tilt;
            _out.roll_rate = _out.pitch_rate = 0;
        }
        if (::fabsf(yaw_input) > 1.0e-6f) {
            _yaw_turning = true;
            _out.yaw_rate = approach(_out.yaw_rate, yaw_input * c.yaw_rate, c.yaw_accel * dt);
            _out.yaw = wrap(_out.yaw + _out.yaw_rate * dt);
        } else {
            // Freeze the reference on the FIRST release sample, not at the end
            // of a timed deceleration. Native angle/rate feedback brakes actual
            // inertia and corrects overshoot back to this fixed heading.
            if (_yaw_turning) { _out.yaw = wrap(measured_yaw); }
            _out.yaw_rate = 0;
            _yaw_turning = false;
        }
        // Restrict yaw target lead only during a turn; while holding, never
        // re-capture a disturbed measured heading, which would silently drift.
        if (::fabsf(yaw_input) > 1.0e-6f) {
            _out.yaw = wrap(measured_yaw + clamp(wrap(_out.yaw - measured_yaw), -c.target_lead, c.target_lead));
        }
        return _out;
    }
private:
    static void move_angle(float &angle, float &rate, float desired, float dt, const Config &c) {
        const float error = desired - angle;
        const float braking_speed = ::sqrtf(2.0f * c.pitch_accel * ::fabsf(error));
        const float desired_rate = ::copysignf(::fminf(c.pitch_rate, braking_speed), error);
        rate = approach(rate, desired_rate, c.pitch_accel * dt);
        const float next = angle + rate * dt;
        // Crossing a newly moved goal while still moving quickly is NOT an
        // instruction to instantaneously drop velocity. Decelerate first.
        const bool reached = error * (desired - next) <= 0 || ::fabsf(error) < 0.001f;
        if (reached && ::fabsf(rate) <= c.pitch_accel * dt + 0.0001f) {
            angle = desired;
            rate = 0;
        } else {
            const float bounded = clamp(next, ::fminf(angle, -c.pitch_limit), ::fmaxf(angle, c.pitch_limit));
            if (::fabsf(bounded - next) > 1.0e-6f) { rate = 0; }
            angle = bounded;
        }
    }
    static bool valid_config(const Config &c) {
        return std::isfinite(c.level_roll) && std::isfinite(c.level_pitch) &&
            ::hypotf(c.level_roll, c.level_pitch) <= 10.0f && std::isfinite(c.pitch_rate) && c.pitch_rate >= 1 && c.pitch_rate <= 90 &&
            std::isfinite(c.pitch_accel) && c.pitch_accel >= 10 && c.pitch_accel <= 360 &&
            std::isfinite(c.pitch_limit) && c.pitch_limit >= 10 && c.pitch_limit <= 60 &&
            std::isfinite(c.target_lead) && c.target_lead >= 1 && c.target_lead <= 30 &&
            std::isfinite(c.yaw_rate) && c.yaw_rate >= 1 && c.yaw_rate <= 120 &&
            std::isfinite(c.yaw_accel) && c.yaw_accel >= 10 && c.yaw_accel <= 720;
    }
    Output _out;
    bool _yaw_turning = false;
};
} // namespace ROV_PilotKernel
