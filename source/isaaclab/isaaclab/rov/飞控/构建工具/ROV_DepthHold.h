// SPDX-License-Identifier: GPL-3.0-or-later
// Release-reference latch only; native pressure/velocity/acceleration PID owns outputs.
#pragma once
#include <cmath>
#include <cstdint>

namespace ROV_DepthHold {
struct Output {
    float altitude_cm = 0;  // Up-positive relative to the native EKF origin
    bool hold = false;
    bool captured = false;
    bool valid = false;
};
class Target {
public:
    void reset() { _out = Output{}; _moving = false; _reset_seen = false; _reset_ms = 0; }
    const Output &output() const { return _out; }
    Output update(float measured_up_cm, float climb_rate_cms,
                  uint32_t reset_ms, float reset_down_m) {
        if (!std::isfinite(measured_up_cm) || !std::isfinite(climb_rate_cms) ||
            !std::isfinite(reset_down_m)) { reset(); return _out; }
        // Shift ONLY an existing reference before capturing a same-tick release.
        // A positive Down-frame origin shift is a negative Up-centimeter shift.
        if (_reset_seen && reset_ms != 0 && reset_ms != _reset_ms && _out.valid) {
            _out.altitude_cm -= reset_down_m * 100.0f;
            if (!std::isfinite(_out.altitude_cm)) { reset(); return _out; }
        }
        _reset_seen = true;
        _reset_ms = reset_ms;
        _out.captured = false;
        if (::fabsf(climb_rate_cms) >= 0.05f) {
            _moving = true;
            _out.hold = false;
            _out.valid = true;
            return _out;
        }
        if (_moving || !_out.valid) {
            _out.altitude_cm = measured_up_cm;
            _out.captured = true;
        }
        _moving = false;
        _out.hold = _out.valid = true;
        return _out;
    }
    void limit_target(float minimum_up_cm, float maximum_up_cm) {
        if (!_out.hold || !std::isfinite(minimum_up_cm) ||
            !std::isfinite(maximum_up_cm) || minimum_up_cm > maximum_up_cm) { return; }
        _out.altitude_cm = ::fmaxf(minimum_up_cm, ::fminf(maximum_up_cm, _out.altitude_cm));
    }
private:
    Output _out;
    bool _moving = false;
    bool _reset_seen = false;
    uint32_t _reset_ms = 0;
};
} // namespace ROV_DepthHold
