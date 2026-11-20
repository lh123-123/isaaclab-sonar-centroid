// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <AP_Param/AP_Param.h>
#include "ROV_PilotKernel.h"
#include "ROV_DepthHold.h"
#include <cstdint>

class ROV_PilotHold {
public:
    ROV_PilotHold();
    static const AP_Param::GroupInfo var_info[];
    bool capture_level(float roll, float pitch);
    bool enabled() const { return _enable.get() == 1; }
    void reset() { _targets.reset(); }
    void enter(bool preserve, float roll_deg, float pitch_deg, float yaw_deg) {
        _targets.enter(preserve && enabled(), roll_deg, pitch_deg, yaw_deg);
    }
    bool pilot_fresh(uint32_t now, uint32_t last_input, bool failed) const;
    bool update(float dt, float roll_deg, float pitch_deg, float yaw_deg,
                float roll_input, float pitch_input, float yaw_input,
                bool fresh, float stock_angle_limit_deg, ROV_PilotKernel::Output &out);
    void shift_yaw(float delta_deg) { _targets.shift_yaw(delta_deg); }
    const ROV_PilotKernel::Output &output() const { return _targets.output(); }
    // Shared across ALT_HOLD / DEPTH_ONLY. Attitude resets must NOT erase depth.
    ROV_DepthHold::Target &depth_target() { return _depth; }
private:
    AP_Float _level_roll, _level_pitch;
    AP_Int8 _enable;
    AP_Float _tilt_rate, _tilt_accel, _yaw_rate, _yaw_accel, _yaw_lead, _input_timeout, _tilt_error;
    ROV_PilotKernel::Targets _targets;
    ROV_DepthHold::Target _depth;
};
