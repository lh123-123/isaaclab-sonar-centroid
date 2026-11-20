// SPDX-License-Identifier: GPL-3.0-or-later
#include "ROV_PilotHold.h"

const AP_Param::GroupInfo ROV_PilotHold::var_info[] = {
    // @Param: EN
    // @DisplayName: ROV smooth pilot angle control
    // @Description: Smooth tilt targets, capture yaw on stick release, and capture depth on climb-input release in ALT_HOLD/DEPTH_ONLY. Native feedback brakes inertia. Disabled retains stock behavior.
    // @Values: 0:Disabled,1:Enabled
    // @User: Advanced
    AP_GROUPINFO("EN", 1, ROV_PilotHold, _enable, 0),
    // @Param: T_RATE
    // @DisplayName: ROV tilt target maximum slew rate
    // @Units: deg/s
    // @Range: 1 90
    // @User: Advanced
    AP_GROUPINFO("T_RATE", 2, ROV_PilotHold, _tilt_rate, 30),
    // @Param: T_ACC
    // @DisplayName: ROV tilt target maximum acceleration
    // @Units: deg/s/s
    // @Range: 10 360
    // @User: Advanced
    AP_GROUPINFO("T_ACC", 3, ROV_PilotHold, _tilt_accel, 90),
    // @Param: Y_RATE
    // @DisplayName: ROV pilot maximum yaw rate
    // @Units: deg/s
    // @Range: 1 120
    // @User: Advanced
    AP_GROUPINFO("Y_RATE", 4, ROV_PilotHold, _yaw_rate, 45),
    // @Param: Y_ACC
    // @DisplayName: ROV yaw target maximum acceleration
    // @Units: deg/s/s
    // @Range: 10 720
    // @User: Advanced
    AP_GROUPINFO("Y_ACC", 5, ROV_PilotHold, _yaw_accel, 120),
    // @Param: Y_LEAD
    // @DisplayName: ROV maximum yaw target lead during a turn
    // @Description: Caps target lead while turning; freezes measured heading on the first release sample, not after a settling delay.
    // @Units: deg
    // @Range: 1 30
    // @User: Advanced
    AP_GROUPINFO("Y_LEAD", 6, ROV_PilotHold, _yaw_lead, 12),
    // @Param: IN_TMO
    // @DisplayName: ROV pilot input neutralization timeout
    // @Description: Neutralize manual demands after this interval without pilot input. Does not replace or disable the existing pilot/GCS failsafe.
    // @Units: s
    // @Range: 0.2 1
    // @User: Advanced
    AP_GROUPINFO("IN_TMO", 7, ROV_PilotHold, _input_timeout, 0.35f),
    // @Param: T_ERR
    // @DisplayName: ROV roll and pitch hold tolerance
    // @Description: Allowed roll/pitch angle error around the pilot target. Within this band the native rate loop still damps rotation; beyond it the angle loop corrects toward the nearest band edge. Zero restores exact-angle hold.
    // @Units: deg
    // @Range: 0 5
    // @User: Standard
    AP_GROUPINFO("T_ERR", 8, ROV_PilotHold, _tilt_error, 0.5f),
    AP_GROUPEND
};

ROV_PilotHold::ROV_PilotHold() { AP_Param::setup_object_defaults(this, var_info); }

bool ROV_PilotHold::pilot_fresh(uint32_t now, uint32_t last_input, bool failed) const {
    if (failed || last_input == 0 || !std::isfinite(_input_timeout.get()) || _input_timeout.get() < 0.2f ||
        _input_timeout.get() > 1.0f) { return false; }
    return uint32_t(now - last_input) <= uint32_t(_input_timeout.get() * 1000.0f);
}

bool ROV_PilotHold::update(float dt, float roll_deg, float pitch_deg, float yaw_deg,
                         float roll_input, float pitch_input, float yaw_input,
                         bool fresh, float stock_angle_limit_deg, ROV_PilotKernel::Output &out) {
    if (!enabled()) { reset(); return false; }
    ROV_PilotKernel::Config c;
    c.pitch_rate = _tilt_rate.get();
    c.pitch_accel = _tilt_accel.get();
    c.pitch_limit = stock_angle_limit_deg;
    c.yaw_rate = _yaw_rate.get();
    c.yaw_accel = _yaw_accel.get();
    c.target_lead = _yaw_lead.get();
    out = _targets.update(dt, roll_deg, pitch_deg, yaw_deg, roll_input, pitch_input, yaw_input, fresh, c);
    if (out.valid) {
        const float configured_error = _tilt_error.get();
        const float tolerance = std::isfinite(configured_error) && configured_error >= 0.0f &&
            configured_error <= 5.0f ? configured_error : 0.0f;
        out.roll = ROV_PilotKernel::tolerant_angle_target(out.roll, roll_deg, tolerance);
        out.pitch = ROV_PilotKernel::tolerant_angle_target(out.pitch, pitch_deg, tolerance);
    }
    return out.valid;
}
