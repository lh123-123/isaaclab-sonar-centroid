// SPDX-License-Identifier: GPL-3.0-or-later
#include "Sub.h"

void Sub::rov_pilot_enter() {
    const bool preserve = control_mode == Mode::Number::STABILIZE || control_mode == Mode::Number::ALT_HOLD;
    g2.rov_pilot.enter(preserve, degrees(ahrs.get_roll_rad()), degrees(ahrs.get_pitch_rad()), degrees(ahrs.get_yaw_rad()));
}

bool Sub::rov_pilot_input_fresh() const {
    if (!g2.rov_pilot.enabled()) { return true; }
    bool failed = failsafe.pilot_input || failsafe.gcs;
#if AP_SUB_RC_ENABLED
    failed = failed || failsafe.radio;
#endif
    return g2.rov_pilot.pilot_fresh(AP_HAL::millis(), failsafe.last_pilot_input_ms,
                                  failed);
}

bool Sub::rov_pilot_attitude(float roll_cd, float pitch_cd) {
    ROV_PilotKernel::Output out;
    // Reuse ArduSub's angle mapping/circular limit, including ALT_HOLD's
    // dynamically permitted tilt, rather than bypassing its depth constraints.
    const float angle_limit = attitude_control.lean_angle_max_cd() * 0.01f;
    const bool valid = g2.rov_pilot.update(G_Dt, degrees(ahrs.get_roll_rad()),
        degrees(ahrs.get_pitch_rad()), degrees(ahrs.get_yaw_rad()),
        roll_cd * 0.01f / angle_limit,
        pitch_cd * 0.01f / angle_limit,
        channel_yaw->get_control_in() / float(ROLL_PITCH_INPUT_MAX),
        rov_pilot_input_fresh(), angle_limit, out);
    if (!valid) { return false; }
    attitude_control.input_euler_angle_roll_pitch_yaw_cd(out.roll * 100.0f, out.pitch * 100.0f, out.yaw * 100.0f, true);
    last_pilot_heading_rad = radians(out.yaw);
    return true;
}
