// SPDX-License-Identifier: GPL-3.0-or-later
#include "Sub.h"

bool ModeDepthOnly::init(bool ignore_checks)
{
    if (!ModeAlthold::init(ignore_checks)) {
        return false;
    }
    g2.rov_pilot.reset();
    attitude_control->relax_attitude_controllers();
    // Angle-demand inputs become direct torque demands. Neutralize the transition.
    sub.set_neutral_controls();
    return true;
}

void ModeDepthOnly::run()
{
    if (!motors.armed()) {
        motors.set_desired_spool_state(AP_Motors::DesiredSpoolState::GROUND_IDLE);
        attitude_control->set_throttle_out(NEUTRAL_THROTTLE, true, g.throttle_filt);
        attitude_control->relax_attitude_controllers();
        position_control->D_relax_controller(motors.get_throttle_hover());
        g2.rov_pilot.reset();
        g2.rov_pilot.depth_target().reset();
        motors.set_depth_hold_active(false);
        return;
    }
    motors.set_desired_spool_state(AP_Motors::DesiredSpoolState::THROTTLE_UNLIMITED);
    // ArduSub's pressure/vertical velocity loop owns throttle. No angle or rate hold.
    position_control->D_set_max_speed_accel_cm(sub.get_pilot_speed_dn(), g.pilot_speed_up, g.pilot_accel_z);
    control_depth();
    const bool fresh = sub.rov_pilot_input_fresh();
    motors.set_roll(fresh ? channel_roll->norm_input() : 0.0f);
    motors.set_pitch(fresh ? channel_pitch->norm_input() : 0.0f);
    motors.set_yaw(fresh ? channel_yaw->norm_input() * g.acro_yaw_p / ACRO_YAW_P : 0.0f);
    motors.set_forward(fresh ? channel_forward->norm_input() : 0.0f);
    motors.set_lateral(fresh ? channel_lateral->norm_input() : 0.0f);
}
