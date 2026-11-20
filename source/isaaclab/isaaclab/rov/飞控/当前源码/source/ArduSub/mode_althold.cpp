#include "Sub.h"


bool ModeAlthold::init(bool ignore_checks) {
    if(!sub.control_check_barometer()) {
        return false;
    }

    // initialize vertical maximum speeds and acceleration
    // sets the maximum speed up and down returned by position controller
    // All limits must be positive
    position_control->D_set_max_speed_accel_cm(sub.get_pilot_speed_dn(), g.pilot_speed_up, g.pilot_accel_z);
    position_control->D_set_correction_speed_accel_cm(sub.get_pilot_speed_dn(), g.pilot_speed_up, g.pilot_accel_z);

    // initialise position and desired velocity
    // Preserve the existing depth target when toggling attitude during depth hold.
    if (sub.control_mode != Mode::Number::ALT_HOLD && sub.control_mode != Mode::Number::DEPTH_ONLY) {
        position_control->D_init_controller();
        g2.rov_pilot.depth_target().reset();
    }

    sub.last_pilot_heading_rad = ahrs.get_yaw_rad();
    sub.rov_pilot_enter();

    return true;
}

// althold_run - runs the althold controller
// should be called at 100hz or more
void ModeAlthold::run()
{
    run_pre();
    control_depth();
    run_post();
}

void ModeAlthold::run_pre()
{
    uint32_t tnow = AP_HAL::millis();

    // initialize vertical speeds and acceleration
    // All limits must be positive
    position_control->D_set_max_speed_accel_cm(sub.get_pilot_speed_dn(), g.pilot_speed_up, g.pilot_accel_z);

    if (!motors.armed()) {
        motors.set_desired_spool_state(AP_Motors::DesiredSpoolState::GROUND_IDLE);
        // Sub vehicles do not stabilize roll/pitch/yaw when not auto-armed (i.e. on the ground, pilot has never raised throttle)
        attitude_control->set_throttle_out(NEUTRAL_THROTTLE,true,g.throttle_filt);
        attitude_control->relax_attitude_controllers();
        position_control->D_relax_controller(motors.get_throttle_hover());
        sub.last_pilot_heading_rad = ahrs.get_yaw_rad();
        g2.rov_pilot.reset();
        return;
    }

    motors.set_desired_spool_state(AP_Motors::DesiredSpoolState::THROTTLE_UNLIMITED);

    // get pilot desired lean angles
    float target_roll, target_pitch;

    // Check if set_attitude_target_no_gps is valid
    if (sub.set_attitude_target_no_gps.last_message_ms != 0 && tnow - sub.set_attitude_target_no_gps.last_message_ms < 5000) {
        g2.rov_pilot.reset();
        float target_yaw;
        Quaternion(
            sub.set_attitude_target_no_gps.packet.q
        ).to_euler(
            target_roll,
            target_pitch,
            target_yaw
        );
        target_roll = degrees(target_roll);
        target_pitch = degrees(target_pitch);
        target_yaw = degrees(target_yaw);

        attitude_control->input_euler_angle_roll_pitch_yaw_cd(target_roll * 1e2f, target_pitch * 1e2f, target_yaw * 1e2f, true);
        return;
    }

    sub.get_pilot_desired_lean_angles(channel_roll->get_control_in(), channel_pitch->get_control_in(), target_roll, target_pitch, attitude_control->get_althold_lean_angle_max_cd());

    if (!sub.rov_pilot_attitude(target_roll, target_pitch)) {
        if (!sub.rov_pilot_input_fresh()) { target_roll = target_pitch = 0.0f; }
        // get pilot's desired yaw rate
        float yaw_input = sub.rov_pilot_input_fresh() ? channel_yaw->pwm_to_angle_dz_trim(channel_yaw->get_dead_zone() * sub.gain, channel_yaw->get_radio_trim()) : 0.0f;
        float target_yaw_rate = sub.get_pilot_desired_yaw_rate(yaw_input);

        // call attitude controller
        if (!is_zero(target_yaw_rate)) { // call attitude controller with rate yaw determined by pilot input
            attitude_control->input_euler_angle_roll_pitch_euler_rate_yaw_cd(target_roll, target_pitch, target_yaw_rate);
            sub.last_pilot_heading_rad = ahrs.get_yaw_rad();
            sub.last_pilot_yaw_input_ms = tnow; // time when pilot last changed heading

        } else { // hold current heading

            // this check is required to prevent bounce back after very fast yaw maneuvers
            // the inertia of the vehicle causes the heading to move slightly past the point when pilot input actually stopped
            if (tnow < sub.last_pilot_yaw_input_ms + 250) { // give 250ms to slow down, then set target heading
                target_yaw_rate = 0; // Stop rotation on yaw axis

                // call attitude controller with target yaw rate = 0 to decelerate on yaw axis
                attitude_control->input_euler_angle_roll_pitch_euler_rate_yaw_cd(target_roll, target_pitch, target_yaw_rate);
                sub.last_pilot_heading_rad = ahrs.get_yaw_rad(); // update heading to hold

            } else { // call attitude controller holding absolute bearing
                attitude_control->input_euler_angle_roll_pitch_yaw_cd(target_roll, target_pitch, rad_to_cd(sub.last_pilot_heading_rad), true);
            }
        }
    }

}

void ModeAlthold::run_post()
{
    motors.set_forward(sub.rov_pilot_input_fresh() ? channel_forward->norm_input() : 0.0f);
    motors.set_lateral(sub.rov_pilot_input_fresh() ? channel_lateral->norm_input() : 0.0f);
}

void ModeAlthold::control_depth() {
    const bool release_hold = g2.rov_pilot.enabled() &&
        (sub.control_mode == Mode::Number::ALT_HOLD || sub.control_mode == Mode::Number::DEPTH_ONLY);
    if (release_hold && !motors.armed()) {
        // run_pre() returning while disarmed does not exit ModeAlthold::run().
        // Do not resurrect a target after its controller has been relaxed.
        g2.rov_pilot.depth_target().reset();
        motors.set_depth_hold_active(false);
        return;
    }
    // return 0.2f when at the surface to p
    // scale linearly between 0.2f and 1.0f as we approach the surface
    float distance_to_surface = (g.surface_depth - inertial_nav.get_position_z_up_cm()) * 0.01f;
    distance_to_surface = constrain_float(distance_to_surface, 0.0f, 1.0f);
    motors.set_max_throttle(g.surface_max_throttle + (1.0f - g.surface_max_throttle) * distance_to_surface);

    const bool pilot_input_fresh = sub.rov_pilot_input_fresh();
    const float throttle_control = pilot_input_fresh ? channel_throttle->get_control_in() : channel_throttle->get_control_mid();
    float target_climb_rate_cms = pilot_input_fresh ? sub.get_pilot_desired_climb_rate(throttle_control) : 0.0f;
    target_climb_rate_cms = constrain_float(target_climb_rate_cms, -sub.get_pilot_speed_dn(), g.pilot_speed_up);
    // THR_DZ is much wider than the PC trigger's actual neutral. Distinguish
    // "zero requested climb" from "trigger fully released" for reference
    // capture, without weakening the existing zero-climb output cap.
    // A stale pilot stream is treated as released; it may never retain a
    // previous nonzero climb request. 5 units is 1% of half travel.
    const bool pilot_released = fabsf(target_climb_rate_cms) < 0.05f &&
        (!pilot_input_fresh || fabsf(throttle_control - channel_throttle->get_control_mid()) <= 5.0f);
    // Preserve the previous cap whenever requested climb is zero, including
    // the narrow interval before the trigger reaches exact neutral.
    // The allocator reports saturation back to the native depth PID.
    motors.set_depth_hold_active(fabsf(target_climb_rate_cms) < 0.05f);

    // desired_climb_rate returns 0 when within the deadzone.
    //we allow full control to the pilot, but as soon as there's no input, we handle being at surface/bottom
    if (fabsf(target_climb_rate_cms) < 0.05f)  {
        if (sub.ap.at_surface) {
            position_control->set_pos_desired_U_cm(MIN(position_control->get_pos_desired_U_cm(), g.surface_depth)); // set target to 5 cm below surface level
        } else if (sub.ap.at_bottom) {
            position_control->set_pos_desired_U_cm(MAX(inertial_nav.get_position_z_up_cm() + 10.0f, position_control->get_pos_desired_U_cm())); // set target to 10 cm above bottom
        }
    }

    if (release_hold) {
        float reset_down_m = 0.0f;
        const uint32_t reset_ms = ahrs.getLastPosDownReset(reset_down_m);
        auto &depth = g2.rov_pilot.depth_target();
        const auto target = depth.update(inertial_nav.get_position_z_up_cm(),
                                         target_climb_rate_cms, pilot_released, reset_ms, reset_down_m);
        if (target.valid && target.hold) {
            // Safety overrides the release reference near the surface/bottom.
            // Store the override so departing the boundary cannot revive an
            // old unsafe reference. Both remain Up-positive centimeters.
            if (sub.ap.at_surface) {
                depth.limit_target(-1.0e20f, g.surface_depth);
            } else if (sub.ap.at_bottom) {
                depth.limit_target(inertial_nav.get_position_z_up_cm() + 10.0f, 1.0e20f);
            }
            // A FIXED reference, not climb(0) which integrates deceleration.
            // Native jerk/accel shaping and pressure/velocity PID brake inertia.
            position_control->set_alt_target_with_slew_cm(depth.output().altitude_cm);
        } else {
            position_control->D_set_pos_target_from_climb_rate_cms(target.valid ? target_climb_rate_cms : 0.0f);
        }
    } else {
        g2.rov_pilot.depth_target().reset();
        position_control->D_set_pos_target_from_climb_rate_cms(target_climb_rate_cms);
    }
    position_control->D_update_controller();
}
