/*
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *       AP_Motors6DOF.cpp - ArduSub motors library
 */

#include <AP_BattMonitor/AP_BattMonitor.h>
#include <AP_HAL/AP_HAL.h>
#include "AP_Motors6DOF.h"
#include "ROV_DeadzonePWM.h"
#include "ROV_AttitudeAllocation.h"
#include "ROV_AttitudeLimit.h"

extern const AP_HAL::HAL& hal;

// parameters for the motor class
const AP_Param::GroupInfo AP_Motors6DOF::var_info[] = {
    AP_NESTEDGROUPINFO(AP_MotorsMulticopter, 0),
    // @Param: 1_DIRECTION
    // @DisplayName: Motor normal or reverse
    // @Description: Used to change motor rotation directions without changing wires
    // @Values: 1:normal,-1:reverse
    // @User: Standard
    AP_GROUPINFO("1_DIRECTION", 1, AP_Motors6DOF, _motor_reverse[0], 1),

    // @Param: 2_DIRECTION
    // @DisplayName: Motor normal or reverse
    // @Description: Used to change motor rotation directions without changing wires
    // @Values: 1:normal,-1:reverse
    // @User: Standard
    AP_GROUPINFO("2_DIRECTION", 2, AP_Motors6DOF, _motor_reverse[1], 1),

    // @Param: 3_DIRECTION
    // @DisplayName: Motor normal or reverse
    // @Description: Used to change motor rotation directions without changing wires
    // @Values: 1:normal,-1:reverse
    // @User: Standard
    AP_GROUPINFO("3_DIRECTION", 3, AP_Motors6DOF, _motor_reverse[2], 1),

    // @Param: 4_DIRECTION
    // @DisplayName: Motor normal or reverse
    // @Description: Used to change motor rotation directions without changing wires
    // @Values: 1:normal,-1:reverse
    // @User: Standard
    AP_GROUPINFO("4_DIRECTION", 4, AP_Motors6DOF, _motor_reverse[3], 1),

    // @Param: 5_DIRECTION
    // @DisplayName: Motor normal or reverse
    // @Description: Used to change motor rotation directions without changing wires
    // @Values: 1:normal,-1:reverse
    // @User: Standard
    AP_GROUPINFO("5_DIRECTION", 5, AP_Motors6DOF, _motor_reverse[4], 1),

    // @Param: 6_DIRECTION
    // @DisplayName: Motor normal or reverse
    // @Description: Used to change motor rotation directions without changing wires
    // @Values: 1:normal,-1:reverse
    // @User: Standard
    AP_GROUPINFO("6_DIRECTION", 6, AP_Motors6DOF, _motor_reverse[5], 1),

    // @Param: 7_DIRECTION
    // @DisplayName: Motor normal or reverse
    // @Description: Used to change motor rotation directions without changing wires
    // @Values: 1:normal,-1:reverse
    // @User: Standard
    AP_GROUPINFO("7_DIRECTION", 7, AP_Motors6DOF, _motor_reverse[6], 1),

    // @Param: 8_DIRECTION
    // @DisplayName: Motor normal or reverse
    // @Description: Used to change motor rotation directions without changing wires
    // @Values: 1:normal,-1:reverse
    // @User: Standard
    AP_GROUPINFO("8_DIRECTION", 8, AP_Motors6DOF, _motor_reverse[7], 1),

    // @Param: FV_CPLNG_K
    // @DisplayName: Forward/vertical to pitch decoupling factor
    // @Description: (Vectored frame only) Used to decouple pitch from forward/vertical motion. 0 to disable, 1.2 normal
    // @Range: 0.0 1.5
    // @Increment: 0.1
    // @User: Standard
    AP_GROUPINFO("FV_CPLNG_K", 9, AP_Motors6DOF, _forwardVerticalCouplingFactor, 1.0),

    // @Param: 9_DIRECTION
    // @DisplayName: Motor normal or reverse
    // @Description: Used to change motor rotation directions without changing wires
    // @Values: 1:normal,-1:reverse
    // @User: Standard
    AP_GROUPINFO("9_DIRECTION", 10, AP_Motors6DOF, _motor_reverse[8], 1),

    // @Param: 10_DIRECTION
    // @DisplayName: Motor normal or reverse
    // @Description: Used to change motor rotation directions without changing wires
    // @Values: 1:normal,-1:reverse
    // @User: Standard
    AP_GROUPINFO("10_DIRECTION", 11, AP_Motors6DOF, _motor_reverse[9], 1),

    // @Param: 11_DIRECTION
    // @DisplayName: Motor normal or reverse
    // @Description: Used to change motor rotation directions without changing wires
    // @Values: 1:normal,-1:reverse
    // @User: Standard
    AP_GROUPINFO("11_DIRECTION", 12, AP_Motors6DOF, _motor_reverse[10], 1),

    // @Param: 12_DIRECTION
    // @DisplayName: Motor normal or reverse
    // @Description: Used to change motor rotation directions without changing wires
    // @Values: 1:normal,-1:reverse
    // @User: Standard
    AP_GROUPINFO("12_DIRECTION", 13, AP_Motors6DOF, _motor_reverse[11], 1),

    // @Param: DZ_PWM
    // @DisplayName: ROV post-mix startup PWM offset
    // @Description: Symmetric startup offset after mixing and motor reversal. Zero disables compensation. Not a measured thrust correction; validate in water before enabling.
    // @Units: PWM
    // @Range: 0 100
    // @Increment: 1
    // @User: Advanced
    AP_GROUPINFO("DZ_PWM", 14, AP_Motors6DOF, _startup_deadzone_pwm, 0),
    // @Param: DZ_IN
    // @DisplayName: ROV startup compensation input deadband
    // @Description: Normalized mixed-output magnitude held at neutral when DZ_PWM is enabled. Applies to both pilot and stabilization requests; excessive values impair holding.
    // @Range: 0.0 0.1
    // @Increment: 0.001
    // @User: Advanced
    AP_GROUPINFO("DZ_IN", 15, AP_Motors6DOF, _startup_deadzone_input, 0.01f),

    // @Param: ATT_PRIO
    // @DisplayName: Vectored 6DOF attitude priority allocation
    // @Description: Reserve roll/pitch/yaw authority before translation. May reduce travel or vertical thrust under saturation; report limits to prevent controller windup. Disabled retains stock allocation.
    // @Values: 0:Disabled,1:Enabled
    // @User: Advanced
    AP_GROUPINFO("ATT_PRIO", 16, AP_Motors6DOF, _attitude_priority, 0),

    // @Param: RP_MAX
    // @DisplayName: ROV roll/pitch correction maximum
    // @Description: Vectored 6DOF with ATT_PRIO enabled. Hard cap on the combined roll/pitch correction delivered to each vertical thruster. It does not cap pilot heave or yaw. 25 percent is approximately 5 N only for a 2 kgf thruster; validate in water.
    // @Units: %
    // @Range: 0 100
    // @Increment: 1
    // @User: Advanced
    AP_GROUPINFO("RP_MAX", 17, AP_Motors6DOF, _roll_pitch_limit_pct, 25),

    // @Param: DH_MAX
    // @DisplayName: Neutral-stick depth hold vertical output maximum
    // @Description: Vectored 6DOF with ATT_PRIO enabled. Maximum absolute combined output of each vertical thruster while the depth stick is neutral. Pilot ascent and descent commands are not capped here. If too low to balance buoyancy, depth cannot be held; verify in water.
    // @Units: %
    // @Range: 0 100
    // @Increment: 1
    // @User: Advanced
    AP_GROUPINFO("DH_MAX", 18, AP_Motors6DOF, _depth_hold_limit_pct, 35),

    // @Param: H_REV
    // @DisplayName: ROV lateral and yaw direction correction
    // @Description: Vectored 6DOF only. Reverses lateral force and yaw torque before allocation for this ROV assembly. Covers manual and automatic heading correction. Forward, heave, pitch, roll, individual motor directions and PWM endpoints are unchanged. Change only while disarmed; verify physical heading feedback before enabling stabilization.
    // @Values: 0:Original geometry,1:Reverse lateral and yaw
    // @User: Advanced
    AP_GROUPINFO("H_REV", 20, AP_Motors6DOF, _horizontal_reverse, 1),

    AP_GROUPEND
};

void AP_Motors6DOF::setup_motors(motor_frame_class frame_class, motor_frame_type frame_type)
{
    // remove existing motors
    for (int8_t i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
        remove_motor(i);
    }

    // hard coded config for supported frames
    switch ((sub_frame_t)frame_class) {
        //                 Motor #              Roll Factor     Pitch Factor    Yaw Factor      Throttle Factor     Forward Factor      Lateral Factor  Testing Order
    case SUB_FRAME_BLUEROV1:
        _frame_class_string = "BLUEROV1";
        add_motor_raw_6dof(AP_MOTORS_MOT_1,     0,              0,              -1.0f,          0,                  1.0f,               0,              1);
        add_motor_raw_6dof(AP_MOTORS_MOT_2,     0,              0,              1.0f,           0,                  1.0f,               0,              2);
        add_motor_raw_6dof(AP_MOTORS_MOT_3,     -0.5f,          0.5f,           0,              0.45f,              0,                  0,              3);
        add_motor_raw_6dof(AP_MOTORS_MOT_4,     0.5f,           0.5f,           0,              0.45f,              0,                  0,              4);
        add_motor_raw_6dof(AP_MOTORS_MOT_5,     0,              -1.0f,          0,              1.0f,               0,                  0,              5);
        add_motor_raw_6dof(AP_MOTORS_MOT_6,     -0.25f,         0,              0,              0,                  0,                  1.0f,           6);
        break;

    case SUB_FRAME_VECTORED_6DOF_90DEG:
        _frame_class_string = "VECTORED_6DOF_90DEG";
        add_motor_raw_6dof(AP_MOTORS_MOT_1,     1.0f,           1.0f,           0,              1.0f,               0,                  0,              1);
        add_motor_raw_6dof(AP_MOTORS_MOT_2,     0,              0,              1.0f,           0,                  1.0f,               0,              2);
        add_motor_raw_6dof(AP_MOTORS_MOT_3,     1.0f,           -1.0f,          0,              1.0f,               0,                  0,              3);
        add_motor_raw_6dof(AP_MOTORS_MOT_4,     0,              0,              0,              0,                  0,                  1.0f,           4);
        add_motor_raw_6dof(AP_MOTORS_MOT_5,     0,              0,              0,              0,                  0,                  1.0f,           5);
        add_motor_raw_6dof(AP_MOTORS_MOT_6,     -1.0f,          1.0f,           0,              1.0f,               0,                  0,              6);
        add_motor_raw_6dof(AP_MOTORS_MOT_7,     0,              0,              -1.0f,          0,                  1.0f,               0,              7);
        add_motor_raw_6dof(AP_MOTORS_MOT_8,     -1.0f,          -1.0f,          0,              1.0f,               0,                  0,              8);
        break;

    case SUB_FRAME_VECTORED_6DOF:
        _frame_class_string = "VECTORED_6DOF";
        add_motor_raw_6dof(AP_MOTORS_MOT_1,     0,              0,              1.0f,           0,                  -1.0f,              1.0f,           1);
        add_motor_raw_6dof(AP_MOTORS_MOT_2,     0,              0,              -1.0f,          0,                  -1.0f,              -1.0f,          2);
        add_motor_raw_6dof(AP_MOTORS_MOT_3,     0,              0,              -1.0f,          0,                  1.0f,               1.0f,           3);
        add_motor_raw_6dof(AP_MOTORS_MOT_4,     0,              0,              1.0f,           0,                  1.0f,               -1.0f,          4);
        add_motor_raw_6dof(AP_MOTORS_MOT_5,     1.0f,           -1.0f,          0,              -1.0f,              0,                  0,              5);
        add_motor_raw_6dof(AP_MOTORS_MOT_6,     -1.0f,          -1.0f,          0,              -1.0f,              0,                  0,              6);
        add_motor_raw_6dof(AP_MOTORS_MOT_7,     1.0f,           1.0f,           0,              -1.0f,              0,                  0,              7);
        add_motor_raw_6dof(AP_MOTORS_MOT_8,     -1.0f,          1.0f,           0,              -1.0f,              0,                  0,              8);
        break;

    case SUB_FRAME_VECTORED:
        _frame_class_string = "VECTORED";
        add_motor_raw_6dof(AP_MOTORS_MOT_1,     0,              0,              1.0f,           0,                  -1.0f,              1.0f,           1);
        add_motor_raw_6dof(AP_MOTORS_MOT_2,     0,              0,              -1.0f,          0,                  -1.0f,              -1.0f,          2);
        add_motor_raw_6dof(AP_MOTORS_MOT_3,     0,              0,              -1.0f,          0,                  1.0f,               1.0f,           3);
        add_motor_raw_6dof(AP_MOTORS_MOT_4,     0,              0,              1.0f,           0,                  1.0f,               -1.0f,          4);
        add_motor_raw_6dof(AP_MOTORS_MOT_5,     1.0f,           0,              0,              -1.0f,              0,                  0,              5);
        add_motor_raw_6dof(AP_MOTORS_MOT_6,     -1.0f,          0,              0,              -1.0f,              0,                  0,              6);
        break;

    case SUB_FRAME_CUSTOM:
        // Put your custom motor setup here
        //break;

    case SUB_FRAME_SIMPLEROV_3:
        _frame_class_string = "SIMPLEROV_3";
        add_motor_raw_6dof(AP_MOTORS_MOT_1,     0,              0,              -1.0f,          0,                  1.0f,               0,              1);
        add_motor_raw_6dof(AP_MOTORS_MOT_2,     0,              0,              1.0f,           0,                  1.0f,               0,              2);
        add_motor_raw_6dof(AP_MOTORS_MOT_3,     0,              0,              0,              -1.0f,              0,                  0,              3);
        break;
    case SUB_FRAME_SIMPLEROV_4:
    case SUB_FRAME_SIMPLEROV_5:
    default:
        _frame_class_string = "DEFAULT";
        add_motor_raw_6dof(AP_MOTORS_MOT_1,     0,              0,              -1.0f,          0,                  1.0f,               0,              1);
        add_motor_raw_6dof(AP_MOTORS_MOT_2,     0,              0,              1.0f,           0,                  1.0f,               0,              2);
        add_motor_raw_6dof(AP_MOTORS_MOT_3,     1.0f,           0,              0,              -1.0f,              0,                  0,              3);
        add_motor_raw_6dof(AP_MOTORS_MOT_4,     -1.0f,          0,              0,              -1.0f,              0,                  0,              4);
        add_motor_raw_6dof(AP_MOTORS_MOT_5,     0,              0,              0,              0,                  0,                  1.0f,           5);
        break;
    }
}

void AP_Motors6DOF::add_motor_raw_6dof(int8_t motor_num, float roll_fac, float pitch_fac, float yaw_fac, float throttle_fac, float forward_fac, float lat_fac, uint8_t testing_order)
{
    //Parent takes care of enabling output and setting up masks
    add_motor_raw(motor_num, roll_fac, pitch_fac, yaw_fac, testing_order);

    //These are additional parameters for an ROV
    _throttle_factor[motor_num] = throttle_fac;
    _forward_factor[motor_num] = forward_fac;
    _lateral_factor[motor_num] = lat_fac;
}

// output_min - sends minimum values out to the motors
void AP_Motors6DOF::output_min()
{
    int8_t i;

    // set limits flags
    limit.set_rpy(true);
    limit.set_throttle(false);

    // fill the motor_out[] array for HIL use and send minimum value to each motor
    // ToDo find a field to store the minimum pwm instead of hard coding 1500
    for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {
            rc_write(i, 1500);
        }
    }
}

int16_t AP_Motors6DOF::calc_thrust_to_pwm(float thrust_in) const
{
    return ROV_DeadzonePWM::calculate(thrust_in,
        get_pwm_output_min(), get_pwm_output_max(),
        _startup_deadzone_pwm.get(), _startup_deadzone_input.get());
}

void AP_Motors6DOF::output_to_motors()
{
    int8_t i;
    int16_t motor_out[AP_MOTORS_MAX_NUM_MOTORS];    // final pwm values sent to the motor

    switch (_spool_state) {
    case SpoolState::SHUT_DOWN:
        // sends minimum values out to the motors
        // set motor output based on thrust requests
        for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
            if (motor_enabled[i]) {
                motor_out[i] = 1500;
            }
        }
        break;
    case SpoolState::GROUND_IDLE:
        // sends output to motors when armed but not flying
        for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
            if (motor_enabled[i]) {
                motor_out[i] = 1500;
            }
        }
        break;
    case SpoolState::SPOOLING_UP:
    case SpoolState::THROTTLE_UNLIMITED:
    case SpoolState::SPOOLING_DOWN:
        // set motor output based on thrust requests
        for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
            if (motor_enabled[i]) {
                motor_out[i] = calc_thrust_to_pwm(_thrust_rpyt_out[i]);
            }
        }
        break;
    }

    // send output to each motor
    for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {
            rc_write(i, motor_out[i]);
        }
    }
}

float AP_Motors6DOF::get_current_limit_max_throttle()
{
    return 1.0f;
}

// output_armed - sends commands to the motors
// includes new scaling stability patch
// TODO pull code that is common to output_armed_not_stabilizing into helper functions
// ToDo calculate headroom for rpy to be added for stabilization during full throttle/forward/lateral commands
void AP_Motors6DOF::output_armed_stabilizing()
{
    // Update the total-current envelope before mixing so the priority allocator
    // can spend the remaining authority on rotation first and translation second.
    // MOT_BAT_CURR_MAX <= 0 keeps the native feature disabled.
#if AP_BATTERY_ENABLED
    const AP_BattMonitor &battery = AP::battery();
    float batt_current = 0.0f;
    if (_batt_current_max <= 0.0f) {
        _output_limited = 1.0f;
    } else if (battery.current_amps(batt_current) && std::isfinite(batt_current) &&
               batt_current >= 0.0f && std::isfinite(_dt_s) && _dt_s > 0.0f && _dt_s <= 0.1f &&
               std::isfinite(_batt_current_time_constant) && _batt_current_time_constant >= 0.0f) {
        const float current_delta = batt_current - _batt_current_last;
        const float predicted_current = batt_current + current_delta * 5.0f;
        float current_ratio = batt_current / _batt_current_max;
        if (predicted_current > _batt_current_max * 1.5f) {
            current_ratio = 2.5f;
        } else if (batt_current < _batt_current_max && predicted_current > _batt_current_max) {
            current_ratio = predicted_current / _batt_current_max;
        }
        _batt_current_last = batt_current;
        _output_limited += (_dt_s / (_dt_s + _batt_current_time_constant)) * (1.0f - current_ratio);
    }
#else
    _output_limited = 1.0f;
#endif
    _output_limited = constrain_float(_output_limited, 0.0f, 1.0f);
    const bool priority_6dof = _attitude_priority.get() == 1 &&
        (sub_frame_t)_active_frame_class == SUB_FRAME_VECTORED_6DOF;

    if ((sub_frame_t)_active_frame_class == SUB_FRAME_VECTORED) {
        output_armed_stabilizing_vectored();
    } else if ((sub_frame_t)_active_frame_class == SUB_FRAME_VECTORED_6DOF) {
        output_armed_stabilizing_vectored_6dof();
    } else {
        uint8_t i;                          // general purpose counter
        float   roll_thrust;                // roll thrust input value, +/- 1.0
        float   pitch_thrust;               // pitch thrust input value, +/- 1.0
        float   yaw_thrust;                 // yaw thrust input value, +/- 1.0
        float   throttle_thrust;            // throttle thrust input value, +/- 1.0
        float   forward_thrust;             // forward thrust input value, +/- 1.0
        float   lateral_thrust;             // lateral thrust input value, +/- 1.0

        roll_thrust = (_roll_in + _roll_in_ff);
        pitch_thrust = (_pitch_in + _pitch_in_ff);
        yaw_thrust = (_yaw_in + _yaw_in_ff);
        throttle_thrust = get_throttle_bidirectional();
        forward_thrust = _forward_in;
        lateral_thrust = _lateral_in;

        float rpy_out[AP_MOTORS_MAX_NUM_MOTORS]; // buffer so we don't have to multiply coefficients multiple times.
        float linear_out[AP_MOTORS_MAX_NUM_MOTORS]; // 3 linear DOF mix for each motor

        // initialize limits flags
        limit.set_all(false);

        // sanity check throttle is above zero and below current limited throttle
        if (throttle_thrust <= -_throttle_thrust_max) {
            throttle_thrust = -_throttle_thrust_max;
            limit.throttle_lower = true;
        }
        if (throttle_thrust >= _throttle_thrust_max) {
            throttle_thrust = _throttle_thrust_max;
            limit.throttle_upper = true;
        }

        // calculate roll, pitch and yaw for each motor
        for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
            if (motor_enabled[i]) {
                rpy_out[i] = roll_thrust * _roll_factor[i] +
                             pitch_thrust * _pitch_factor[i] +
                             yaw_thrust * _yaw_factor[i];

            }
        }

        // calculate linear command for each motor
        // linear factors should be 0.0 or 1.0 for now
        for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
            if (motor_enabled[i]) {
                linear_out[i] = throttle_thrust * _throttle_factor[i] +
                                forward_thrust * _forward_factor[i] +
                                lateral_thrust * _lateral_factor[i];
            }
        }

        // Calculate final output for each motor
        for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
            if (motor_enabled[i]) {
                _thrust_rpyt_out[i] = constrain_float(_motor_reverse[i]*(rpy_out[i] + linear_out[i]),-1.0f,1.0f);
            }
        }
    }

    // Priority Vectored-6DOF already allocated directly inside the current
    // envelope. Other frame/mixer paths retain uniform native current scaling.
    if (!priority_6dof) {
        for (uint8_t i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; i++) {
            if (motor_enabled[i]) {
                _thrust_rpyt_out[i] *= _output_limited;
            }
        }
    }
}

// output_armed - sends commands to the motors
// includes new scaling stability patch
// TODO pull code that is common to output_armed_not_stabilizing into helper functions
// ToDo calculate headroom for rpy to be added for stabilization during full throttle/forward/lateral commands
void AP_Motors6DOF::output_armed_stabilizing_vectored()
{
    uint8_t i;                          // general purpose counter
    float   roll_thrust;                // roll thrust input value, +/- 1.0
    float   pitch_thrust;               // pitch thrust input value, +/- 1.0
    float   yaw_thrust;                 // yaw thrust input value, +/- 1.0
    float   throttle_thrust;            // throttle thrust input value, +/- 1.0
    float   forward_thrust;             // forward thrust input value, +/- 1.0
    float   lateral_thrust;             // lateral thrust input value, +/- 1.0

    roll_thrust = (_roll_in + _roll_in_ff);
    pitch_thrust = (_pitch_in + _pitch_in_ff);
    yaw_thrust = (_yaw_in + _yaw_in_ff);
    throttle_thrust = get_throttle_bidirectional();
    forward_thrust = _forward_in;
    lateral_thrust = _lateral_in;

    float rpy_out[AP_MOTORS_MAX_NUM_MOTORS]; // buffer so we don't have to multiply coefficients multiple times.
    float linear_out[AP_MOTORS_MAX_NUM_MOTORS]; // 3 linear DOF mix for each motor

    // initialize limits flags
    limit.set_all(false);

    // sanity check throttle is above zero and below current limited throttle
    if (throttle_thrust <= -_throttle_thrust_max) {
        throttle_thrust = -_throttle_thrust_max;
        limit.throttle_lower = true;
    }

    if (throttle_thrust >= _throttle_thrust_max) {
        throttle_thrust = _throttle_thrust_max;
        limit.throttle_upper = true;
    }

    // calculate roll, pitch and yaw for each motor
    for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {
            rpy_out[i] = roll_thrust * _roll_factor[i] +
                         pitch_thrust * _pitch_factor[i] +
                         yaw_thrust * _yaw_factor[i];
        }
    }

    float forward_coupling_limit = 1-_forwardVerticalCouplingFactor*float(fabsf(throttle_thrust));
    if (forward_coupling_limit < 0) {
        forward_coupling_limit = 0;
    }
    int8_t forward_coupling_direction[] = {-1,-1,1,1,0,0,0,0,0,0,0,0};

    // calculate linear command for each motor
    // linear factors should be 0.0 or 1.0 for now
    for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {

            float forward_thrust_limited = forward_thrust;

            // The following statements decouple forward/vertical hydrodynamic coupling on
            // vectored ROVs. This is done by limiting the maximum output of the "rear" vectored
            // thruster (where "rear" depends on direction of travel).
            if (!is_zero(forward_thrust_limited)) {
                if ((forward_thrust < 0) == (forward_coupling_direction[i] < 0) && forward_coupling_direction[i] != 0) {
                    forward_thrust_limited = constrain_float(forward_thrust, -forward_coupling_limit, forward_coupling_limit);
                }
            }

            linear_out[i] = throttle_thrust * _throttle_factor[i] +
                            forward_thrust_limited * _forward_factor[i] +
                            lateral_thrust * _lateral_factor[i];
        }
    }

    // Calculate final output for each motor
    for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {
            _thrust_rpyt_out[i] = constrain_float(_motor_reverse[i]*(rpy_out[i] + linear_out[i]), -1.0f, 1.0f);
        }
    }
}

// Band Aid fix for motor normalization issues.
// TODO: find a global solution for managing saturation that works for all vehicles
void AP_Motors6DOF::output_armed_stabilizing_vectored_6dof()
{
    uint8_t i;                          // general purpose counter
    float   roll_thrust;                // roll thrust input value, +/- 1.0
    float   pitch_thrust;               // pitch thrust input value, +/- 1.0
    float   yaw_thrust;                 // yaw thrust input value, +/- 1.0
    float   throttle_thrust;            // throttle thrust input value, +/- 1.0
    float   forward_thrust;             // forward thrust input value, +/- 1.0
    float   lateral_thrust;             // lateral thrust input value, +/- 1.0

    roll_thrust = (_roll_in + _roll_in_ff);
    pitch_thrust = (_pitch_in + _pitch_in_ff);
    yaw_thrust = (_yaw_in + _yaw_in_ff);
    throttle_thrust = get_throttle_bidirectional();
    forward_thrust = _forward_in;
    lateral_thrust = _lateral_in;

    // Correct physical force/torque signs at the common actuator boundary.
    // Pilot commands and native heading PID both use this mapping exactly once.
    const int8_t horizontal_reverse = _horizontal_reverse.get();
    if (horizontal_reverse != 0 && horizontal_reverse != 1) {
        limit.set_all(true);
        for (i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; ++i) { _thrust_rpyt_out[i] = 0; }
        return;
    }
    if (horizontal_reverse == 1) {
        yaw_thrust = -yaw_thrust;
        lateral_thrust = -lateral_thrust;
    }

    float rpt_out[AP_MOTORS_MAX_NUM_MOTORS]; // buffer so we don't have to multiply coefficients multiple times.
    float yfl_out[AP_MOTORS_MAX_NUM_MOTORS]; // 3 linear DOF mix for each motor
    float rpt_max;
    float yfl_max;

    // initialize limits flags
    limit.set_all(false);

    // sanity check throttle is above zero and below current limited throttle
    if (throttle_thrust <= -_throttle_thrust_max) {
        throttle_thrust = -_throttle_thrust_max;
        limit.throttle_lower = true;
    }

    if (throttle_thrust >= _throttle_thrust_max) {
        throttle_thrust = _throttle_thrust_max;
        limit.throttle_upper = true;
    }

    const float requested_throttle = throttle_thrust;
    throttle_thrust = constrain_float(throttle_thrust, -1.0f, _max_throttle);
    if (_attitude_priority.get() == 1) {
        // Near-surface safety can clip before the allocator. Tell depth PID
        // about that lost authority as well; do not remove the surface guard.
        limit.throttle_upper = limit.throttle_upper || throttle_thrust < requested_throttle;
        limit.throttle_lower = limit.throttle_lower || throttle_thrust > requested_throttle;
    }

    if (_attitude_priority.get() == 1) {
        float rv[AP_MOTORS_MAX_NUM_MOTORS] = {};
        float rh[AP_MOTORS_MAX_NUM_MOTORS] = {};
        float tv[AP_MOTORS_MAX_NUM_MOTORS] = {};
        float th[AP_MOTORS_MAX_NUM_MOTORS] = {};
        bool vertical[AP_MOTORS_MAX_NUM_MOTORS] = {};
        bool horizontal[AP_MOTORS_MAX_NUM_MOTORS] = {};
        for (i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; ++i) {
            if (!motor_enabled[i]) { continue; }
            vertical[i] = !is_zero(_throttle_factor[i]);
            horizontal[i] = !vertical[i];
            rv[i] = roll_thrust * _roll_factor[i] + pitch_thrust * _pitch_factor[i];
            rh[i] = yaw_thrust * _yaw_factor[i];
            tv[i] = throttle_thrust * _throttle_factor[i];
            th[i] = forward_thrust * _forward_factor[i] + lateral_thrust * _lateral_factor[i];
        }
        const float rp_limit_pct = _roll_pitch_limit_pct.get();
        const float rp_limit = rp_limit_pct * 0.01f;
        const auto rp_cap = ROV_AttitudeLimit::cap(rv, vertical, AP_MOTORS_MAX_NUM_MOTORS, rp_limit);
        const float dh_limit_pct = _depth_hold_limit_pct.get();
        const float dh_limit = dh_limit_pct * 0.01f;
        if (!std::isfinite(dh_limit) || dh_limit < 0.0f || dh_limit > 1.0f) {
            limit.set_all(true);
            for (i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; ++i) { _thrust_rpyt_out[i] = 0; }
            return;
        }
        const float vertical_envelope = _depth_hold_active ? MIN(_output_limited, dh_limit) : _output_limited;
        const auto v = ROV_AttitudeAllocation::allocate(rv, tv, vertical, AP_MOTORS_MAX_NUM_MOTORS, vertical_envelope);
        const auto h = ROV_AttitudeAllocation::allocate(rh, th, horizontal, AP_MOTORS_MAX_NUM_MOTORS, _output_limited);
        if (!rp_cap.valid || !v.valid || !h.valid) {
            limit.set_all(true);
            for (i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; ++i) { _thrust_rpyt_out[i] = 0; }
            return;
        }
        limit.roll = limit.pitch = rp_cap.scale < 0.9999f || v.rotation_scale < 0.9999f;
        limit.yaw = h.rotation_scale < 0.9999f;
        if (v.translation_scale < 0.9999f) {
            limit.throttle_lower = limit.throttle_lower || throttle_thrust < 0;
            limit.throttle_upper = limit.throttle_upper || throttle_thrust > 0;
        }
        for (i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; ++i) {
            if (motor_enabled[i]) {
                _thrust_rpyt_out[i] = constrain_float(_motor_reverse[i] * (vertical[i] ? rv[i] : rh[i]), -1.0f, 1.0f);
            }
        }
        return;
    }

    // calculate roll, pitch and Throttle for each motor (only used by vertical thrusters)
    rpt_max = 1; //Initialized to 1 so that normalization will only occur if value is saturated
    for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {
            rpt_out[i] = roll_thrust * _roll_factor[i] +
                         pitch_thrust * _pitch_factor[i] +
                         throttle_thrust * _throttle_factor[i];
            if (fabsf(rpt_out[i]) > rpt_max) {
                rpt_max = fabsf(rpt_out[i]);
            }
        }
    }

    // calculate linear/yaw command for each motor (only used for translational thrusters)
    // linear factors should be 0.0 or 1.0 for now
    yfl_max = 1; //Initialized to 1 so that normalization will only occur if value is saturated
    for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {
            yfl_out[i] = yaw_thrust * _yaw_factor[i] +
                         forward_thrust * _forward_factor[i] +
                         lateral_thrust * _lateral_factor[i];
            if (fabsf(yfl_out[i]) > yfl_max) {
                yfl_max = fabsf(yfl_out[i]);
            }
        }
    }

    // Calculate final output for each motor and normalize if necessary
    for (i=0; i<AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {
            _thrust_rpyt_out[i] = constrain_float(_motor_reverse[i]*(rpt_out[i]/rpt_max + yfl_out[i]/yfl_max),-1.0f,1.0f);
        }
    }
}

Vector3f AP_Motors6DOF::get_motor_angular_factors(int motor_number) {
     if (motor_number < 0 || motor_number >= AP_MOTORS_MAX_NUM_MOTORS) {
        return Vector3f(0,0,0);
    }
    const float yaw_sign = ((sub_frame_t)_active_frame_class == SUB_FRAME_VECTORED_6DOF &&
                            _horizontal_reverse.get() == 1) ? -1.0f : 1.0f;
    return Vector3f(_roll_factor[motor_number], _pitch_factor[motor_number], yaw_sign * _yaw_factor[motor_number]);
}

bool AP_Motors6DOF::motor_is_enabled(int motor_number) {
    if (motor_number < 0 || motor_number >= AP_MOTORS_MAX_NUM_MOTORS) {
        return false;
    }
    return motor_enabled[motor_number];
}

bool AP_Motors6DOF::set_reversed(int motor_number, bool reversed) {
    if (motor_number < 0 || motor_number >= AP_MOTORS_MAX_NUM_MOTORS) {
        return false;
    }
    if (reversed) {
        _motor_reverse[motor_number].set_and_save(-1);
    } else {
        _motor_reverse[motor_number].set_and_save(1);
    }
    return true;
}
