"""Generate a reviewable combined patch from the pinned, local source excerpts.

This never connects to a Pi/vehicle, builds a firmware, or changes parameters.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
COMMIT = "dbe792162d06cab66c3475fd5556bf7a120f119e"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Pinned source anchor not unique: {old[:90]!r}")
    return text.replace(old, new, 1)


def build_sources(baseline: Path):
    paths = ("ArduSub/Parameters.h", "ArduSub/Parameters.cpp", "ArduSub/Sub.h",
             "ArduSub/mode_stabilize.cpp", "ArduSub/mode_althold.cpp",
             "ArduSub/Attitude.cpp", "ArduSub/GCS_MAVLink_Sub.cpp",
             "libraries/AP_Motors/AP_Motors6DOF.cpp", "libraries/AP_Motors/AP_Motors6DOF.h")
    original = {p: (baseline / p).read_text(encoding="utf-8") for p in paths}
    updated = original.copy()
    def edit(p, old, new):
        updated[p] = replace_once(updated[p], old, new)

    edit("ArduSub/Sub.h", '#include "Parameters.h"', '#include "ROV_PilotHold.h"\n#include "Parameters.h"')
    edit("ArduSub/Sub.h", '    void run_rate_controller();',
         '    void rov_pilot_enter();\n    bool rov_pilot_input_fresh() const;\n    bool rov_pilot_attitude(float roll_cd, float pitch_cd);\n    void run_rate_controller();')
    edit("ArduSub/Parameters.h", '    Actuators actuators;', '    Actuators actuators;\n    ROV_PilotHold rov_pilot;')
    edit("ArduSub/Parameters.cpp", '    AP_GROUPINFO_FLAGS("PARM_FMT_VER", 24, ParametersG2, param_conversion_increment, 0, AP_PARAM_FLAG_HIDDEN),',
         '    AP_GROUPINFO_FLAGS("PARM_FMT_VER", 24, ParametersG2, param_conversion_increment, 0, AP_PARAM_FLAG_HIDDEN),\n\n'
         '    // @Group: ROV_\n    // @Path: ROV_PilotHold.cpp\n'
         '    AP_SUBGROUPINFO(rov_pilot, "ROV_", 25, ParametersG2, ROV_PilotHold),')
    for p in ("ArduSub/mode_stabilize.cpp", "ArduSub/mode_althold.cpp"):
        edit(p, '    sub.last_pilot_heading_rad = ahrs.get_yaw_rad();\n\n    return true;',
             '    sub.last_pilot_heading_rad = ahrs.get_yaw_rad();\n    sub.rov_pilot_enter();\n\n    return true;')
        edit(p, '        sub.last_pilot_heading_rad = ahrs.get_yaw_rad();\n        return;',
             '        sub.last_pilot_heading_rad = ahrs.get_yaw_rad();\n        g2.rov_pilot.reset();\n        return;')
        # Wrap only the stock yaw controller, preserving it as the disabled fallback.
        start = updated[p].index('    // get pilot\'s desired yaw rate')
        end = (updated[p].index('    // output pilot\'s throttle', start)
               if 'stabilize.cpp' in p else updated[p].index('\n}\n\nvoid ModeAlthold::run_post()', start))
        stock = updated[p][start:end].rstrip()
        stock = replace_once(stock,
            'float yaw_input = channel_yaw->pwm_to_angle_dz_trim(channel_yaw->get_dead_zone() * sub.gain, channel_yaw->get_radio_trim());',
            'float yaw_input = sub.rov_pilot_input_fresh() ? channel_yaw->pwm_to_angle_dz_trim(channel_yaw->get_dead_zone() * sub.gain, channel_yaw->get_radio_trim()) : 0.0f;')
        updated[p] = updated[p][:start] + (
            '    if (!sub.rov_pilot_attitude(target_roll, target_pitch)) {\n'
            '        if (!sub.rov_pilot_input_fresh()) { target_roll = target_pitch = 0.0f; }\n') + '\n'.join(
                '    ' + line.rstrip() if line.strip() else '' for line in stock.splitlines()) + '\n    }\n' + updated[p][end:]
    edit("ArduSub/mode_stabilize.cpp", '(channel_throttle->norm_input() + 1.0f) / 2.0f',
         '(sub.rov_pilot_input_fresh() ? (channel_throttle->norm_input() + 1.0f) / 2.0f : NEUTRAL_THROTTLE)')
    for p in ("ArduSub/mode_stabilize.cpp", "ArduSub/mode_althold.cpp"):
        edit(p, 'motors.set_forward(channel_forward->norm_input());',
             'motors.set_forward(sub.rov_pilot_input_fresh() ? channel_forward->norm_input() : 0.0f);')
        edit(p, 'motors.set_lateral(channel_lateral->norm_input());',
             'motors.set_lateral(sub.rov_pilot_input_fresh() ? channel_lateral->norm_input() : 0.0f);')
    edit("ArduSub/mode_althold.cpp", 'if (tnow - sub.set_attitude_target_no_gps.last_message_ms < 5000)',
         'if (sub.set_attitude_target_no_gps.last_message_ms != 0 && tnow - sub.set_attitude_target_no_gps.last_message_ms < 5000)')
    edit("ArduSub/mode_althold.cpp", '        float target_yaw;', '        g2.rov_pilot.reset();\n        float target_yaw;')
    edit("ArduSub/mode_althold.cpp", 'float target_climb_rate_cms = sub.get_pilot_desired_climb_rate(channel_throttle->get_control_in());',
         'float target_climb_rate_cms = sub.rov_pilot_input_fresh() ? sub.get_pilot_desired_climb_rate(channel_throttle->get_control_in()) : 0.0f;')
    edit("ArduSub/Attitude.cpp", '        attitude_control.inertial_frame_reset();',
         '        attitude_control.inertial_frame_reset();\n        g2.rov_pilot.shift_yaw(degrees(yaw_angle_change_rad));')
    telemetry = '''    CHECK_PAYLOAD_SIZE(NAMED_VALUE_FLOAT);
    send_named_float("ROV_EN", sub.g2.rov_pilot.enabled() ? 1.0f : 0.0f);
    const auto &rov_target = sub.g2.rov_pilot.output();
    const bool rov_active = sub.g2.rov_pilot.enabled() && rov_target.valid && sub.motors.armed() &&
        (sub.control_mode == Mode::Number::STABILIZE || sub.control_mode == Mode::Number::ALT_HOLD);
    CHECK_PAYLOAD_SIZE(NAMED_VALUE_FLOAT);
    send_named_float("ROV_ACT", rov_active ? 1.0f : 0.0f);
    if (rov_active) {
        CHECK_PAYLOAD_SIZE(NAMED_VALUE_FLOAT);
        send_named_float("ROV_PTGT", rov_target.pitch);
        CHECK_PAYLOAD_SIZE(NAMED_VALUE_FLOAT);
        send_named_float("ROV_YTGT", rov_target.yaw);
    }

'''
    edit("ArduSub/GCS_MAVLink_Sub.cpp",
         '    send_named_float("RFTarget", sub.mode_surftrak.get_rangefinder_target_cm() * 0.01f);',
         '    send_named_float("RFTarget", sub.mode_surftrak.get_rangefinder_target_cm() * 0.01f);\n\n' + telemetry)
    # Existing deadzone patch is folded in via exact, validated replacements.
    p = "libraries/AP_Motors/AP_Motors6DOF.cpp"
    edit(p, '#include "AP_Motors6DOF.h"', '#include "AP_Motors6DOF.h"\n#include "ROV_DeadzonePWM.h"\n#include "ROV_AttitudeAllocation.h"')
    deadzone = (ROOT / 'ardusub_deadzone.patch').read_text(encoding='utf-8')
    parameter_block = deadzone.split('+    // @Param: DZ_PWM', 1)[1].split('+    AP_GROUPEND', 1)[0]
    parameter_block = '    // @Param: DZ_PWM\n' + ''.join(line[1:] + '\n' for line in parameter_block.splitlines()[1:] if line.startswith('+'))
    edit(p, '    AP_GROUPEND', parameter_block + '''    // @Param: ATT_PRIO
    // @DisplayName: Vectored 6DOF attitude priority allocation
    // @Description: Reserve roll/pitch/yaw authority before translation. May reduce travel or vertical thrust under saturation; report limits to prevent controller windup. Disabled retains stock allocation.
    // @Values: 0:Disabled,1:Enabled
    // @User: Advanced
    AP_GROUPINFO("ATT_PRIO", 16, AP_Motors6DOF, _attitude_priority, 0),

    AP_GROUPEND''')
    edit(p, '''    int16_t range_up = get_pwm_output_max() - 1500;
    int16_t range_down = 1500 - get_pwm_output_min();
    return 1500 + thrust_in * (thrust_in > 0 ? range_up : range_down);''',
         '''    return ROV_DeadzonePWM::calculate(thrust_in,
        get_pwm_output_min(), get_pwm_output_max(),
        _startup_deadzone_pwm.get(), _startup_deadzone_input.get());''')
    edit(p, '    throttle_thrust = constrain_float(throttle_thrust, -1.0f, _max_throttle);',
         '''    const float requested_throttle = throttle_thrust;
    throttle_thrust = constrain_float(throttle_thrust, -1.0f, _max_throttle);
    if (_attitude_priority.get() == 1) {
        // Near-surface safety can clip before the allocator. Tell depth PID
        // about that lost authority as well; do not remove the surface guard.
        limit.throttle_upper = limit.throttle_upper || throttle_thrust < requested_throttle;
        limit.throttle_lower = limit.throttle_lower || throttle_thrust > requested_throttle;
    }''')
    edit(p, '    _output_limited = constrain_float(_output_limited, 0.0f, 1.0f);',
         '''    _output_limited = constrain_float(_output_limited, 0.0f, 1.0f);
    if (_attitude_priority.get() == 1 &&
        (sub_frame_t)_active_frame_class == SUB_FRAME_VECTORED_6DOF && _output_limited < 0.9999f) {
        // Preserve the stock TOTAL current safety scaling. Feedback loops must
        // know that even feasible attitude/depth requests were reduced here.
        limit.roll = limit.roll || !is_zero(_roll_in + _roll_in_ff);
        limit.pitch = limit.pitch || !is_zero(_pitch_in + _pitch_in_ff);
        limit.yaw = limit.yaw || !is_zero(_yaw_in + _yaw_in_ff);
        limit.throttle_lower = limit.throttle_lower || get_throttle_bidirectional() < 0;
        limit.throttle_upper = limit.throttle_upper || get_throttle_bidirectional() > 0;
    }''')
    branch = '''    if (_attitude_priority.get() == 1) {
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
        const auto v = ROV_AttitudeAllocation::allocate(rv, tv, vertical, AP_MOTORS_MAX_NUM_MOTORS);
        const auto h = ROV_AttitudeAllocation::allocate(rh, th, horizontal, AP_MOTORS_MAX_NUM_MOTORS);
        if (!v.valid || !h.valid) {
            limit.set_all(true);
            for (i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; ++i) { _thrust_rpyt_out[i] = 0; }
            return;
        }
        limit.roll = limit.pitch = v.rotation_scale < 0.9999f;
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

'''
    edit(p, '    // calculate roll, pitch and Throttle for each motor (only used by vertical thrusters)',
         branch + '    // calculate roll, pitch and Throttle for each motor (only used by vertical thrusters)')
    edit("libraries/AP_Motors/AP_Motors6DOF.h", '    AP_Float            _forwardVerticalCouplingFactor;',
         '    AP_Float            _forwardVerticalCouplingFactor;\n    AP_Int16            _startup_deadzone_pwm;\n    AP_Float            _startup_deadzone_input;\n    AP_Int8             _attitude_priority;')
    for name in ('ROV_PilotKernel.h', 'ROV_DepthHold.h', 'ROV_PilotHold.h', 'ROV_PilotHold.cpp', 'ROV_PilotControl.cpp'):
        updated['ArduSub/' + name] = (ROOT / name).read_text(encoding='utf-8')
    for name in ('ROV_DeadzonePWM.h', 'ROV_AttitudeAllocation.h'):
        updated['libraries/AP_Motors/' + name] = (ROOT / name).read_text(encoding='utf-8')
    return original, updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=ROOT / 'baseline_dbe79216')
    parser.add_argument('--output', type=Path, default=ROOT / 'prepared_control_20261005')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a new output folder; previous firmware evidence is preserved')
    old, new = build_sources(args.baseline)
    patch = ''.join('diff --git a/' + p + ' b/' + p + '\n' +
        ('new file mode 100644\n' if p not in old else '') + ''.join(difflib.unified_diff(
        old.get(p, '').splitlines(keepends=True), new[p].splitlines(keepends=True),
        fromfile='a/' + p if p in old else '/dev/null', tofile='b/' + p)) for p in sorted(new))
    # Check outside the enclosing project Git repository: a subdirectory check
    # can silently skip paths rather than proving that the patch applies.
    with tempfile.TemporaryDirectory() as directory:
        tree = Path(directory) / 'source'
        shutil.copytree(args.baseline, tree)
        candidate = Path(directory) / 'candidate.patch'
        candidate.write_text(patch, encoding='utf-8', newline='\n')
        subprocess.run(['git', 'apply', '--check', str(candidate)], cwd=tree, check=True)
        subprocess.run(['git', 'apply', str(candidate)], cwd=tree, check=True)
        for p, expected in new.items():
            if (tree / p).read_text(encoding='utf-8') != expected:
                raise ValueError(f'Applied patch does not reproduce prepared source: {p}')
    args.output.mkdir(parents=True, exist_ok=True)
    for p, text in new.items():
        dst = args.output / 'source' / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding='utf-8', newline='\n')
    target = args.output / 'ardusub_rov_control.patch'
    target.write_text(patch, encoding='utf-8', newline='\n')
    manifest = {'source_commit': COMMIT, 'status': 'OFFLINE_PATCH_NOT_FIRMWARE_NOT_FLASHED',
                'baseline_sha256': {p: hashlib.sha256((args.baseline / p).read_bytes()).hexdigest() for p in old},
                'patch_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'source_sha256': {p: hashlib.sha256(text.encode()).hexdigest() for p, text in new.items()}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Combined patch validated against {COMMIT}: {target}')


if __name__ == '__main__':
    main()
