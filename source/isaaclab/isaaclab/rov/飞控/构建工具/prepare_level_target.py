"""Native-angle level setpoints, no AHRS trim or coordinate conversion."""
import hashlib
import json
from pathlib import Path
try:
    from .prepare_horizontal_direction import ROOT, replace_once, write_release
except ImportError:
    from prepare_horizontal_direction import ROOT, replace_once, write_release

PARENT = ROOT / 'prepared_horizontal_20261007'
NAME = 'prepared_level_target_20261007'
KERNEL = 'ArduSub/ROV_PilotKernel.h'


def sources():
    manifest = json.loads((PARENT / 'manifest.json').read_text(encoding='utf8'))
    original = {}
    for name, expected in manifest['source_sha256'].items():
        raw = (PARENT / 'source' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError('Parent source changed: ' + name)
        original[name] = raw.decode('utf8')
    updated = original.copy()
    updated[KERNEL] = replace_once(updated[KERNEL], '    float pitch_rate = 30.0f;',
        '    float level_roll = 0;\n    float level_pitch = 0;\n    float pitch_rate = 30.0f;')
    updated[KERNEL] = replace_once(updated[KERNEL],
        '        move_angle(_out.roll, _out.roll_rate, roll_input * c.pitch_limit, dt, c);\n        move_angle(_out.pitch, _out.pitch_rate, pitch_input * c.pitch_limit, dt, c);',
        '''        float desired_roll = c.level_roll + roll_input * c.pitch_limit;
        float desired_pitch = c.level_pitch + pitch_input * c.pitch_limit;
        const float desired_tilt = ::hypotf(desired_roll, desired_pitch);
        if (desired_tilt > c.pitch_limit) {
            desired_roll *= c.pitch_limit / desired_tilt;
            desired_pitch *= c.pitch_limit / desired_tilt;
        }
        move_angle(_out.roll, _out.roll_rate, desired_roll, dt, c);
        move_angle(_out.pitch, _out.pitch_rate, desired_pitch, dt, c);''')
    updated[KERNEL] = replace_once(updated[KERNEL], '        return std::isfinite(c.pitch_rate)',
        '''        return std::isfinite(c.level_roll) && std::isfinite(c.level_pitch) &&
            ::hypotf(c.level_roll, c.level_pitch) <= 10.0f && std::isfinite(c.pitch_rate)''')
    header = 'ArduSub/ROV_PilotHold.h'
    updated[header] = replace_once(updated[header], '    bool enabled() const',
        '    bool capture_level(float roll, float pitch);\n    bool enabled() const')
    updated[header] = replace_once(updated[header], '    AP_Int8 _enable;',
        '    AP_Float _level_roll, _level_pitch;\n    AP_Int8 _enable;')
    cpp = 'ArduSub/ROV_PilotHold.cpp'
    updated[cpp] = replace_once(updated[cpp], '    AP_GROUPEND', '''    // @Param: LVL_R
    // @DisplayName: Native roll leveling target
    // @Description: Absolute native roll angle returned to on stick release. Captured with pitch while disarmed by the ROV level target command. Does not change AHRS calibration. Combined roll/pitch magnitude must not exceed 10 degrees.
    // @Units: deg
    // @Range: -10 10
    // @User: Advanced
    AP_GROUPINFO("LVL_R", 9, ROV_PilotHold, _level_roll, 0),
    // @Param: LVL_P
    // @DisplayName: Native pitch leveling target
    // @Description: Absolute native pitch leveling target. Combined roll/pitch magnitude must not exceed 10 degrees. Does not change measured attitude or heading.
    // @Units: deg
    // @Range: -10 10
    // @User: Advanced
    AP_GROUPINFO("LVL_P", 10, ROV_PilotHold, _level_pitch, 0),
    AP_GROUPEND''')
    updated[cpp] = replace_once(updated[cpp], '    c.pitch_rate = _tilt_rate.get();',
        '    c.level_roll = _level_roll.get();\n    c.level_pitch = _level_pitch.get();\n    c.pitch_rate = _tilt_rate.get();')
    updated[cpp] += '''
bool ROV_PilotHold::capture_level(float roll, float pitch) {
    if (!std::isfinite(roll) || !std::isfinite(pitch) || ::hypotf(roll, pitch) > 10.0f) {
        return false;
    }
    _level_roll.set_and_save(roll);
    _level_pitch.set_and_save(pitch);
    reset();
    return true;
}
'''
    gcs = 'ArduSub/GCS_MAVLink_Sub.cpp'
    updated[gcs] = replace_once(updated[gcs], '    case MAV_CMD_CONDITION_YAW:\n', '''    // Private ROV command, sent as COMMAND_INT. Never calibrates the AHRS.
    case MAV_CMD_USER_1: {
        if (packet.param1 != 61007.0f || packet.param2 != 1.0f) {
            return MAV_RESULT_UNSUPPORTED;
        }
        const Vector3f gyro = sub.ahrs.get_gyro();
        if (sub.motors.armed() || sub.control_mode != Mode::Number::MANUAL ||
            !sub.ahrs.healthy() || !std::isfinite(gyro.length()) || gyro.length() > 0.05f) {
            return MAV_RESULT_TEMPORARILY_REJECTED;
        }
        if (!sub.g2.rov_pilot.capture_level(degrees(sub.ahrs.roll), degrees(sub.ahrs.pitch))) {
            return MAV_RESULT_DENIED;
        }
        return MAV_RESULT_ACCEPTED;
    }

    case MAV_CMD_CONDITION_YAW:
''')
    return original, updated


if __name__ == '__main__':
    original, updated = sources()
    write_release(ROOT / NAME, PARENT, original, updated, {
        'defaults': {'MOT_H_REV': 1, 'ROV_LVL_R': 0, 'ROV_LVL_P': 0},
        'level_capture': 'MAV_CMD_USER_1 COMMAND_INT param1=61007 param2=1; native roll/pitch only, disarmed Manual, stationary, combined tilt <=10deg',
    })
