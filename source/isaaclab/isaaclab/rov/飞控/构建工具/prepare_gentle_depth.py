"""Prepare a successor with a bounded neutral-stick depth hold output.

The bound applies to the combined vertical motor output while a depth target
is being held. Explicit pilot climb commands retain their configured range.
"""
import json

try:
    from .prepare_horizontal_direction import ROOT, replace_once, write_release
except ImportError:
    from prepare_horizontal_direction import ROOT, replace_once, write_release


PARENT = ROOT / 'prepared_gentle_attitude_20261008'
NAME = 'prepared_gentle_depth_20261008'
MOTOR = 'libraries/AP_Motors/AP_Motors6DOF.cpp'
MOTOR_H = 'libraries/AP_Motors/AP_Motors6DOF.h'
DEPTH = 'ArduSub/mode_althold.cpp'
MODE = 'ArduSub/mode.cpp'
DEPTH_ONLY = 'ArduSub/mode_depth_only.cpp'


def sources():
    import hashlib
    manifest = json.loads((PARENT / 'manifest.json').read_text(encoding='utf8'))
    original = {}
    for name, expected in manifest['source_sha256'].items():
        raw = (PARENT / 'source' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError('Parent source changed: ' + name)
        original[name] = raw.decode('utf8')
    updated = original.copy()

    updated[MOTOR_H] = replace_once(updated[MOTOR_H],
        '    void set_max_throttle(float max_throttle) { _max_throttle = max_throttle; }',
        '    void set_max_throttle(float max_throttle) { _max_throttle = max_throttle; }\n'
        '    void set_depth_hold_active(bool active) { _depth_hold_active = active; }')
    updated[MOTOR_H] = replace_once(updated[MOTOR_H],
        '    AP_Float            _roll_pitch_limit_pct;',
        '    AP_Float            _roll_pitch_limit_pct;\n'
        '    AP_Float            _depth_hold_limit_pct;')
    updated[MOTOR_H] = replace_once(updated[MOTOR_H],
        '    float _max_throttle = 1.0f;',
        '    float _max_throttle = 1.0f;\n'
        '    bool _depth_hold_active = false;')

    parameter = '''    // @Param: DH_MAX
    // @DisplayName: Neutral-stick depth hold vertical output maximum
    // @Description: Vectored 6DOF with ATT_PRIO enabled. Maximum absolute combined output of each vertical thruster while the depth stick is neutral. Pilot ascent and descent commands are not capped here. If too low to balance buoyancy, depth cannot be held; verify in water.
    // @Units: %
    // @Range: 0 100
    // @Increment: 1
    // @User: Advanced
    AP_GROUPINFO("DH_MAX", 18, AP_Motors6DOF, _depth_hold_limit_pct, 35),

'''
    updated[MOTOR] = replace_once(updated[MOTOR],
        '    // @Param: H_REV\n', parameter + '    // @Param: H_REV\n')
    updated[MOTOR] = replace_once(updated[MOTOR],
        '        const auto v = ROV_AttitudeAllocation::allocate(rv, tv, vertical, AP_MOTORS_MAX_NUM_MOTORS, _output_limited);',
        '''        const float dh_limit_pct = _depth_hold_limit_pct.get();
        const float dh_limit = dh_limit_pct * 0.01f;
        if (!std::isfinite(dh_limit) || dh_limit < 0.0f || dh_limit > 1.0f) {
            limit.set_all(true);
            for (i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; ++i) { _thrust_rpyt_out[i] = 0; }
            return;
        }
        const float vertical_envelope = _depth_hold_active ? MIN(_output_limited, dh_limit) : _output_limited;
        const auto v = ROV_AttitudeAllocation::allocate(rv, tv, vertical, AP_MOTORS_MAX_NUM_MOTORS, vertical_envelope);''')

    updated[DEPTH] = replace_once(updated[DEPTH],
        '''        g2.rov_pilot.depth_target().reset();
        return;
    }
    // return 0.2f''',
        '''        g2.rov_pilot.depth_target().reset();
        motors.set_depth_hold_active(false);
        return;
    }
    // return 0.2f''')
    updated[DEPTH] = replace_once(updated[DEPTH],
        '''    target_climb_rate_cms = constrain_float(target_climb_rate_cms, -sub.get_pilot_speed_dn(), g.pilot_speed_up);
''',
        '''    target_climb_rate_cms = constrain_float(target_climb_rate_cms, -sub.get_pilot_speed_dn(), g.pilot_speed_up);
    // Only neutral-stick depth holding uses the vertical output cap.
    // The motor allocator feeds saturation back to the native depth PID.
    motors.set_depth_hold_active(fabsf(target_climb_rate_cms) < 0.05f);
''')
    updated[MODE] = updated[MODE].replace(
        '    motors.set_max_throttle(1.0f);',
        '    motors.set_max_throttle(1.0f);\n    motors.set_depth_hold_active(false);')
    if updated[MODE].count('    motors.set_depth_hold_active(false);') != 2:
        raise RuntimeError('Expected both mode-exit paths to clear depth hold')
    updated[DEPTH_ONLY] = replace_once(updated[DEPTH_ONLY],
        '''        g2.rov_pilot.depth_target().reset();
        return;
    }''',
        '''        g2.rov_pilot.depth_target().reset();
        motors.set_depth_hold_active(false);
        return;
    }''')
    return original, updated


if __name__ == '__main__':
    original, updated = sources()
    write_release(ROOT / NAME, PARENT, original, updated, {
        'defaults': {'MOT_H_REV': 1, 'MOT_RP_MAX': 25.0, 'MOT_DH_MAX': 35.0,
                     'MOT_BAT_CURR_MAX': 15.0, 'MOT_BAT_CURR_TC': 0.5},
        'recommended_activation': {'ATC_ANG_RLL_P': 2.0, 'ATC_ANG_PIT_P': 2.0,
                                   'MOT_RP_MAX': 25.0, 'MOT_DH_MAX': 35.0},
        'depth_guard': 'neutral-stick combined vertical output envelope 35 percent; native depth PID receives saturation flags',
        'force_note': 'Output percentage is not a calibrated force. Validate holding ability and current in water.',
        'electrical_safety_certified': False,
    })
