"""Prepare a bounded, gentle roll/pitch correction release.

The current flashed release is kept immutable.  This successor adds a hard
combined roll/pitch actuator cap before translation is allocated.  Yaw and
normal heave authority are deliberately not capped by this parameter.
"""
import hashlib
import json

try:
    from .prepare_horizontal_direction import ROOT, replace_once, write_release
except ImportError:
    from prepare_horizontal_direction import ROOT, replace_once, write_release


PARENT = ROOT / 'prepared_current_guard_20261008_v2'
NAME = 'prepared_gentle_attitude_20261008'
MOTOR = 'libraries/AP_Motors/AP_Motors6DOF.cpp'
MOTOR_H = 'libraries/AP_Motors/AP_Motors6DOF.h'
ATT_LIMIT = 'libraries/AP_Motors/ROV_AttitudeLimit.h'


LIMIT_HEADER = r'''// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cmath>
#include <cstddef>

namespace ROV_AttitudeLimit {
struct Result { float scale; bool valid; };

// Uniformly scale the combined roll/pitch contribution so its direction is
// retained while no enabled vertical thruster receives more than `limit`.
inline Result cap(float *rotation, const bool *enabled, std::size_t count,
                  float limit) {
    if (!std::isfinite(limit) || limit < 0.0f || limit > 1.0f) {
        return {0.0f, false};
    }
    float maximum = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        if (!enabled[i]) { continue; }
        if (!std::isfinite(rotation[i])) { return {0.0f, false}; }
        maximum = ::fmaxf(maximum, ::fabsf(rotation[i]));
    }
    const float scale = maximum > limit && maximum > 0.0f ? limit / maximum : 1.0f;
    for (std::size_t i = 0; i < count; ++i) {
        if (enabled[i]) { rotation[i] *= scale; }
    }
    return {scale, true};
}
} // namespace ROV_AttitudeLimit
'''


def sources():
    manifest = json.loads((PARENT / 'manifest.json').read_text(encoding='utf8'))
    original = {}
    for name, expected in manifest['source_sha256'].items():
        raw = (PARENT / 'source' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError('Parent source changed: ' + name)
        original[name] = raw.decode('utf8')

    updated = original.copy()
    updated[ATT_LIMIT] = LIMIT_HEADER
    updated[MOTOR_H] = replace_once(
        updated[MOTOR_H],
        '    AP_Int8             _attitude_priority;\n    AP_Int8             _horizontal_reverse;',
        '    AP_Int8             _attitude_priority;\n'
        '    AP_Float            _roll_pitch_limit_pct;\n'
        '    AP_Int8             _horizontal_reverse;')

    motor = replace_once(
        updated[MOTOR],
        '#include "ROV_AttitudeAllocation.h"',
        '#include "ROV_AttitudeAllocation.h"\n#include "ROV_AttitudeLimit.h"')
    parameter = '''    // @Param: RP_MAX
    // @DisplayName: ROV roll/pitch correction maximum
    // @Description: Vectored 6DOF with ATT_PRIO enabled. Hard cap on the combined roll/pitch correction delivered to each vertical thruster. It does not cap pilot heave or yaw. 25 percent is approximately 5 N only for a 2 kgf thruster; validate in water.
    // @Units: %
    // @Range: 0 100
    // @Increment: 1
    // @User: Advanced
    AP_GROUPINFO("RP_MAX", 17, AP_Motors6DOF, _roll_pitch_limit_pct, 25),

'''
    motor = replace_once(motor, '    // @Param: H_REV\n', parameter + '    // @Param: H_REV\n')
    old = '''        const auto v = ROV_AttitudeAllocation::allocate(rv, tv, vertical, AP_MOTORS_MAX_NUM_MOTORS, _output_limited);
        const auto h = ROV_AttitudeAllocation::allocate(rh, th, horizontal, AP_MOTORS_MAX_NUM_MOTORS, _output_limited);
        if (!v.valid || !h.valid) {'''
    new = '''        const float rp_limit_pct = _roll_pitch_limit_pct.get();
        const float rp_limit = rp_limit_pct * 0.01f;
        const auto rp_cap = ROV_AttitudeLimit::cap(rv, vertical, AP_MOTORS_MAX_NUM_MOTORS, rp_limit);
        const auto v = ROV_AttitudeAllocation::allocate(rv, tv, vertical, AP_MOTORS_MAX_NUM_MOTORS, _output_limited);
        const auto h = ROV_AttitudeAllocation::allocate(rh, th, horizontal, AP_MOTORS_MAX_NUM_MOTORS, _output_limited);
        if (!rp_cap.valid || !v.valid || !h.valid) {'''
    motor = replace_once(motor, old, new)
    motor = replace_once(
        motor,
        '        limit.roll = limit.pitch = v.rotation_scale < 0.9999f;',
        '        limit.roll = limit.pitch = rp_cap.scale < 0.9999f || v.rotation_scale < 0.9999f;')
    updated[MOTOR] = motor
    return original, updated


if __name__ == '__main__':
    original, updated = sources()
    write_release(ROOT / NAME, PARENT, original, updated, {
        'defaults': {'MOT_H_REV': 1, 'ROV_LVL_R': 0, 'ROV_LVL_P': 0,
                     'MOT_BAT_CURR_MAX': 15.0, 'MOT_BAT_CURR_TC': 0.5,
                     'MOT_RP_MAX': 25.0},
        'recommended_activation': {'ATC_ANG_RLL_P': 2.0, 'ATC_ANG_PIT_P': 2.0,
                                   'MOT_RP_MAX': 25.0},
        'attitude_guard': 'combined roll/pitch capped before heave allocation; yaw unchanged; saturation fed back',
        'force_note': '25 percent is an estimate near 5 N per 2 kgf thruster, not a calibrated force value',
        'electrical_safety_certified': False,
    })
