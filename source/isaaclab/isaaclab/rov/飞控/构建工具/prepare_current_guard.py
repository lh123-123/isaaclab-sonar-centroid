"""Integrate native measured-current limiting with the priority allocator."""
import hashlib
import json
try:
    from .prepare_horizontal_direction import ROOT, replace_once, write_release
except ImportError:
    from prepare_horizontal_direction import ROOT, replace_once, write_release

PARENT = ROOT / 'prepared_level_target_20261007'
NAME = 'prepared_current_guard_20261008'
MOTOR = 'libraries/AP_Motors/AP_Motors6DOF.cpp'
ALLOC = 'libraries/AP_Motors/ROV_AttitudeAllocation.h'


def sources():
    manifest = json.loads((PARENT / 'manifest.json').read_text(encoding='utf8'))
    original = {}
    for name, expected in manifest['source_sha256'].items():
        raw = (PARENT / 'source' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError('Parent source changed: ' + name)
        original[name] = raw.decode('utf8')
    updated = original.copy()
    allocation = updated[ALLOC]
    allocation = replace_once(allocation,
        '''inline Result allocate(float *rotation, const float *translation,
                       const bool *enabled, std::size_t count) {
    float maximum = 1.0f;''',
        '''inline Result allocate(float *rotation, const float *translation,
                       const bool *enabled, std::size_t count, float envelope = 1.0f) {
    if (!std::isfinite(envelope) || envelope < 0.0f || envelope > 1.0f) {
        return {0, 0, false};
    }
    float maximum = 0.0f;''')
    allocation = replace_once(allocation,
        '''    const float rotation_scale = 1.0f / maximum;
    float translation_scale = 1.0f;''',
        '''    const float rotation_scale = maximum > envelope && maximum > 0.0f ? envelope / maximum : 1.0f;
    float translation_scale = 1.0f;''')
    allocation = allocation.replace('(1.0f - rotation[i]) / linear', '(envelope - rotation[i]) / linear')
    allocation = allocation.replace('(-1.0f - rotation[i]) / linear', '(-envelope - rotation[i]) / linear')
    updated[ALLOC] = allocation

    source = updated[MOTOR]
    start = '''void AP_Motors6DOF::output_armed_stabilizing()
{
    if ((sub_frame_t)_active_frame_class == SUB_FRAME_VECTORED) {'''
    preamble = '''void AP_Motors6DOF::output_armed_stabilizing()
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

    if ((sub_frame_t)_active_frame_class == SUB_FRAME_VECTORED) {'''
    source = replace_once(source, start, preamble)
    old_tail = '''#if AP_BATTERY_ENABLED
    const AP_BattMonitor &battery = AP::battery();

\t// Current limiting
    float _batt_current;
    if (_batt_current_max <= 0.0f || !battery.current_amps(_batt_current)) {
        return;
    }

    float _batt_current_delta = _batt_current - _batt_current_last;

    float _current_change_rate = _batt_current_delta / _dt_s;

    float predicted_current = _batt_current + (_current_change_rate * _dt_s * 5);

    float batt_current_ratio = _batt_current / _batt_current_max;

    float predicted_current_ratio = predicted_current / _batt_current_max;
    _batt_current_last = _batt_current;

    if (predicted_current > _batt_current_max * 1.5f) {
        batt_current_ratio = 2.5f;
    } else if (_batt_current < _batt_current_max && predicted_current > _batt_current_max) {
        batt_current_ratio = predicted_current_ratio;
    }
    _output_limited += (_dt_s / (_dt_s + _batt_current_time_constant)) * (1 - batt_current_ratio);
#endif

    _output_limited = constrain_float(_output_limited, 0.0f, 1.0f);
    if (_attitude_priority.get() == 1 &&
        (sub_frame_t)_active_frame_class == SUB_FRAME_VECTORED_6DOF && _output_limited < 0.9999f) {
        // Preserve the stock TOTAL current safety scaling. Feedback loops must
        // know that even feasible attitude/depth requests were reduced here.
        limit.roll = limit.roll || !is_zero(_roll_in + _roll_in_ff);
        limit.pitch = limit.pitch || !is_zero(_pitch_in + _pitch_in_ff);
        limit.yaw = limit.yaw || !is_zero(_yaw_in + _yaw_in_ff);
        limit.throttle_lower = limit.throttle_lower || get_throttle_bidirectional() < 0;
        limit.throttle_upper = limit.throttle_upper || get_throttle_bidirectional() > 0;
    }

    for (uint8_t i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {
            _thrust_rpyt_out[i] *= _output_limited;
        }
    }'''
    new_tail = '''    // Priority Vectored-6DOF already allocated directly inside the current
    // envelope. Other frame/mixer paths retain uniform native current scaling.
    if (!priority_6dof) {
        for (uint8_t i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; i++) {
            if (motor_enabled[i]) {
                _thrust_rpyt_out[i] *= _output_limited;
            }
        }
    }'''
    source = replace_once(source, old_tail, new_tail)
    source = replace_once(source,
        'ROV_AttitudeAllocation::allocate(rv, tv, vertical, AP_MOTORS_MAX_NUM_MOTORS)',
        'ROV_AttitudeAllocation::allocate(rv, tv, vertical, AP_MOTORS_MAX_NUM_MOTORS, _output_limited)')
    source = replace_once(source,
        'ROV_AttitudeAllocation::allocate(rh, th, horizontal, AP_MOTORS_MAX_NUM_MOTORS)',
        'ROV_AttitudeAllocation::allocate(rh, th, horizontal, AP_MOTORS_MAX_NUM_MOTORS, _output_limited)')
    updated[MOTOR] = source
    return original, updated


if __name__ == '__main__':
    original, updated = sources()
    write_release(ROOT / NAME, PARENT, original, updated, {
        'defaults': {'MOT_H_REV': 1, 'ROV_LVL_R': 0, 'ROV_LVL_P': 0,
                     'MOT_BAT_CURR_MAX': 15.0, 'MOT_BAT_CURR_TC': 0.5},
        'current_guard': 'measured-current envelope; attitude first, translation second, all axes bounded if required',
        'electrical_safety_certified': False,
    })
