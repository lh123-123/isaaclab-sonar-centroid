"""Make a NEW offline source bundle; never edits deployed sources or contacts hardware."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'prepared_tolerance_20261005'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Source anchor must match exactly once: ' + old[:80])
    return text.replace(old, new, 1)


def sources():
    manifest = json.loads((PREVIOUS / 'manifest.json').read_text(encoding='utf8'))
    original = {}
    for name, digest in manifest['source_sha256'].items():
        content = (PREVIOUS / 'source' / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('Previously deployed source changed: ' + name)
        original[name] = content.decode('utf8')
    new = original.copy()
    cpp = 'libraries/AP_Motors/AP_Motors6DOF.cpp'
    hdr = 'libraries/AP_Motors/AP_Motors6DOF.h'
    new[hdr] = replace_once(new[hdr], '#include "AP_MotorsMatrix.h"',
                           '#include "AP_MotorsMatrix.h"\n#include "ROV_OutputEnvelope.h"')
    new[hdr] = replace_once(new[hdr], '    void output_to_motors() override;',
                           '    void output_to_motors() override;\n'
                           '    bool output_test_num(uint8_t motor, int16_t pwm);\n'
                           '    bool rov_output_limited() const { return _rov_output_limited; }\n'
                           '    float rov_output_cap() const { return _rov_output_cap.get(); }')
    new[hdr] = replace_once(new[hdr], '    // Parameters',
                           '    void rov_guard_outputs(int16_t *pwm, float dt, bool active);\n'
                           '    void _output_test_seq(uint8_t seq, int16_t pwm) override;\n\n    // Parameters')
    new[hdr] = replace_once(new[hdr], '    AP_Int8             _attitude_priority;',
                           '    AP_Int8             _attitude_priority;\n'
                           '    AP_Float _rov_output_cap;\n    AP_Float _rov_output_sum;\n'
                           '    AP_Float _rov_output_rise;\n'
                           '    ROV_OutputEnvelope::Envelope<AP_MOTORS_MAX_NUM_MOTORS> _rov_envelope;\n'
                           '    bool _rov_output_limited = false;\n'
                           '    bool _rov_test_active = false;\n'
                           '    uint32_t _rov_test_last_us = 0;\n'
                           '    int16_t _rov_test_pwm[AP_MOTORS_MAX_NUM_MOTORS] = {};')
    parameters = '''    // @Param: OUT_CAP
    // @DisplayName: ROV final motor PWM excursion cap
    // @Description: Maximum final PWM excursion as percent of each configured directional span. Applies after mixing and startup compensation, including motor tests. NOT a measured thrust or current percentage. Zero inhibits propulsion.
    // @Range: 0 100
    // @Units: %
    // @User: Standard
    AP_GROUPINFO("OUT_CAP", 17, AP_Motors6DOF, _rov_output_cap, 20),
    // @Param: OUT_SUM
    // @DisplayName: ROV aggregate PWM excursion budget
    // @Description: Sum of absolute final PWM excursions normalized by each directional span. One is one full-span equivalent. NOT amperes or watts. Zero inhibits propulsion.
    // @Range: 0 8
    // @User: Advanced
    AP_GROUPINFO("OUT_SUM", 18, AP_Motors6DOF, _rov_output_sum, 1.0f),
    // @Param: OUT_RISE
    // @DisplayName: ROV output excursion rise rate
    // @Description: Maximum normalized excursion increase per second. Reductions and disarm are immediate; reversals cross neutral. Invalid parameters inhibit propulsion.
    // @Range: 0.01 5
    // @User: Advanced
    AP_GROUPINFO("OUT_RISE", 19, AP_Motors6DOF, _rov_output_rise, 0.5f),

'''
    new[cpp] = replace_once(new[cpp], '    AP_GROUPEND', parameters + '    AP_GROUPEND')
    new[cpp] = replace_once(new[cpp], 'void AP_Motors6DOF::output_min()\n{',
                           'void AP_Motors6DOF::output_min()\n{\n'
                           '    _rov_envelope.reset();\n    _rov_test_active = false;\n'
                           '    _rov_output_limited = false;')
    new[cpp] = replace_once(new[cpp], '    // send output to each motor',
                           '    _rov_test_active = false;\n'
                           '    const bool active = armed() && _spool_state != SpoolState::SHUT_DOWN &&\n'
                           '        _spool_state != SpoolState::GROUND_IDLE;\n'
                           '    rov_guard_outputs(motor_out, _dt_s, active);\n\n'
                           '    // send output to each motor')
    new[cpp] += '''
void AP_Motors6DOF::rov_guard_outputs(int16_t *pwm, float dt, bool active)
{
    ROV_OutputEnvelope::Config c;
    c.per_motor = _rov_output_cap.get() * 0.01f;
    c.sum = _rov_output_sum.get();
    c.rise_per_second = _rov_output_rise.get();
    int16_t bounded[AP_MOTORS_MAX_NUM_MOTORS];
    const auto result = _rov_envelope.apply(pwm, motor_enabled, bounded,
        get_pwm_output_min(), get_pwm_output_max(), dt, active, c);
    _rov_output_limited = result.limited || !result.valid;
    if (_rov_output_limited) {
        // Final electrical envelope can reduce all axes. Conservatively inform
        // every controller; never hide clipping from attitude/depth integrators.
        limit.set_all(true);
    }
    for (uint8_t i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; ++i) { pwm[i] = bounded[i]; }
}

bool AP_Motors6DOF::output_test_num(uint8_t motor, int16_t pwm)
{
    if (!armed()) { output_min(); return false; }
    if (motor >= AP_MOTORS_MAX_NUM_MOTORS || !motor_enabled[motor]) { return false; }
    const uint32_t now = AP_HAL::micros();
    float dt = 0;
    if (!_rov_test_active || uint32_t(now - _rov_test_last_us) > 100000U) {
        // Starting/restarting a test cannot inherit a preceding large output.
        _rov_envelope.reset();
        for (uint8_t i=0; i<AP_MOTORS_MAX_NUM_MOTORS; ++i) { _rov_test_pwm[i] = 1500; }
    } else { dt = uint32_t(now - _rov_test_last_us) * 1.0e-6f; }
    _rov_test_active = true;
    _rov_test_last_us = now;
    _rov_test_pwm[motor] = pwm;
    int16_t bounded[AP_MOTORS_MAX_NUM_MOTORS];
    for (uint8_t i=0; i<AP_MOTORS_MAX_NUM_MOTORS; ++i) { bounded[i] = _rov_test_pwm[i]; }
    rov_guard_outputs(bounded, dt, true);
    for (uint8_t i=0; i<AP_MOTORS_MAX_NUM_MOTORS; ++i) {
        if (motor_enabled[i]) { rc_write(i, bounded[i]); }
    }
    return true;
}

void AP_Motors6DOF::_output_test_seq(uint8_t seq, int16_t pwm)
{
    for (uint8_t i=0; i<AP_MOTORS_MAX_NUM_MOTORS; ++i) {
        if (motor_enabled[i] && _test_order[i] == seq) { output_test_num(i, pwm); }
    }
}
'''
    new['libraries/AP_Motors/ROV_OutputEnvelope.h'] = (ROOT / 'ROV_OutputEnvelope.h').read_text(encoding='utf8')
    gcs = 'ArduSub/GCS_MAVLink_Sub.cpp'
    new[gcs] = replace_once(new[gcs], '    send_named_float("ROV_EN", sub.g2.rov_pilot.enabled() ? 1.0f : 0.0f);',
                           '    send_named_float("ROV_EN", sub.g2.rov_pilot.enabled() ? 1.0f : 0.0f);\n'
                           '    CHECK_PAYLOAD_SIZE(NAMED_VALUE_FLOAT);\n'
                           '    send_named_float("ROV_CAP", sub.motors.rov_output_cap());\n'
                           '    CHECK_PAYLOAD_SIZE(NAMED_VALUE_FLOAT);\n'
                           '    send_named_float("ROV_LIM", sub.motors.rov_output_limited() ? 1.0f : 0.0f);')
    return original, new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'prepared_output_envelope_20261007')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Do not overwrite prepared artifacts; choose a new directory')
    previous, new = sources()
    baseline = ROOT / 'baseline_dbe79216'
    old = {p: (baseline / p).read_text(encoding='utf8') for p in new if (baseline / p).is_file()}
    patch = ''.join('diff --git a/' + p + ' b/' + p + '\n' +
                    ('new file mode 100644\n' if p not in old else '') +
                    ''.join(difflib.unified_diff(old.get(p, '').splitlines(keepends=True),
                        new[p].splitlines(keepends=True), fromfile='a/'+p if p in old else '/dev/null',
                        tofile='b/'+p)) for p in sorted(new))
    with tempfile.TemporaryDirectory() as temporary:
        tree = Path(temporary) / 'source'
        shutil.copytree(baseline, tree)
        candidate = Path(temporary) / 'candidate.patch'
        candidate.write_text(patch, encoding='utf8', newline='\n')
        subprocess.run(['git', 'apply', '--check', str(candidate)], cwd=tree, check=True)
        subprocess.run(['git', 'apply', str(candidate)], cwd=tree, check=True)
        for p, content in new.items():
            if (tree / p).read_text(encoding='utf8') != content:
                raise RuntimeError('Patch source parity failed: ' + p)
    args.output.mkdir(parents=True)
    for p, content in new.items():
        destination = args.output / 'source' / p
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding='utf8', newline='\n')
    (args.output / 'ardusub_rov_control.patch').write_text(patch, encoding='utf8', newline='\n')
    record = {'status': 'OFFLINE_NOT_BUILT_NOT_FLASHED_NOT_WATER_VALIDATED',
              'source_commit': 'dbe792162d06cab66c3475fd5556bf7a120f119e',
              'parent': PREVIOUS.name,
              'source_sha256': {p: hashlib.sha256((args.output / 'source' / p).read_bytes()).hexdigest() for p in new},
              'parent_source_sha256': {p: hashlib.sha256((PREVIOUS / 'source' / p).read_bytes()).hexdigest() for p in previous},
              'baseline_sha256': {p: hashlib.sha256((baseline / p).read_bytes()).hexdigest() for p in old},
              'patch_sha256': hashlib.sha256((args.output / 'ardusub_rov_control.patch').read_bytes()).hexdigest(),
              'defaults': {'MOT_OUT_CAP': 20, 'MOT_OUT_SUM': 1, 'MOT_OUT_RISE': 0.5},
              'electrical_safety_certified': False}
    (args.output / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(args.output)


if __name__ == '__main__':
    main()
