"""Prepare a separate direction-fix release from the verified deployed source.

No serial/network access, parameter writes, or changes to the deployed artifacts.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / 'prepared_tolerance_20261005'
NAME = 'prepared_horizontal_20261007'
MOTOR_CPP = 'libraries/AP_Motors/AP_Motors6DOF.cpp'
MOTOR_H = 'libraries/AP_Motors/AP_Motors6DOF.h'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected exactly one source anchor: ' + old[:100])
    return text.replace(old, new, 1)


def sources():
    manifest = json.loads((PARENT / 'manifest.json').read_text(encoding='utf8'))
    original = {}
    for name, expected in manifest['source_sha256'].items():
        raw = (PARENT / 'source' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Deployed source hash mismatch: ' + name)
        original[name] = raw.decode('utf8')
    updated = original.copy()
    updated[MOTOR_H] = replace_once(updated[MOTOR_H],
        '    AP_Int8             _attitude_priority;',
        '    AP_Int8             _attitude_priority;\n    AP_Int8             _horizontal_reverse;')
    parameters = '''    // @Param: H_REV
    // @DisplayName: ROV lateral and yaw direction correction
    // @Description: Vectored 6DOF only. Reverses lateral force and yaw torque before allocation for this ROV assembly. Covers manual and automatic heading correction. Forward, heave, pitch, roll, individual motor directions and PWM endpoints are unchanged. Change only while disarmed; verify physical heading feedback before enabling stabilization.
    // @Values: 0:Original geometry,1:Reverse lateral and yaw
    // @User: Advanced
    AP_GROUPINFO("H_REV", 20, AP_Motors6DOF, _horizontal_reverse, 1),

'''
    updated[MOTOR_CPP] = replace_once(updated[MOTOR_CPP], '    AP_GROUPEND',
                                      parameters+'    AP_GROUPEND')
    marker = 'void AP_Motors6DOF::output_armed_stabilizing_vectored_6dof()'
    prefix, body = updated[MOTOR_CPP].split(marker, 1)
    body = replace_once(body, '    lateral_thrust = _lateral_in;\n', '''    lateral_thrust = _lateral_in;

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
''')
    updated[MOTOR_CPP] = prefix+marker+body
    updated[MOTOR_CPP] = replace_once(updated[MOTOR_CPP],
        '    return Vector3f(_roll_factor[motor_number], _pitch_factor[motor_number], _yaw_factor[motor_number]);',
        '''    const float yaw_sign = ((sub_frame_t)_active_frame_class == SUB_FRAME_VECTORED_6DOF &&
                            _horizontal_reverse.get() == 1) ? -1.0f : 1.0f;
    return Vector3f(_roll_factor[motor_number], _pitch_factor[motor_number], yaw_sign * _yaw_factor[motor_number]);''')
    return original, updated


def write_release(output, parent, previous, updated, extra=None):
    args = argparse.Namespace(output=output)
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite an existing release')
    baseline = ROOT / 'baseline_dbe79216'
    old = {p: (baseline / p).read_text(encoding='utf8')
           for p in updated if (baseline / p).is_file()}
    patch = ''.join('diff --git a/' + p + ' b/' + p + '\n' +
                    ('new file mode 100644\n' if p not in old else '') +
                    ''.join(difflib.unified_diff(old.get(p, '').splitlines(keepends=True),
                        updated[p].splitlines(keepends=True),
                        fromfile='a/'+p if p in old else '/dev/null', tofile='b/'+p))
                    for p in sorted(updated))
    # Keep patch verification inside the workspace. Some managed Windows
    # sessions expose a non-writable redirected system temp directory.
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        tree = Path(temporary) / 'source'
        shutil.copytree(baseline, tree)
        patch_file = Path(temporary) / 'candidate.patch'
        patch_file.write_text(patch, encoding='utf8', newline='\n')
        git_env = os.environ.copy()
        git_env['GIT_CEILING_DIRECTORIES'] = str(ROOT)
        subprocess.run(['git', 'apply', '--check', str(patch_file)], cwd=tree, env=git_env, check=True)
        subprocess.run(['git', 'apply', str(patch_file)], cwd=tree, env=git_env, check=True)
        for name, content in updated.items():
            actual_content = (tree / name).read_text(encoding='utf8')
            if actual_content != content:
                mismatch = next((i for i, pair in enumerate(zip(actual_content, content))
                                 if pair[0] != pair[1]), min(len(actual_content), len(content)))
                raise RuntimeError(f'Patch parity failure: {name} at {mismatch}; '
                                   f'actual={len(actual_content)} expected={len(content)}')
    args.output.mkdir(parents=True)
    for name, content in updated.items():
        path = args.output / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf8', newline='\n')
    patch_path = args.output / 'ardusub_rov_control.patch'
    patch_path.write_text(patch, encoding='utf8', newline='\n')
    record = {
        'status': 'OFFLINE_NOT_BUILT_NOT_FLASHED_NOT_WATER_VALIDATED',
        'source_commit': 'dbe792162d06cab66c3475fd5556bf7a120f119e',
        'parent': parent.name,
        'source_sha256': {p: hashlib.sha256((args.output / 'source' / p).read_bytes()).hexdigest() for p in updated},
        'parent_source_sha256': {p: hashlib.sha256((parent / 'source' / p).read_bytes()).hexdigest() for p in previous},
        'baseline_sha256': {p: hashlib.sha256((baseline / p).read_bytes()).hexdigest() for p in old},
        'patch_sha256': hashlib.sha256(patch_path.read_bytes()).hexdigest(),
        'defaults': {'MOT_H_REV': 1},
        'preserve_existing_parameters': ['MOT_PWM_MIN', 'MOT_PWM_MAX', 'MOT_1_DIRECTION..MOT_8_DIRECTION', 'SERVO1_FUNCTION..SERVO8_FUNCTION'],
        'assumption': 'Reported manual forward/heave correct, lateral/yaw reversed. Native heading sign still requires physical verification.',
    }
    if extra:
        record.update(extra)
    (args.output / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / NAME)
    args = parser.parse_args()
    previous, updated = sources()
    write_release(args.output, PARENT, previous, updated)


if __name__ == '__main__':
    main()
