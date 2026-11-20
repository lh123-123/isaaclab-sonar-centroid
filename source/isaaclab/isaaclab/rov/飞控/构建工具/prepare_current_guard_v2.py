"""Prepare a compile-clean successor to the reviewed current-guard release.

The parent release is immutable.  This release changes only the level-capture
AHRS accessor and removes direct floating-point comparisons rejected by the
Pixhawk1 target compiler's warning policy.
"""
import hashlib
import json

try:
    from .prepare_horizontal_direction import ROOT, replace_once, write_release
except ImportError:
    from prepare_horizontal_direction import ROOT, replace_once, write_release


PARENT = ROOT / 'prepared_current_guard_20261008'
NAME = 'prepared_current_guard_20261008_v2'
GCS = 'ArduSub/GCS_MAVLink_Sub.cpp'
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
    updated[GCS] = replace_once(
        updated[GCS],
        '        if (packet.param1 != 61007.0f || packet.param2 != 1.0f) {',
        '        if (::fabsf(packet.param1 - 61007.0f) > 0.01f || ::fabsf(packet.param2 - 1.0f) > 0.01f) {')
    updated[GCS] = replace_once(
        updated[GCS],
        '        if (!sub.g2.rov_pilot.capture_level(degrees(sub.ahrs.roll), degrees(sub.ahrs.pitch))) {',
        '        if (!sub.g2.rov_pilot.capture_level(sub.ahrs.roll_sensor * 0.01f, sub.ahrs.pitch_sensor * 0.01f)) {')

    yaw_anchor = '        if (yaw_input != 0) {'
    if updated[KERNEL].count(yaw_anchor) != 2:
        raise RuntimeError('Unexpected yaw-input comparison count')
    updated[KERNEL] = updated[KERNEL].replace(
        yaw_anchor, '        if (::fabsf(yaw_input) > 1.0e-6f) {')
    updated[KERNEL] = replace_once(
        updated[KERNEL], '            if (bounded != next) { rate = 0; }',
        '            if (::fabsf(bounded - next) > 1.0e-6f) { rate = 0; }')
    return original, updated


if __name__ == '__main__':
    original, updated = sources()
    write_release(ROOT / NAME, PARENT, original, updated, {
        'defaults': {'MOT_H_REV': 1, 'ROV_LVL_R': 0, 'ROV_LVL_P': 0,
                     'MOT_BAT_CURR_MAX': 15.0, 'MOT_BAT_CURR_TC': 0.5},
        'level_capture': 'native roll_sensor/pitch_sensor centidegrees; disarmed Manual and stationary only',
        'current_guard': 'measured-current envelope; attitude first, translation second, all axes bounded if required',
        'compile_hotfix': 'use public AHRS centidegree accessors and tolerance-based float comparisons',
        'electrical_safety_certified': False,
    })
