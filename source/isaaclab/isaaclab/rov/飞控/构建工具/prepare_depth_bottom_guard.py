"""Prepare a successor that separates depth-hold saturation from bottom contact.

The bounded vertical allocator must still report saturation to the depth PID
for anti-windup. Its software limit is not evidence of touching the bottom.
No flight controller, parameter, or service is changed by this preparation.
"""
import hashlib
import json

try:
    from .prepare_horizontal_direction import ROOT, replace_once, write_release
except ImportError:
    from prepare_horizontal_direction import ROOT, replace_once, write_release


PARENT = ROOT / 'prepared_gentle_depth_20261008'
NAME = 'prepared_depth_bottom_guard_20261008'
MOTOR_H = 'libraries/AP_Motors/AP_Motors6DOF.h'
BOTTOM = 'ArduSub/surface_bottom_detector.cpp'


def sources():
    manifest = json.loads((PARENT / 'manifest.json').read_text(encoding='utf8'))
    original = {}
    for name, expected in manifest['source_sha256'].items():
        raw = (PARENT / 'source' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError('Parent source changed: ' + name)
        original[name] = raw.decode('utf8')
    # This file was unchanged by earlier ROV releases. The base revision is
    # bundled here so the new patch can be verified without touching the Pi.
    baseline = ROOT / 'baseline_dbe79216' / BOTTOM
    original[BOTTOM] = baseline.read_text(encoding='utf8')
    updated = original.copy()

    updated[MOTOR_H] = replace_once(updated[MOTOR_H],
        '    void set_depth_hold_active(bool active) { _depth_hold_active = active; }',
        '    void set_depth_hold_active(bool active) { _depth_hold_active = active; }\n'
        '    bool depth_hold_active() const { return _depth_hold_active; }')

    detector = updated[BOTTOM]
    detector = replace_once(detector,
        '        set_bottomed(false);\n        return;',
        '        bottom_detector_count = 0;\n        set_bottomed(false);\n        return;')
    detector = replace_once(detector,
        '        if (motors.limit.throttle_lower && vel_stationary) {',
        '        // The bounded neutral-stick output reports saturation to the PID,\n'
        '        // but that software limit is not physical bottom contact.\n'
        '        if (!motors.depth_hold_active() && motors.limit.throttle_lower && vel_stationary) {')
    detector = replace_once(detector,
        '        } else {\n            set_bottomed(false);\n        }',
        '        } else {\n            bottom_detector_count = 0;\n            set_bottomed(false);\n        }')
    detector = replace_once(detector,
        '        if (motors.limit.throttle_upper) {',
        '        if (motors.limit.throttle_upper) {\n'
        '            bottom_detector_count = 0;\n'
        '            set_bottomed(false);')
    detector = replace_once(detector,
        '        } else if (motors.limit.throttle_lower) {',
        '        } else if (!motors.depth_hold_active() && motors.limit.throttle_lower) {')
    detector = replace_once(detector,
        '        } else { // we\'re not at the limits of throttle, so reset both detectors\n'
        '            set_surfaced(false);\n            set_bottomed(false);',
        '        } else { // no physical limit candidate; discard partial contact counts\n'
        '            bottom_detector_count = 0;\n'
        '            set_surfaced(false);\n            set_bottomed(false);')
    detector = replace_once(detector,
        '    } else { // we\'re moving up or down, so reset both detectors\n'
        '        set_surfaced(false);\n        set_bottomed(false);',
        '    } else { // we\'re moving up or down, so reset both detectors\n'
        '        bottom_detector_count = 0;\n'
        '        set_surfaced(false);\n        set_bottomed(false);')
    updated[BOTTOM] = detector
    return original, updated


if __name__ == '__main__':
    original, updated = sources()
    # Parent records only its modified files; BOTTOM is a newly added source
    # in this release and has no parent/source counterpart for manifesting.
    parent_sources = {name: content for name, content in original.items() if name != BOTTOM}
    write_release(ROOT / NAME, PARENT, parent_sources, updated, {
        'depth_bottom_guard': 'Neutral-stick depth output cap retains PID anti-windup but cannot trigger bottom contact or the automatic 10 cm climb.',
        'physical_bottom_detection': 'Still active on commanded descent; interrupted candidates reset their counter.',
        'defaults': {'MOT_H_REV': 1, 'MOT_RP_MAX': 25.0, 'MOT_DH_MAX': 35.0,
                     'MOT_BAT_CURR_MAX': 15.0, 'MOT_BAT_CURR_TC': 0.5},
        'live_parameter_note': 'Existing MOT_DH_MAX is preserved; last verified on vehicle was 15 percent.',
        'electrical_safety_certified': False,
    })
