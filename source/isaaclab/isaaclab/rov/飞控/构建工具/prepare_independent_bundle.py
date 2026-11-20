"""Offline unified firmware for four independent X/Y combinations. Never flashes."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from prepare_control_bundle import ROOT, COMMIT, build_sources, replace_once


def sources(baseline):
    old, new = build_sources(baseline)
    for path in ('ArduSub/mode.h', 'ArduSub/mode.cpp', 'ArduSub/Sub.cpp'):
        old[path] = (baseline / path).read_text(encoding='utf-8')
        new[path] = old[path]
    if 'control_mode(Mode::Number::MANUAL)' not in new['ArduSub/Sub.cpp']:
        raise ValueError('Unexpected startup mode: default must be manual')
    def edit(path, before, after):
        new[path] = replace_once(new[path], before, after)
    edit('ArduSub/mode.h', '        SURFTRAK =     21   // Track distance above seafloor (hold range)',
         '        SURFTRAK =     21,  // Track distance above seafloor (hold range)\n'
         '        DEPTH_ONLY =   22   // Private pinned ROV mode: pressure hold, manual attitude')
    declaration = '''class ModeDepthOnly : public ModeAlthold
{
public:
    using ModeAlthold::ModeAlthold;
    bool init(bool ignore_checks) override;
    void run() override;
protected:
    const char *name() const override { return "Depth Only"; }
    const char *name4() const override { return "DPNO"; }
    Mode::Number number() const override { return Mode::Number::DEPTH_ONLY; }
};

'''
    edit('ArduSub/mode.h', 'class ModeSurftrak : public ModeAlthold', declaration + 'class ModeSurftrak : public ModeAlthold')
    edit('ArduSub/Sub.h', '    ModeAlthold mode_althold;', '    ModeAlthold mode_althold;\n    ModeDepthOnly mode_depth_only;')
    edit('ArduSub/Sub.h', '    friend class ModeAlthold;',
         '    friend class ModeAlthold;\n    friend class ModeDepthOnly;')
    edit('ArduSub/mode.cpp', '    case Mode::Number::ALT_HOLD:',
         '    case Mode::Number::DEPTH_ONLY:\n        ret = &mode_depth_only;\n        break;\n    case Mode::Number::ALT_HOLD:')
    # Fast-loop PID must not overwrite the direct angular demands of depth-only mode.
    edit('ArduSub/Sub.cpp', 'control_mode != Mode::Number::MANUAL && control_mode != Mode::Number::MOTOR_DETECT',
         'control_mode != Mode::Number::MANUAL && control_mode != Mode::Number::MOTOR_DETECT && control_mode != Mode::Number::DEPTH_ONLY')
    edit('ArduSub/mode_althold.cpp', '    position_control->D_init_controller();',
         '    // Preserve the existing depth target when toggling attitude during depth hold.\n'
         '    if (sub.control_mode != Mode::Number::ALT_HOLD && sub.control_mode != Mode::Number::DEPTH_ONLY) {\n'
         '        position_control->D_init_controller();\n'
         '        g2.rov_pilot.depth_target().reset();\n    }')
    edit('ArduSub/mode_althold.cpp', 'void ModeAlthold::control_depth() {',
         '''void ModeAlthold::control_depth() {
    const bool release_hold = g2.rov_pilot.enabled() &&
        (sub.control_mode == Mode::Number::ALT_HOLD || sub.control_mode == Mode::Number::DEPTH_ONLY);
    if (release_hold && !motors.armed()) {
        // run_pre() returning while disarmed does not exit ModeAlthold::run().
        // Do not resurrect a target after its controller has been relaxed.
        g2.rov_pilot.depth_target().reset();
        return;
    }''')
    edit('ArduSub/mode_althold.cpp',
         '    position_control->D_set_pos_target_from_climb_rate_cms(target_climb_rate_cms);',
         '''    if (release_hold) {
        float reset_down_m = 0.0f;
        const uint32_t reset_ms = ahrs.getLastPosDownReset(reset_down_m);
        auto &depth = g2.rov_pilot.depth_target();
        const auto target = depth.update(inertial_nav.get_position_z_up_cm(),
                                         target_climb_rate_cms, reset_ms, reset_down_m);
        if (target.valid && target.hold) {
            // Safety overrides the release reference near the surface/bottom.
            // Store the override so departing the boundary cannot revive an
            // old unsafe reference. Both remain Up-positive centimeters.
            if (sub.ap.at_surface) {
                depth.limit_target(-1.0e20f, g.surface_depth);
            } else if (sub.ap.at_bottom) {
                depth.limit_target(inertial_nav.get_position_z_up_cm() + 10.0f, 1.0e20f);
            }
            // A FIXED reference, not climb(0) which integrates deceleration.
            // Native jerk/accel shaping and pressure/velocity PID brake inertia.
            position_control->set_alt_target_with_slew_cm(depth.output().altitude_cm);
        } else {
            position_control->D_set_pos_target_from_climb_rate_cms(target.valid ? target_climb_rate_cms : 0.0f);
        }
    } else {
        g2.rov_pilot.depth_target().reset();
        position_control->D_set_pos_target_from_climb_rate_cms(target_climb_rate_cms);
    }''')
    edit('ArduSub/GCS_MAVLink_Sub.cpp', '        &sub.mode_althold,',
         '        &sub.mode_althold,\n        &sub.mode_depth_only,')
    edit('ArduSub/GCS_MAVLink_Sub.cpp', '    send_named_float("ROV_EN", sub.g2.rov_pilot.enabled() ? 1.0f : 0.0f);',
         '    send_named_float("ROV_EN", sub.g2.rov_pilot.enabled() ? 1.0f : 0.0f);\n'
         '    CHECK_PAYLOAD_SIZE(NAMED_VALUE_FLOAT);\n    send_named_float("ROV_DONLY", 1.0f);')
    edit('ArduSub/GCS_MAVLink_Sub.cpp', 'uint8_t _base_mode = MAV_MODE_FLAG_STABILIZE_ENABLED;',
         'uint8_t _base_mode = (sub.control_mode == Mode::Number::MANUAL ||\n'
         '                          sub.control_mode == Mode::Number::DEPTH_ONLY) ? 0 : MAV_MODE_FLAG_STABILIZE_ENABLED;')
    new['ArduSub/mode_depth_only.cpp'] = (ROOT / 'mode_depth_only.cpp').read_text(encoding='utf-8')
    return old, new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=ROOT / 'baseline_dbe79216')
    parser.add_argument('--output', type=Path, default=ROOT / 'prepared_independent_20261005')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a new output folder; previous firmware evidence is preserved')
    old, new = sources(args.baseline)
    patch = ''.join('diff --git a/' + p + ' b/' + p + '\n' +
        ('new file mode 100644\n' if p not in old else '') + ''.join(difflib.unified_diff(
        old.get(p, '').splitlines(keepends=True), new[p].splitlines(keepends=True),
        fromfile='a/' + p if p in old else '/dev/null', tofile='b/' + p)) for p in sorted(new))
    with tempfile.TemporaryDirectory() as directory:
        tree = Path(directory) / 'source'
        shutil.copytree(args.baseline, tree)
        candidate = Path(directory) / 'control.patch'
        candidate.write_text(patch, encoding='utf-8', newline='\n')
        subprocess.run(['git', 'apply', '--check', str(candidate)], cwd=tree, check=True)
        subprocess.run(['git', 'apply', str(candidate)], cwd=tree, check=True)
        for p, expected in new.items():
            if (tree / p).read_text(encoding='utf-8') != expected:
                raise RuntimeError('Patch payload mismatch: ' + p)
    for p, content in new.items():
        dst = args.output / 'source' / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding='utf-8', newline='\n')
    target = args.output / 'ardusub_rov_control.patch'
    target.write_text(patch, encoding='utf-8', newline='\n')
    manifest = {'source_commit': COMMIT, 'status': 'OFFLINE_NOT_BUILT_NOT_FLASHED',
                'depth_only_mode': 22, 'capability': 'ROV_DONLY=1',
                'baseline_sha256': {p: hashlib.sha256((args.baseline / p).read_bytes()).hexdigest() for p in old},
                'source_sha256': {p: hashlib.sha256(t.encode()).hexdigest() for p, t in new.items()},
                'patch_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Independent control patch verified; NOT built/flashed:', args.output)


if __name__ == '__main__':
    main()
