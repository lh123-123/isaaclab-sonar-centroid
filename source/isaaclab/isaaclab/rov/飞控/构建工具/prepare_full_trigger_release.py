"""Build an offline successor that latches depth at true trigger neutral.

The existing deployed release and all live parameters remain unchanged. This
generator produces a separate, hash-checked patch bundle for build/flash.
"""

import hashlib
import json

try:
    from .prepare_horizontal_direction import ROOT, replace_once, write_release
except ImportError:
    from prepare_horizontal_direction import ROOT, replace_once, write_release


PARENT = ROOT / 'prepared_depth_bottom_guard_20261008'
NAME = 'prepared_full_trigger_release_safe_20261013'
HEADER = 'ArduSub/ROV_DepthHold.h'
DEPTH = 'ArduSub/mode_althold.cpp'


def sources():
    manifest = json.loads((PARENT / 'manifest.json').read_text(encoding='utf-8'))
    previous = {}
    for name, expected in manifest['source_sha256'].items():
        raw = (PARENT / 'source' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError('Current release source changed: ' + name)
        previous[name] = raw.decode('utf-8')
    updated = previous.copy()

    updated[HEADER] = replace_once(updated[HEADER],
        '''    Output update(float measured_up_cm, float climb_rate_cms,
                  uint32_t reset_ms, float reset_down_m) {''',
        '''    Output update(float measured_up_cm, float climb_rate_cms,
                  bool pilot_released, uint32_t reset_ms, float reset_down_m) {''')
    updated[HEADER] = replace_once(updated[HEADER],
        '''        if (::fabsf(climb_rate_cms) >= 0.05f) {''',
        '''        // THR_DZ can make climb_rate zero while the trigger is still
        // physically pressed. Coast with the native zero-velocity trajectory
        // until the actual neutral edge, then capture that tick's depth.
        if (!pilot_released || ::fabsf(climb_rate_cms) >= 0.05f) {''')

    updated[DEPTH] = replace_once(updated[DEPTH],
        '''    float target_climb_rate_cms = sub.rov_pilot_input_fresh() ? sub.get_pilot_desired_climb_rate(channel_throttle->get_control_in()) : 0.0f;
    target_climb_rate_cms = constrain_float(target_climb_rate_cms, -sub.get_pilot_speed_dn(), g.pilot_speed_up);
    // Only neutral-stick depth holding uses the vertical output cap.
    // The motor allocator feeds saturation back to the native depth PID.
    motors.set_depth_hold_active(fabsf(target_climb_rate_cms) < 0.05f);''',
        '''    const bool pilot_input_fresh = sub.rov_pilot_input_fresh();
    const float throttle_control = pilot_input_fresh ? channel_throttle->get_control_in() : channel_throttle->get_control_mid();
    float target_climb_rate_cms = pilot_input_fresh ? sub.get_pilot_desired_climb_rate(throttle_control) : 0.0f;
    target_climb_rate_cms = constrain_float(target_climb_rate_cms, -sub.get_pilot_speed_dn(), g.pilot_speed_up);
    // THR_DZ is much wider than the PC trigger's actual neutral. Distinguish
    // "zero requested climb" from "trigger fully released" for reference
    // capture, without weakening the existing zero-climb output cap.
    // A stale pilot stream is treated as released; it may never retain a
    // previous nonzero climb request. 5 units is 1% of half travel.
    const bool pilot_released = fabsf(target_climb_rate_cms) < 0.05f &&
        (!pilot_input_fresh || fabsf(throttle_control - channel_throttle->get_control_mid()) <= 5.0f);
    // Preserve the previous cap whenever requested climb is zero, including
    // the narrow interval before the trigger reaches exact neutral.
    // The allocator reports saturation back to the native depth PID.
    motors.set_depth_hold_active(fabsf(target_climb_rate_cms) < 0.05f);''')
    updated[DEPTH] = replace_once(updated[DEPTH],
        '''        const auto target = depth.update(inertial_nav.get_position_z_up_cm(),
                                         target_climb_rate_cms, reset_ms, reset_down_m);''',
        '''        const auto target = depth.update(inertial_nav.get_position_z_up_cm(),
                                         target_climb_rate_cms, pilot_released, reset_ms, reset_down_m);''')
    return previous, updated


if __name__ == '__main__':
    previous, updated = sources()
    write_release(ROOT / NAME, PARENT, previous, updated, {
        'depth_release': 'Capture measured depth at actual neutral input, not THR_DZ entry; retain the original zero-climb MOT_DH_MAX cap.',
        'preserve_current_live_parameters': True,
        'electrical_safety_certified': False,
        'water_acceptance': False,
    })
