"""Regression tests for the gentle roll/pitch correction release."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from firmware.prepare_gentle_attitude import ATT_LIMIT, MOTOR, MOTOR_H, sources
from firmware.activate_gentle_attitude import PARAMS


def working_gpp():
    compiler = shutil.which('g++')
    if not compiler:
        return False
    try:
        return subprocess.run([compiler, '--version'], capture_output=True,
                              timeout=5).returncode == 0
    except OSError:
        return False


HARNESS = r'''
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include "ROV_AttitudeLimit.h"
static void require(bool value, const char *message) {
    if (!value) { std::cerr << message << '\n'; std::exit(1); }
}
int main() {
    bool enabled[8] = {false,false,false,false,true,true,true,true};
    float gentle[8] = {0,0,0,0,.05f,-.05f,.05f,-.05f};
    auto unchanged = ROV_AttitudeLimit::cap(gentle, enabled, 8, .25f);
    require(unchanged.valid && std::fabs(unchanged.scale-1.0f)<1e-6f,
            "small correction was changed");
    float large[8] = {0,0,0,0,.8f,-.4f,.2f,-.8f};
    auto limited = ROV_AttitudeLimit::cap(large, enabled, 8, .25f);
    require(limited.valid && std::fabs(limited.scale-.3125f)<1e-6f,
            "large correction scale is wrong");
    for (int i=0;i<8;++i) if (enabled[i]) {
        require(std::fabs(large[i])<=.250001f, "per-thruster correction escaped cap");
    }
    float invalid[8] = {};
    auto failed = ROV_AttitudeLimit::cap(invalid, enabled, 8, NAN);
    require(!failed.valid, "invalid configuration failed open");
}
'''


class GentleAttitudeTests(unittest.TestCase):
    def test_parameter_profile_changes_only_outer_angle_gain_and_cap(self):
        self.assertEqual(PARAMS, (
            ('ATC_ANG_RLL_P', 2.0, 9),
            ('ATC_ANG_PIT_P', 2.0, 9),
            ('MOT_RP_MAX', 25.0, 9),
        ))

    def test_cap_is_parameterized_and_reports_saturation_to_both_axes(self):
        _old, updated = sources()
        motor, header = updated[MOTOR], updated[MOTOR_H]
        self.assertIn('AP_GROUPINFO("RP_MAX", 17, AP_Motors6DOF, _roll_pitch_limit_pct, 25)', motor)
        self.assertIn('AP_Float            _roll_pitch_limit_pct;', header)
        cap_call = 'ROV_AttitudeLimit::cap(rv, vertical, AP_MOTORS_MAX_NUM_MOTORS, rp_limit)'
        self.assertIn(cap_call, motor)
        self.assertIn('rp_cap.scale < 0.9999f || v.rotation_scale < 0.9999f', motor)
        self.assertLess(motor.index(cap_call), motor.index('ROV_AttitudeAllocation::allocate(rv, tv'))

    @unittest.skipUnless(working_gpp(), 'working host compiler unavailable')
    def test_exact_cap_helper_preserves_small_requests_and_bounds_large_ones(self):
        _old, updated = sources()
        # Keep the compiler harness inside the writable workspace.  Managed
        # Windows sessions may redirect the system temp folder read-only.
        with tempfile.TemporaryDirectory(prefix='rov_attitude_cap_',
                                         dir=Path(__file__).with_name('firmware')) as folder:
            folder = Path(folder)
            (folder/'ROV_AttitudeLimit.h').write_text(updated[ATT_LIMIT], encoding='utf8')
            source = folder/'test.cpp'
            source.write_text(HARNESS, encoding='utf8')
            executable = folder/'test.exe'
            result = subprocess.run(['g++', '-std=c++11', '-Wall', '-Wextra', '-Werror',
                                     '-pedantic', str(source), '-o', str(executable)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
