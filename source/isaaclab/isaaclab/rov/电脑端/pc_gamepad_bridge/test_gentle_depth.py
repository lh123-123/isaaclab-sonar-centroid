"""Regression checks for neutral-stick depth output bounding."""
import unittest

from firmware.prepare_gentle_depth import DEPTH, DEPTH_ONLY, MODE, MOTOR, MOTOR_H, sources


class GentleDepthTests(unittest.TestCase):
    def test_depth_hold_uses_a_combined_vertical_envelope_and_reports_saturation(self):
        _old, updated = sources()
        motor = updated[MOTOR]
        header = updated[MOTOR_H]
        self.assertIn('AP_GROUPINFO("DH_MAX", 18, AP_Motors6DOF, _depth_hold_limit_pct, 35)', motor)
        self.assertIn('bool _depth_hold_active = false;', header)
        self.assertIn('const float vertical_envelope = _depth_hold_active ?', motor)
        self.assertIn('ROV_AttitudeAllocation::allocate(rv, tv, vertical, AP_MOTORS_MAX_NUM_MOTORS, vertical_envelope)', motor)
        self.assertIn('v.translation_scale < 0.9999f', motor)
        self.assertIn('limit.throttle_upper = limit.throttle_upper || throttle_thrust > 0;', motor)
        self.assertIn('limit.throttle_lower = limit.throttle_lower || throttle_thrust < 0;', motor)

    def test_only_neutral_depth_hold_is_capped_and_mode_exit_clears_it(self):
        _old, updated = sources()
        self.assertIn('motors.set_depth_hold_active(fabsf(target_climb_rate_cms) < 0.05f);', updated[DEPTH])
        self.assertIn('motors.set_depth_hold_active(false);', updated[DEPTH_ONLY])
        self.assertEqual(updated[MODE].count('motors.set_depth_hold_active(false);'), 2)


if __name__ == '__main__':
    unittest.main()
