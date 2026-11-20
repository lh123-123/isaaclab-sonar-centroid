"""Regression checks for false bottom contact during bounded depth hold."""
import unittest

from firmware.prepare_depth_bottom_guard import BOTTOM, MOTOR_H, sources


class DepthBottomGuardTests(unittest.TestCase):
    def test_soft_depth_hold_limit_cannot_claim_bottom_contact(self):
        _old, updated = sources()
        detector = updated[BOTTOM]
        self.assertIn('!motors.depth_hold_active() && motors.limit.throttle_lower && vel_stationary', detector)
        self.assertIn('!motors.depth_hold_active() && motors.limit.throttle_lower)', detector)
        self.assertIn('bool depth_hold_active() const { return _depth_hold_active; }', updated[MOTOR_H])

    def test_intermittent_limit_does_not_accumulate_a_bottom_count(self):
        _old, updated = sources()
        detector = updated[BOTTOM]
        self.assertGreaterEqual(detector.count('bottom_detector_count = 0;'), 5)
        self.assertIn('bottom_detector_count = 0;\n            set_bottomed(false);', detector)


if __name__ == '__main__':
    unittest.main()
