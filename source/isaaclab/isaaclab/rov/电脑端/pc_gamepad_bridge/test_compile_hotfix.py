import unittest

from firmware.prepare_current_guard_v2 import GCS, KERNEL, sources


class CompileHotfixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.updated = sources()

    def test_level_capture_uses_public_native_centidegree_attitude(self):
        text = self.updated[GCS]
        self.assertIn('sub.ahrs.roll_sensor * 0.01f', text)
        self.assertIn('sub.ahrs.pitch_sensor * 0.01f', text)
        self.assertNotIn('degrees(sub.ahrs.roll)', text)
        self.assertNotIn('degrees(sub.ahrs.pitch)', text)

    def test_float_comparisons_are_tolerance_based(self):
        self.assertNotIn('packet.param1 != 61007.0f', self.updated[GCS])
        self.assertNotIn('yaw_input != 0', self.updated[KERNEL])
        self.assertNotIn('bounded != next', self.updated[KERNEL])

    def test_hotfix_does_not_change_mixer_or_allocator(self):
        for path in ('libraries/AP_Motors/AP_Motors6DOF.cpp',
                     'libraries/AP_Motors/ROV_AttitudeAllocation.h'):
            self.assertEqual(self.original[path], self.updated[path])


if __name__ == '__main__':
    unittest.main()
