import math
import unittest

from attitude_display import AttitudeDisplay


class AttitudeDisplayTests(unittest.TestCase):
    def test_stationary_samples_never_create_zero_or_relative_angles(self):
        display = AttitudeDisplay()
        for armed in (False, True):
            for i in range(100):
                display.observe((8, -5, 175), i*.1, disarmed=not armed)
                self.assertEqual(display.absolute, (8, -5, 175))
                self.assertIsNone(display.reference)
                self.assertIsNone(display.relative)
                self.assertIsNone(display.continuous)

    def test_turns_and_wrap_crossings_never_accumulate(self):
        display = AttitudeDisplay()
        for i, angle in enumerate((0, 90, 179, -179, -90, 0)*20):
            display.observe((angle, 30, angle), i*.1, disarmed=False)
            self.assertEqual(display.absolute, (angle, 30, angle))
            self.assertIn(f'横滚 {angle:+.1f}°', display.angle_text(i*.1, True))
            self.assertIn(f'航向 {angle % 360:.1f}°', display.angle_text(i*.1, True))

    def test_legacy_zero_callbacks_are_noops(self):
        display = AttitudeDisplay()
        display.observe((8, -5, 175), 10, disarmed=True)
        for disarmed in (False, True):
            self.assertFalse(display.zero(10, disarmed=disarmed))
            self.assertEqual(display.absolute, (8, -5, 175))
            self.assertIsNone(display.reference)

    def test_stale_or_invalid_packets_cannot_display_fake_zero(self):
        display = AttitudeDisplay()
        display.observe((8, -5, 175), 10, disarmed=True)
        self.assertFalse(display.observe((math.nan, 0, 0), 10.1, disarmed=True))
        self.assertEqual(display.absolute, (8, -5, 175))
        self.assertIn('未知', display.angle_text(11.1, True))
        self.assertIn('未知', display.angle_text(10.1, False))

    def test_reboot_and_reconnection_use_next_native_pose(self):
        display = AttitudeDisplay()
        display.observe((8, -5, 175), 10, disarmed=True, boot_ms=100000)
        display.observe((1, 2, 3), 11, disarmed=True, boot_ms=1000)
        self.assertEqual(display.absolute, (1, 2, 3))
        self.assertIn('重启', display.notice)
        display.observe((12, 13, -170), 30, disarmed=True, boot_ms=20000)
        self.assertEqual(display.absolute, (12, 13, -170))
        self.assertIsNone(display.reference)

    def test_clock_wrap_duplicates_and_reordered_packets(self):
        display = AttitudeDisplay()
        display.observe((0, 0, 1), 10, disarmed=True, boot_ms=0xfffffff0)
        self.assertTrue(display.observe((0, 0, 2), 10.1, disarmed=True, boot_ms=50))
        self.assertFalse(display.observe((0, 0, 3), 10.2, disarmed=True, boot_ms=50))
        self.assertFalse(display.observe((0, 0, 4), 10.3, disarmed=True, boot_ms=40))
        self.assertEqual(display.absolute, (0, 0, 2))

    def test_near_vertical_pitch_is_not_transformed(self):
        display = AttitudeDisplay()
        display.observe((150, 89, -20), 10, disarmed=True)
        self.assertEqual(display.absolute, (150, 89, -20))
        self.assertIn('俯仰 +89.0°', display.angle_text(10, True))
        self.assertIn('航向 340.0°', display.raw_text(10, True))


if __name__ == '__main__':
    unittest.main()
