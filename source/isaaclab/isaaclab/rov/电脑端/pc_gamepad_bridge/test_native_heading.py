import math
from types import SimpleNamespace
import unittest

from control_profile import LocalActionController, load_profile
from flight_mode import FlightModeController
from qgc_fly_status import CompetitionTelemetry, make_snapshot
from servo_control import ServoController


def message(kind, **values):
    return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: 1,
                           get_srcComponent=lambda: 1, **values)


class NativeHeadingTests(unittest.TestCase):
    def setUp(self):
        self.modes = FlightModeController()
        self.modes.observe(message('HEARTBEAT', type=12, autopilot=3,
                                  base_mode=0, custom_mode=19), 10)
        profile = load_profile()
        self.actions, self.servos = LocalActionController(profile), ServoController.from_profile(profile)

    def snapshot(self, now):
        return make_snapshot(self.modes, CompetitionTelemetry(), self.actions, self.servos, now,
            hid_fresh=True, sending=False, ready=False, allow_arm=False)

    def test_raw_yaw_is_only_heading_source_and_never_extrapolated(self):
        self.modes.observe(message('ATTITUDE', roll=.1, pitch=-.2, yaw=math.radians(-10),
                                  yawspeed=2, time_boot_ms=10000), 10)
        value = self.snapshot(10.5)
        self.assertEqual(value['heading_source'], 'ATTITUDE.yaw')
        self.assertAlmostEqual(value['heading_deg'], 350)
        self.assertAlmostEqual(value['yaw_rate_deg_s'], math.degrees(2))
        self.assertEqual(value['attitude_time_boot_ms'], 10000)
        self.assertIn('350.0°', self.modes.attitude_display.angle_text(10.5, True))
        self.assertIsNone(self.snapshot(11.1)['heading_deg'])

    def test_target_vfr_and_relative_datum_cannot_replace_measured_yaw(self):
        for n in range(15):
            self.modes.observe(message('ATTITUDE', roll=.15, pitch=-.12, yaw=math.radians(134),
                yawspeed=0, time_boot_ms=10000+50*n), 10+.05*n)
        self.modes.observe(message('NAMED_VALUE_FLOAT', name='ROV_YTGT', value=-90), 10.8)
        self.modes.observe(message('VFR_HUD', heading=44), 10.8)
        self.modes.attitude_display.zero(10.7, disarmed=True)
        self.assertAlmostEqual(self.snapshot(10.8)['heading_deg'], 134)

    def test_wrap_replay_does_not_accumulate_or_unwrap_angle(self):
        for n, degrees in enumerate((179, -179, -1, .1, 179, -179)):
            now = 10+n*.1
            self.modes.observe(message('ATTITUDE', roll=0, pitch=0, yaw=math.radians(degrees),
                                      yawspeed=.5, time_boot_ms=10000+100*n), now)
            self.assertAlmostEqual(self.snapshot(now)['heading_deg'], degrees % 360)

    def test_roll_pitch_and_heading_share_native_measurement_after_zero_attempt(self):
        self.modes.observe(message('ATTITUDE', roll=.3, pitch=-.4, yaw=-.5,
                                  time_boot_ms=10000), 10)
        self.assertFalse(self.modes.attitude_display.zero(10, disarmed=True))
        value = self.snapshot(10.1)
        self.assertAlmostEqual(value['roll_deg'], math.degrees(.3))
        self.assertAlmostEqual(value['pitch_deg'], math.degrees(-.4))
        self.assertAlmostEqual(value['heading_deg'], math.degrees(-.5) % 360)


if __name__ == '__main__':
    unittest.main()
