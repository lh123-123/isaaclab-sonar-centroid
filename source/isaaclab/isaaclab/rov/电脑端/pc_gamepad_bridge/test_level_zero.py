import math
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

from level_zero import LevelZero


def message(kind, system=1, component=1, **values):
    return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: system,
                           get_srcComponent=lambda: component, **values)


class LevelZeroTests(unittest.TestCase):
    def setUp(self):
        self.zero = LevelZero()
        self.link = Mock()
        self.modes = SimpleNamespace(fresh=lambda now: True, armed=False, mode=19, pending=None)
        self.feed(10, 5, -3)

    def feed(self, end, roll=0, pitch=0, yaw=175, rates=(0, 0, 0)):
        for i in range(13):
            self.zero.observe(message('ATTITUDE', roll=math.radians(roll), pitch=math.radians(pitch),
                yaw=math.radians(yaw), rollspeed=rates[0], pitchspeed=rates[1], yawspeed=rates[2]), end - 1.2 + .1 * i)

    def start(self):
        self.zero.start(self.link, self.modes, 10, stopped=True, horizontal_confirmed=True)

    def params(self, now, orientation=0, roll=0, pitch=0, yaw=0):
        for name, value in zip(self.zero.PARAMS, (orientation, roll, pitch, yaw)):
            self.zero.observe(message('PARAM_VALUE', param_id=name, param_value=value,
                              param_type=2 if name == 'AHRS_ORIENTATION' else 9), now)

    def test_no_startup_calibration_and_explicit_gates(self):
        self.zero.tick(self.modes, 10)
        self.assertEqual(self.link.mock_calls, [])
        for field, value in (('armed', True), ('mode', 2), ('pending', 0)):
            previous = getattr(self.modes, field)
            setattr(self.modes, field, value)
            with self.assertRaises(ValueError):
                self.start()
            setattr(self.modes, field, previous)
        with self.assertRaises(ValueError):
            self.zero.start(self.link, self.modes, 10, stopped=False, horizontal_confirmed=True)
        with self.assertRaises(ValueError):
            self.zero.start(self.link, self.modes, 10, stopped=True, horizontal_confirmed=False)

    def test_native_command_ack_trim_and_actual_angles_required(self):
        self.start()
        self.link.mav.command_long_send.assert_not_called()
        self.params(10.1)
        self.link.mav.command_long_send.assert_called_once_with(1, 1, 241, 0, 0, 0, 0, 0, 2, 0, 0)
        self.zero.observe(message('COMMAND_ACK', command=241, result=0), 11)
        self.params(11.1, roll=.04, pitch=-.02)
        self.zero.tick(self.modes, 11.2)
        self.assertTrue(self.zero.busy)
        self.feed(12.5, .2, -.1, 176)
        self.zero.tick(self.modes, 12.5)
        self.assertFalse(self.zero.busy)
        self.assertFalse(self.zero.report['yaw_reference_reset'])
        self.assertAlmostEqual(self.zero.report['actual_attitude_deg'][2], 176)
        self.link.mav.param_set_send.assert_not_called()
        self.link.mav.manual_control_send.assert_not_called()

    def test_rejection_wrong_source_and_timeout_never_claim_zero(self):
        self.start()
        self.params(10.1)
        self.zero.observe(message('COMMAND_ACK', system=2, command=241, result=0), 11)
        self.assertEqual(self.zero.phase, 'calibrate')
        self.zero.observe(message('COMMAND_ACK', command=241, result=2), 11.1)
        self.assertIsNone(self.zero.report)
        self.assertFalse(self.zero.busy)
        self.feed(12)
        self.zero.start(self.link, self.modes, 12, stopped=True, horizontal_confirmed=True)
        self.zero.tick(self.modes, 18)
        self.assertIsNone(self.zero.report)
        self.assertIn('未确认', self.zero.notice)

    def test_installed_rotation_or_yaw_trim_refused(self):
        for orientation, yaw in ((101, 0), (0, .01)):
            self.zero = LevelZero()
            self.feed(10, 5, -3)
            self.start()
            self.params(10.1, orientation=orientation, yaw=yaw)
            self.assertFalse(self.zero.busy)
            self.link.mav.command_long_send.assert_not_called()

    def test_motion_and_excess_angle_refused(self):
        for rates, roll in (((.2, 0, 0), 0), ((0, 0, 0), 12)):
            self.zero.samples.clear()
            self.feed(10, roll, rates=rates)
            with self.assertRaises(ValueError):
                self.start()

    def test_ack_not_enough_and_no_automatic_retries(self):
        self.start()
        self.params(10.1)
        self.zero.observe(message('COMMAND_ACK', command=241, result=0), 11)
        self.params(11.1, roll=.04, pitch=-.02)
        self.feed(12.5, 5, -3)
        self.zero.tick(self.modes, 19.1)
        self.assertIsNone(self.zero.report)
        self.assertFalse(self.zero.busy)
        self.link.mav.command_long_send.assert_called_once()


if __name__ == '__main__':
    unittest.main()
