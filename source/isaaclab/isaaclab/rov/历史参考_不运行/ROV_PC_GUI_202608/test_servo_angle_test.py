import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "servo_angle_test_under_test", ROOT / "pi_runtime" / "servo_angle_test.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakePwm:
    def __init__(self):
        self.writes = []

    def set_pulse_us(self, channel, pulse):
        self.writes.append((channel, pulse))

    def disable(self, channel):
        self.writes.append((channel, "disabled"))

    def close(self):
        self.writes.append(("closed",))


class ServoAngleTestTests(unittest.TestCase):
    SETTINGS = {
        "channel": 8,
        "pulse_min": 1000,
        "pulse_center": 1500,
        "pulse_max": 2000,
        "angle_min": 0.0,
        "angle_max": 180.0,
    }

    def test_center_pulse_is_displayed_as_90_degrees(self):
        tester = MODULE.ServoTester(self.SETTINGS, None)
        self.assertEqual(tester.pulse_for_angle(0.0), 1000)
        self.assertEqual(tester.pulse_for_angle(180.0), 2000)
        self.assertEqual(tester.angle_for_pulse(1500), 90.0)

    def test_exit_protection_reports_the_actual_center_angle(self):
        pwm = _FakePwm()
        tester = MODULE.ServoTester(self.SETTINGS, pwm)
        with mock.patch.object(MODULE.time, "sleep"):
            output = io.StringIO()
            with redirect_stdout(output):
                tester.close()
        self.assertIn("回到 90°（1500 us）", output.getvalue())
        self.assertEqual(pwm.writes[:1], [(8, 1500)])


if __name__ == "__main__":
    unittest.main()
