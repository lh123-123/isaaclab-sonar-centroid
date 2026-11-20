import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from control_profile import load_profile
from flight_mode import FlightModeController
from thrust_monitor import ThrustMonitor


def packet(kind, system=1, component=1, **fields):
    return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: system,
                           get_srcComponent=lambda: component, **fields)


class ThrustMonitorTests(unittest.TestCase):
    def setUp(self):
        self.audit = ThrustMonitor(load_profile()["pixhawk_button_parameters"])
        self.modes = FlightModeController()
        self.modes.last_heartbeat = 10
        self.modes.mode = 19

    def test_reads_only_named_parameters_and_rate_limits(self):
        link = Mock()
        self.audit.request_one(link, self.modes, 10)
        self.audit.request_one(link, self.modes, 10.1)
        self.assertEqual(link.mav.param_request_read_send.call_count, 1)
        self.audit.request_one(link, self.modes, 10.21)
        self.assertEqual(link.mav.param_request_read_send.call_count, 2)
        self.assertTrue(all(call[0] == "mav.param_request_read_send" for call in link.mock_calls))
        self.assertNotEqual(link.mock_calls[0].args[2], link.mock_calls[1].args[2])

    def test_armed_or_stale_does_not_query(self):
        link = Mock()
        self.modes.armed = True
        self.audit.request_one(link, self.modes, 10)
        self.modes.armed = False
        self.audit.request_one(link, self.modes, 14)
        self.assertEqual(link.mock_calls, [])

    def test_unconfirmed_values_are_unknown_and_wrong_source_ignored(self):
        for system, component in ((2, 1), (1, 190)):
            self.audit.observe(packet("PARAM_VALUE", system, component,
                                      param_id=b"MOT_PWM_MIN", param_value=1000), 10)
        self.assertIsNone(self.audit.value("MOT_PWM_MIN", 10))
        self.audit.observe(packet("PARAM_VALUE", param_id=b"MOT_PWM_MIN\x00", param_value=1100), 10)
        self.assertIn("MOT_PWM_MIN: 实际1100 / 目标1000", self.audit.status(self.modes, 10))
        self.assertIsNone(self.audit.value("MOT_PWM_MIN", 41))

    def test_actual_pwm_is_physical_output_not_remapped_twice(self):
        fields = {f"servo{i}_raw": 1400 + i for i in range(1, 9)}
        self.audit.observe(packet("SERVO_OUTPUT_RAW", port=0, **fields), 10)
        status = self.audit.status(self.modes, 10)
        self.assertIn("1 左前斜推 · Motor2 · 方向未知 · 1401µs", status)
        self.assertIn("8 右后垂推 · Motor7 · 方向未知 · 1408µs", status)
        self.assertNotIn("1401µs", self.audit.status(self.modes, 12.1))
        self.assertIn("离线", self.audit.status(self.modes, 14))

    def test_current_pilot_gain_distinguished_from_default_parameter(self):
        self.audit.observe(packet("PARAM_VALUE", param_id="JS_GAIN_DEFAULT", param_value=1), 10)
        self.modes.observe(packet("NAMED_VALUE_FLOAT", name="PilotGain", value=.5), 10)
        self.assertIn("运行增益不是100%", self.audit.status(self.modes, 10))

    def test_all_parameters_matching_is_not_physical_acceptance(self):
        for name, value in self.audit.expected.items():
            self.audit.observe(packet("PARAM_VALUE", param_id=name, param_value=value), 10)
        self.assertIn("仍需验证实际水流", self.audit.status(self.modes, 10))

    def test_safety_and_pid_are_readback_only_never_local_defaults(self):
        self.assertIn('ATC_RAT_PIT_P', self.audit.read_order)
        self.assertIn('PSC_D_ACC_P', self.audit.read_order)
        self.assertNotIn('PSC_ACCZ_P', self.audit.read_order)
        for name, value in (('ARMING_SKIPCHK', 1), ('FS_EKF_ACTION', 0), ('FS_PILOT_INPUT', 0),
                            ('ATC_RAT_PIT_P', .135), ('ATC_RAT_PIT_I', .135), ('ATC_RAT_PIT_D', .003)):
            self.audit.observe(packet('PARAM_VALUE', param_id=name, param_value=value), 10)
        status = self.audit.status(self.modes, 10)
        self.assertIn('被跳过的解锁检查', status)
        self.assertIn('EKF失效动作关闭', status)
        self.assertIn('手动输入失控动作关闭', status)
        self.assertIn('俯仰角速度PID P/I/D：0.135 / 0.135 / 0.003', status)
        self.assertIn('定深加速度PID P/I/D：未知 / 未知 / 未知', status)
        self.audit.observe(packet('PARAM_VALUE', param_id='ATC_RAT_PIT_P', param_value=float('nan')), 11)
        self.assertEqual(self.audit.value('ATC_RAT_PIT_P', 11), .135)
        self.modes.last_heartbeat = 42
        self.assertIn('俯仰角速度PID P/I/D：未知 / 未知 / 未知', self.audit.status(self.modes, 42))

    def test_missing_custom_reads_cannot_starve_safety_reads(self):
        link = Mock()
        count = len(self.audit.read_order)
        for step in range(count):
            now = 10 + step * .21
            self.modes.last_heartbeat = now
            self.audit.request_one(link, self.modes, now)
        queried = [call.args[2].decode('ascii') for call in link.mock_calls]
        self.assertEqual(len(set(queried)), count)
        self.assertIn('FS_EKF_ACTION', queried)
        self.assertTrue(all(call[0] == 'mav.param_request_read_send' for call in link.mock_calls))


if __name__ == "__main__":
    unittest.main()
