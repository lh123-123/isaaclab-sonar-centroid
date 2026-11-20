import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from attitude_tolerance import AttitudeTolerance, parse_tolerance


def packet(value, *, source=1, component=1, name=b"ROV_T_ERR\x00", kind="PARAM_VALUE", param_type=9):
    return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: source,
                           get_srcComponent=lambda: component, param_id=name,
                           param_value=value, param_type=param_type)


class ToleranceTests(unittest.TestCase):
    def setUp(self):
        self.setting = AttitudeTolerance()
        self.link = Mock()
        self.modes = SimpleNamespace(fresh=lambda now: True, armed=False, mode=19, pending=None)

    def test_range_and_non_finite_values(self):
        for value in ("", "abc", -0.1, 5.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                parse_tolerance(value)
        self.assertEqual(parse_tolerance("0.5"), 0.5)
        self.assertEqual(parse_tolerance("0"), 0)

    def test_read_apply_ack_and_independent_readback(self):
        self.setting.read(self.link, 10)
        self.link.mav.param_request_read_send.assert_called_once_with(1, 1, b"ROV_T_ERR", -1)
        self.setting.observe(packet(.5, source=2), 10.1)
        self.assertTrue(self.setting.busy)
        self.setting.observe(packet(.5), 10.2)
        self.assertEqual(self.setting.actual, .5)
        self.setting.apply(self.link, self.modes, 11, "0.8", stopped=True)
        self.link.mav.param_set_send.assert_called_once_with(1, 1, b"ROV_T_ERR", .8, 9)
        self.setting.observe(packet(.8), 11.1)
        self.assertEqual(self.setting.phase, "verify")
        self.assertEqual(self.link.mav.param_request_read_send.call_count, 2)
        self.setting.observe(packet(.8), 11.2)
        self.assertFalse(self.setting.busy)
        self.assertAlmostEqual(self.setting.actual, .8)
        self.assertIn("已读回", self.setting.notice)

    def test_no_write_without_firmware_support_or_safe_state(self):
        with self.assertRaises(ValueError):
            self.setting.apply(self.link, self.modes, 10, .5, stopped=True)
        self.setting.actual = .5
        for field, value in (("armed", True), ("mode", 2), ("pending", 0)):
            before = getattr(self.modes, field)
            setattr(self.modes, field, value)
            with self.assertRaises(ValueError):
                self.setting.apply(self.link, self.modes, 10, .5, stopped=True)
            setattr(self.modes, field, before)
        with self.assertRaises(ValueError):
            self.setting.apply(self.link, self.modes, 10, .5, stopped=False)
        self.link.mav.param_set_send.assert_not_called()

    def test_timeout_and_mismatch_not_claimed(self):
        self.setting.read(self.link, 10)
        self.setting.tick(self.modes, 14)
        self.assertFalse(self.setting.busy)
        self.assertIn("可能尚不支持", self.setting.notice)
        self.setting.actual = .5
        self.setting.apply(self.link, self.modes, 20, 1, stopped=True)
        self.setting.observe(packet(.5), 20.1)
        self.assertFalse(self.setting.busy)
        self.assertEqual(self.setting.actual, .5)
        self.assertIn("未接受", self.setting.notice)

    def test_armed_during_write_aborts_confirmation(self):
        self.setting.actual = .5
        self.setting.apply(self.link, self.modes, 10, 1, stopped=True)
        self.modes.armed = True
        self.setting.tick(self.modes, 10.1)
        self.assertFalse(self.setting.busy)
        self.assertIn("未确认", self.setting.notice)


if __name__ == "__main__":
    unittest.main()
