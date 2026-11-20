import unittest
from unittest import mock

from gamepad_adapter import GamepadAdapter
from secure_protocol import validate_control_payload


class _Buttons:
    def __init__(self, pressed=(), count=11):
        self.pressed = set(pressed)
        self.count = count

    def get_numbuttons(self):
        return self.count

    def get_button(self, index):
        return int(index in self.pressed)


def _adapter():
    adapter = GamepadAdapter.__new__(GamepadAdapter)
    adapter.gripper_open_position = 2.0 / 3.0
    adapter.gripper_closed_position = 1.0 / 18.0
    adapter.gripper_jog_open_position = 1.0
    adapter.gripper_jog_step = 0.05
    adapter.servo_angle_min = 0.0
    adapter.servo_angle_max = 180.0
    adapter.gripper_position = adapter.gripper_closed_position
    adapter.gripper_close_on_next_press = False
    adapter._send = mock.Mock()
    return adapter


class StandaloneGamepadGripperTests(unittest.TestCase):
    def test_button_layout_keeps_l3_r3_separate_from_exit(self):
        self.assertEqual(GamepadAdapter.EXIT_BUTTON, 6)
        self.assertEqual(GamepadAdapter.GRIPPER_CLOSE_JOG_BUTTON, 7)
        self.assertEqual(GamepadAdapter.GRIPPER_OPEN_JOG_BUTTON, 8)

    def test_rb_toggles_immediately_between_120_and_close_angle(self):
        adapter = _adapter()

        adapter.send_gripper_toggle()
        adapter.send_gripper_toggle()

        first, second = (call.args[0] for call in adapter._send.call_args_list)
        self.assertAlmostEqual(first["servo_position"], 2.0 / 3.0)
        self.assertAlmostEqual(second["servo_position"], 1.0 / 18.0)
        self.assertTrue(first["servo_immediate"])
        self.assertTrue(second["servo_immediate"])
        validate_control_payload(first)
        validate_control_payload(second)

    def test_r3_l3_jog_is_slow_bounded_and_sends_dynamic_minimum(self):
        adapter = _adapter()

        self.assertTrue(adapter.send_gripper_jog(1))
        opened = adapter._send.call_args.args[0]
        self.assertAlmostEqual(
            opened["servo_position"], 1.0 / 18.0 + 0.05
        )
        self.assertAlmostEqual(
            opened["servo_min_position"], 1.0 / 18.0
        )
        self.assertFalse(opened["servo_immediate"])
        validate_control_payload(opened)

        self.assertTrue(adapter.send_gripper_jog(-1))
        self.assertAlmostEqual(adapter.gripper_position, 1.0 / 18.0)
        self.assertFalse(adapter.send_gripper_jog(-1))

        adapter.gripper_position = 0.99
        self.assertTrue(adapter.send_gripper_jog(1))
        self.assertEqual(adapter.gripper_position, 1.0)
        self.assertFalse(adapter.send_gripper_jog(1))

    def test_jog_never_changes_the_next_rb_full_action(self):
        adapter = _adapter()
        adapter.gripper_position = 0.50
        adapter.gripper_close_on_next_press = True

        self.assertTrue(adapter.send_gripper_jog(-1))
        self.assertTrue(adapter.gripper_close_on_next_press)

        adapter.gripper_close_on_next_press = False
        self.assertTrue(adapter.send_gripper_jog(1))
        self.assertFalse(adapter.gripper_close_on_next_press)

    def test_both_stick_clicks_cancel_and_missing_buttons_are_safe(self):
        adapter = _adapter()
        adapter.js = _Buttons((7, 8))
        self.assertEqual(adapter.gripper_jog_direction(), 0)

        adapter.js = _Buttons((8,))
        self.assertEqual(adapter.gripper_jog_direction(), 1)

        adapter.js = _Buttons((7,))
        self.assertEqual(adapter.gripper_jog_direction(), -1)

        adapter.js = _Buttons((), count=6)
        self.assertEqual(adapter.gripper_jog_direction(), 0)


if __name__ == "__main__":
    unittest.main()
