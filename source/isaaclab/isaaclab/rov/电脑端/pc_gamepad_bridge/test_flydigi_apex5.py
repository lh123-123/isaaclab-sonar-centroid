import math
import unittest
from dataclasses import replace

from flydigi_apex5 import (
    EXTRA_BUTTON_INDEX,
    Apex5State,
    build_manual_control,
    parse_input_report,
)


def make_report(*, extra=0, left_trigger=0, right_trigger=0):
    report = bytearray(32)
    report[0:4] = bytes((0x04, 0x5A, 0xA5, 0xEF))
    report[14] = extra
    report[16] = left_trigger
    report[17] = right_trigger
    return bytes(report)


class Apex5ReportTests(unittest.TestCase):
    def test_yaw_sensitivity_changes_only_yaw_preserves_deadzone_and_endpoints(self):
        for direction in (-1, 1):
            state = replace(Apex5State.neutral(), right_x=direction*.25, left_y=.4)
            stock = build_manual_control(state)
            sensitive = build_manual_control(state, yaw_response_exponent=.75)
            self.assertGreater(abs(sensitive.r), abs(stock.r))
            self.assertEqual((sensitive.x, sensitive.y, sensitive.z, sensitive.s),
                             (stock.x, stock.y, stock.z, stock.s))
            full = build_manual_control(replace(state, right_x=direction),
                                        yaw_response_exponent=.75, motion_scale=.6)
            self.assertEqual(full.r, direction*600)
        drift = build_manual_control(replace(Apex5State.neutral(), right_x=.02),
                                     yaw_response_exponent=.75)
        self.assertEqual(drift.r, 0)
        for invalid in (0, .1, float('nan'), float('inf'), 3):
            with self.assertRaises(ValueError):
                build_manual_control(Apex5State.neutral(), yaw_response_exponent=invalid)

    def test_six_extra_buttons_have_unique_indices(self):
        masks = {
            "M1": 0x04,
            "M2": 0x08,
            "M3": 0x10,
            "M4": 0x20,
            "LM": 0x40,
            "RM": 0x80,
        }

        for name, mask in masks.items():
            with self.subTest(name=name):
                state = parse_input_report(make_report(extra=mask))
                self.assertTrue(state.buttons & (1 << EXTRA_BUTTON_INDEX[name]))
                self.assertEqual(state.extra_buttons, frozenset((name,)))

    def test_triggers_are_combined_linearly(self):
        half_up = parse_input_report(make_report(right_trigger=128))
        half_down = parse_input_report(make_report(left_trigger=128))
        cancel = parse_input_report(
            make_report(left_trigger=173, right_trigger=173)
        )

        self.assertAlmostEqual(half_up.vertical, 128 / 255, places=6)
        self.assertAlmostEqual(half_down.vertical, -(128 / 255), places=6)
        self.assertEqual(cancel.vertical, 0.0)

    def test_manual_control_neutral_and_full_vertical(self):
        neutral = build_manual_control(Apex5State.neutral(), vertical_scale=1.0)
        self.assertEqual(neutral.z, 500)

        up = parse_input_report(make_report(right_trigger=255))
        down = parse_input_report(make_report(left_trigger=255))
        self.assertEqual(build_manual_control(up, vertical_scale=1.0).z, 1000)
        self.assertEqual(build_manual_control(down, vertical_scale=1.0).z, 0)

    def test_vertical_scale_preserves_linear_relation(self):
        up = parse_input_report(make_report(right_trigger=128))
        command = build_manual_control(up, vertical_scale=0.4)
        self.assertEqual(command.z, 600)

    def test_motion_scale_applies_to_all_motion_axes(self):
        state = Apex5State(
            left_x=0.6,
            left_y=0.8,
            right_x=0.6,
            right_y=0.8,
            left_trigger=0.0,
            right_trigger=1.0,
            buttons=0,
            extra_buttons=frozenset(),
        )
        command = build_manual_control(
            state,
            stick_deadzone=0.0,
            vertical_scale=1.0,
            motion_scale=0.4,
        )
        self.assertEqual((command.x, command.y, command.z), (320, 240, 700))
        self.assertEqual((command.r, command.s, command.t), (240, -128, 0))

    def test_lb_does_not_remap_yaw_to_roll(self):
        state = Apex5State(
            left_x=0.0,
            left_y=0.0,
            right_x=1.0,
            right_y=0.0,
            left_trigger=0.0,
            right_trigger=0.0,
            buttons=1 << 9,
            extra_buttons=frozenset(),
        )
        command = build_manual_control(state, stick_deadzone=0.0)
        self.assertEqual(command.r, 1000)
        self.assertEqual(command.t, 0)

    def test_full_motion_gear_reaches_each_protocol_endpoint(self):
        cases = (("left_y", "x", 1000), ("left_x", "y", 1000),
                 ("right_x", "r", 1000), ("right_y", "s", -400),
                 ("right_trigger", "z", 1000), ("left_trigger", "z", 0))
        for axis, field, expected in cases:
            with self.subTest(axis=axis):
                state = replace(Apex5State.neutral(), **{axis: 1.0})
                cmd = build_manual_control(state, motion_scale=1.0, vertical_scale=1.0)
                self.assertEqual(getattr(cmd, field), expected)

    def test_translation_does_not_accidentally_command_vertical_or_pitch(self):
        cmd = build_manual_control(replace(Apex5State.neutral(), left_y=1.0),
                                   motion_scale=1.0, vertical_scale=1.0)
        self.assertEqual((cmd.x, cmd.y, cmd.z, cmd.s), (1000, 0, 500, 0))

    def test_stick_and_trigger_gears_have_separate_authority(self):
        state = replace(Apex5State.neutral(), left_y=1.0, right_x=1.0, right_trigger=1.0)
        cmd = build_manual_control(state, motion_scale=0.4, vertical_motion_scale=1.0,
                                   vertical_scale=1.0)
        self.assertEqual((cmd.x, cmd.r, cmd.z), (400, 400, 1000))
        cmd = build_manual_control(state, motion_scale=1.0, vertical_motion_scale=0.4,
                                   vertical_scale=1.0)
        self.assertEqual((cmd.x, cmd.r, cmd.z), (1000, 1000, 700))
        pitch = build_manual_control(replace(Apex5State.neutral(), right_y=1.0),
                                     motion_scale=1.0, vertical_motion_scale=0.4)
        self.assertEqual(pitch.s, -400)
        down = build_manual_control(replace(Apex5State.neutral(), left_trigger=1.0),
                                    motion_scale=0.4, vertical_motion_scale=1.0, vertical_scale=1.0)
        self.assertEqual(down.z, 0)

    def test_invalid_independent_trigger_gain_is_rejected(self):
        for gain in (0, -0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                build_manual_control(Apex5State.neutral(), vertical_motion_scale=gain)

    def test_soft_axis_snap_removes_small_unintended_cross_axis(self):
        state = Apex5State(
            left_x=0.1,
            left_y=0.8,
            right_x=0.0,
            right_y=0.0,
            left_trigger=0.0,
            right_trigger=0.0,
            buttons=0,
            extra_buttons=frozenset(),
        )
        command = build_manual_control(state, stick_deadzone=0.0)
        self.assertEqual(command.x, 806)
        self.assertEqual(command.y, 0)

    def test_cardinal_assistance_preserves_stick_magnitude(self):
        state = replace(Apex5State.neutral(), left_x=0.1, left_y=0.8)
        command = build_manual_control(state, stick_deadzone=0.0)
        self.assertEqual(command.y, 0)
        self.assertAlmostEqual(math.hypot(command.x, command.y),
                               math.hypot(0.1, 0.8) * 1000, delta=1)

    def test_full_stick_near_cardinal_keeps_full_authority(self):
        angle = math.radians(7)
        state = replace(Apex5State.neutral(), left_x=math.sin(angle),
                        left_y=math.cos(angle))
        command = build_manual_control(state)
        self.assertEqual((command.x, command.y), (1000, 0))

    def test_deliberate_diagonal_input_is_preserved(self):
        state = Apex5State(
            left_x=0.6,
            left_y=0.8,
            right_x=0.0,
            right_y=0.0,
            left_trigger=0.0,
            right_trigger=0.0,
            buttons=0,
            extra_buttons=frozenset(),
        )
        command = build_manual_control(state, stick_deadzone=0.0)
        self.assertEqual((command.x, command.y), (800, 600))

    def test_arm_button_is_blocked_unless_explicitly_allowed(self):
        report = bytearray(make_report())
        report[12] = 0x10  # A button
        state = parse_input_report(bytes(report))

        blocked = build_manual_control(state, allow_arm=False)
        allowed = build_manual_control(state, allow_arm=True)
        self.assertEqual(blocked.buttons & 0x01, 0)
        self.assertEqual(allowed.buttons & 0x01, 1)

    def test_button_mask_blocks_unassigned_buttons(self):
        report = bytearray(make_report(extra=0x04))  # M1 / button 15
        report[12] = 0x10 | 0x20  # A and B
        state = parse_input_report(bytes(report))

        command = build_manual_control(
            state,
            allow_arm=True,
            button_mask=(1 << 1) | (1 << 15),
        )
        self.assertFalse(command.buttons & (1 << 0))
        self.assertTrue(command.buttons & (1 << 1))
        self.assertEqual(command.buttons2, 0)
        self.assertTrue(command.buttons & (1 << 15))

    def test_lm_rm_are_forwarded_in_buttons2(self):
        state = Apex5State.neutral()
        state = Apex5State(
            state.left_x,
            state.left_y,
            state.right_x,
            state.right_y,
            state.left_trigger,
            state.right_trigger,
            (1 << 19) | (1 << 20),
            frozenset(("LM", "RM")),
        )
        command = build_manual_control(
            state,
            button_mask=(1 << 19) | (1 << 20),
        )
        self.assertEqual(command.buttons, 0)
        self.assertEqual(command.buttons2, (1 << 3) | (1 << 4))


if __name__ == "__main__":
    unittest.main()
