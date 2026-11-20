import json
import tempfile
import unittest
from pathlib import Path

from servo_control import ServoController, save_servo_ranges


def bit(index: int) -> int:
    return 1 << index


class ServoControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = ServoController.for_competition_defaults()

    def test_gripper_hold_moves_slowly_and_can_go_below_qgc_800_limit(self):
        self.controller.sync_channel(9, 800)
        self.assertEqual(self.controller.update(bit(18), 0.0, enabled=True), ())
        commands = self.controller.update(bit(18), 1.0, enabled=True)
        self.assertEqual(commands[-1].channel, 9)
        self.assertEqual(commands[-1].pwm, 799)
        self.assertGreaterEqual(commands[-1].pwm, 500)

    def test_m1_is_up_m2_is_down(self):
        self.controller.sync_channel(10, 1500)
        self.controller.update(bit(15), 0.0, enabled=True)
        self.assertEqual(self.controller.update(bit(15), 1.0, enabled=True)[-1].pwm, 1499)

        other = ServoController.for_competition_defaults()
        other.sync_channel(10, 1500)
        other.update(bit(16), 0.0, enabled=True)
        self.assertEqual(other.update(bit(16), 1.0, enabled=True)[-1].pwm, 1501)

    def test_right_preset_is_one_guarded_command_and_slow_motion_continues(self):
        self.controller.sync_channel(10, 1500)
        command = self.controller.command_preset("angle", 920, 0.0)
        self.assertEqual((command.channel, command.pwm, command.reason), (10, 920, "angle-preset"))
        self.assertEqual(self.controller.channel_pwm(10), 920)
        with self.assertRaises(ValueError):
            self.controller.command_preset("angle", 499, 0.0)
        self.controller.update(bit(15), 0.0, enabled=True)
        self.assertEqual(self.controller.update(bit(15), 0.02, enabled=True)[-1].pwm, 919)

    def test_lm_is_counterclockwise_rm_is_clockwise(self):
        self.controller.sync_channel(11, 1500)
        self.controller.update(bit(19), 0.0, enabled=True)
        self.assertEqual(self.controller.update(bit(19), 1.0, enabled=True)[-1].pwm, 1499)

        other = ServoController.for_competition_defaults()
        other.sync_channel(11, 1500)
        other.update(bit(20), 0.0, enabled=True)
        self.assertEqual(other.update(bit(20), 1.0, enabled=True)[-1].pwm, 1501)

    def test_slow_motion_is_clamped_to_small_steps_after_a_stall(self):
        self.controller.sync_channel(9, 1500)
        self.controller.update(bit(17), 0.0, enabled=True)
        command = self.controller.update(bit(17), 10.0, enabled=True)[-1]
        self.assertEqual(command.pwm, 1501)

    def test_continuous_hold_sends_single_microsecond_steps_at_50hz(self):
        self.controller.sync_channel(9, 700)
        self.controller.update(bit(17), 0, enabled=True)
        values = []
        for i in range(1, 101):
            values.extend(c.pwm for c in self.controller.update(bit(17), i/100, enabled=True))
        self.assertEqual(values, list(range(701, 751)))
        self.assertEqual(self.controller.update(0, 1.01, enabled=True), ())
        self.assertEqual(self.controller.update(0, 2, enabled=True), ())
        self.assertEqual(self.controller.channel_pwm(9), 750)

    def test_faster_setting_changes_continuous_target_speed_without_burst(self):
        for rate in (75, 100, 150):
            with self.subTest(rate=rate):
                c = ServoController.for_competition_defaults()
                c.slow_rate_us_per_s = rate
                c.sync_channel(9, 700)
                c.update(bit(17), 0, enabled=True)
                previous = 700
                for i in range(1, 101):
                    for command in c.update(bit(17), i / 100, enabled=True):
                        self.assertLessEqual(command.pwm - previous, (rate + 49) // 50)
                        previous = command.pwm
                self.assertAlmostEqual(c.channel_pwm(9) - 700, rate, delta=1)
                before = c.channel_pwm(9)
                commands = c.update(bit(17), 10, enabled=True)
                self.assertLessEqual(commands[0].pwm - before, (rate + 49) // 50)
                self.assertEqual(c.update(0, 10.01, enabled=True), ())
                self.assertEqual(c.update(0, 11, enabled=True), ())

    def test_slow_motion_does_not_queue_catchup_or_move_past_bounds(self):
        self.controller.configure_ranges({"gripper": (500, 502)})
        self.controller.sync_channel(9, 500)
        self.controller.update(bit(17), 0, enabled=True)
        self.assertEqual(self.controller.update(bit(17), 10, enabled=True)[0].pwm, 501)
        self.assertEqual(self.controller.update(bit(17), 10.001, enabled=True), ())
        self.assertEqual(self.controller.update(bit(17), 10.021, enabled=True)[0].pwm, 502)
        self.assertEqual(self.controller.update(bit(17), 11, enabled=True), ())

    def test_rb_toggles_true_open_and_closed_endpoints(self):
        self.controller.sync_channel(9, 900)
        opened = self.controller.update(bit(10), 0.0, enabled=True)
        self.assertEqual((opened[0].channel, opened[0].pwm), (9, 2500))
        self.assertEqual(self.controller.next_gripper_action, "close")

        self.controller.update(0, 0.1, enabled=True)
        closed = self.controller.update(bit(10), 0.2, enabled=True)
        self.assertEqual((closed[0].channel, closed[0].pwm), (9, 500))
        self.assertEqual(self.controller.next_gripper_action, "open")

    def test_manual_adjustment_defines_next_rb_action(self):
        self.controller.sync_channel(9, 1500)
        self.controller.update(bit(17), 0.0, enabled=True)
        self.controller.update(bit(17), 0.1, enabled=True)
        self.assertEqual(self.controller.next_gripper_action, "close")
        self.controller.update(0, 0.2, enabled=True)
        self.assertEqual(self.controller.update(bit(10), 0.3, enabled=True)[0].pwm, 500)

        self.controller.update(0, 0.4, enabled=True)
        self.controller.update(bit(18), 0.5, enabled=True)
        self.controller.update(bit(18), 0.6, enabled=True)
        self.assertEqual(self.controller.next_gripper_action, "open")
        self.controller.update(0, 0.7, enabled=True)
        self.assertEqual(self.controller.update(bit(10), 0.8, enabled=True)[0].pwm, 2500)

    def test_disabled_controller_never_accumulates_or_emits_motion(self):
        self.controller.sync_channel(9, 1500)
        self.assertEqual(self.controller.update(bit(17), 0.0, enabled=False), ())
        self.assertEqual(self.controller.update(bit(17), 5.0, enabled=False), ())
        self.assertEqual(self.controller.channel_pwm(9), 1500)

    def test_servo_output_telemetry_seeds_position_before_first_command(self):
        class Message:
            servo9_raw = 1750
            servo10_raw = 1400
            servo11_raw = 1600

            @staticmethod
            def get_type():
                return "SERVO_OUTPUT_RAW"

        self.controller.observe(Message(), 0.0)
        self.assertEqual(self.controller.channel_pwm(9), 1750)
        self.assertEqual(self.controller.channel_pwm(10), 1400)
        self.assertEqual(self.controller.channel_pwm(11), 1600)
        self.assertEqual(self.controller.next_gripper_action, "close")

    def test_each_servo_range_is_configurable_inside_500_to_2500(self):
        self.controller.configure_ranges(
            {"gripper": (720, 2240), "angle": (650, 2100), "rotation": (900, 2300)}
        )
        self.controller.sync_channel(9, 1000)
        opened = self.controller.update(bit(10), 0.0, enabled=True)
        self.assertEqual(opened[0].pwm, 2240)
        self.controller.update(0, 0.1, enabled=True)
        closed = self.controller.update(bit(10), 0.2, enabled=True)
        self.assertEqual(closed[0].pwm, 720)
        self.assertIn("720-2240", self.controller.status_text())

    def test_servo_ranges_reject_values_outside_absolute_spec(self):
        with self.assertRaises(ValueError):
            self.controller.configure_ranges({"gripper": (499, 2000)})
        with self.assertRaises(ValueError):
            self.controller.configure_ranges({"angle": (1000, 2501)})
        with self.assertRaises(ValueError):
            self.controller.configure_ranges({"rotation": (1600, 1500)})

    def test_saved_ranges_persist_in_controller_profile(self):
        source = Path(__file__).with_name("controller_profile.json")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "profile.json"
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            save_servo_ranges(
                target,
                {"gripper": (700, 2200), "angle": (800, 2050), "rotation": (900, 2300)},
            )
            saved = json.loads(target.read_text(encoding="utf-8"))["servos"]
            self.assertEqual((saved["gripper"]["min_pwm"], saved["gripper"]["max_pwm"]), (700, 2200))
            self.assertEqual((saved["gripper"]["closed_pwm"], saved["gripper"]["open_pwm"]), (700, 2200))
            self.assertEqual((saved["angle"]["min_pwm"], saved["angle"]["max_pwm"]), (800, 2050))

    def test_individual_speed_settings_do_not_change_positions_or_fast_toggle(self):
        self.controller.configure_speeds({"gripper": 100, "angle": 150, "rotation": 75})
        before = tuple(self.controller.channel_pwm(i) for i in (9, 10, 11))
        self.assertEqual(before, (1500, 1500, 1500))
        self.controller.update(bit(17) | bit(16) | bit(20), 0, enabled=True)
        for i in range(1, 101):
            self.controller.update(bit(17) | bit(16) | bit(20), i / 100, enabled=True)
        self.assertEqual(tuple(self.controller.channel_pwm(i) for i in (9, 10, 11)), (1600, 1650, 1575))
        self.controller.update(0, 1.1, enabled=True)
        self.assertEqual(self.controller.update(bit(10), 1.2, enabled=True)[0].pwm, 500)

    def test_saved_speed_settings_survive_restart_and_preserve_other_configuration(self):
        source = Path(__file__).with_name("controller_profile.json")
        original = json.loads(source.read_text(encoding="utf-8"))
        ranges = {name: (original["servos"][name]["min_pwm"], original["servos"][name]["max_pwm"])
                  for name in ("gripper", "angle", "rotation")}
        speeds = {"gripper": 100, "angle": 140, "rotation": 80}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "profile.json"
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            save_servo_ranges(target, ranges, speeds)
            saved = json.loads(target.read_text(encoding="utf-8"))
            c = ServoController.from_profile(saved)
            self.assertEqual({name: c.speed_for(name) for name in speeds}, speeds)
            self.assertEqual(saved["pixhawk_button_parameters"], original["pixhawk_button_parameters"])
            self.assertEqual(saved["servos"]["gripper"]["initial_pwm"], original["servos"]["gripper"]["initial_pwm"])

    def test_invalid_speed_is_rejected_atomically(self):
        for value in (0, 19, 501, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.controller.configure_speeds({"gripper": 100, "angle": value, "rotation": 80})
                self.assertEqual(self.controller.speed_for("gripper"), 50)

    def test_minimum_and_maximum_configurable_speeds_stop_at_endpoints(self):
        for rate in (20, 500):
            c = ServoController.for_competition_defaults()
            c.configure_speeds({"gripper": rate, "angle": rate, "rotation": rate})
            c.configure_ranges({"gripper": (500, 600)})
            c.sync_channel(9, 599)
            c.update(bit(17), 0, enabled=True)
            for i in range(1, 201):
                c.update(bit(17), i / 100, enabled=True)
            self.assertEqual(c.channel_pwm(9), 600)
            c.update(0, 2.01, enabled=True)
            self.assertEqual(c.update(0, 10, enabled=True), ())


if __name__ == "__main__":
    unittest.main()
