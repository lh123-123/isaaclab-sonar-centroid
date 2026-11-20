import unittest

from config_loader import load_config


class ThrusterPwmLimitConfigTests(unittest.TestCase):
    def test_group_limits_start_at_twenty_percent_under_a_half_power_ceiling(self):
        config = load_config()
        neutral = config.getint("hardware", "pwm_neutral_us")
        fraction = config.getfloat("thrusters", "max_command_fraction")
        lows = [int(value) for value in config.get("thrusters", "pwm_min_us").split(",")]
        highs = [int(value) for value in config.get("thrusters", "pwm_max_us").split(",")]

        self.assertEqual(neutral, 1540)
        self.assertEqual(fraction, 0.50)
        self.assertEqual(config.getfloat("thrusters", "horizontal_limit_fraction"), 0.20)
        self.assertEqual(config.getfloat("thrusters", "vertical_limit_fraction"), 0.30)
        self.assertEqual(lows, [1316] * 8)
        self.assertEqual(highs, [1714] * 8)
        self.assertEqual(config.getint("thrusters", "reverse_start_us"), 1532)
        self.assertEqual(config.getint("thrusters", "forward_start_us"), 1578)
        hover = config.getfloat("control", "vertical_hover_force_per_thruster_n")
        self.assertGreaterEqual(hover, 0.0)
        self.assertLessEqual(hover, 55.0 * 0.30)
        # These four values are operator tuning that the Pi persists at
        # runtime; tests enforce their safety range instead of overwriting a
        # field-calibrated value with an old fixture number.
        for option in (
            "vertical_ascent_front_rear_trim_percent",
            "vertical_descent_front_rear_trim_percent",
            "forward_front_rear_trim_percent",
            "backward_front_rear_trim_percent",
        ):
            self.assertLessEqual(
                abs(config.getfloat("control", option)), 100.0
            )
        for section in ("mode_balance", "mode_rage"):
            for option in (
                "left_lateral_surge_trim_percent",
                "right_lateral_surge_trim_percent",
            ):
                self.assertLessEqual(abs(config.getfloat(section, option)), 100.0)
            for option in (
                "roll_max_moment_nm",
                "pitch_max_moment_nm",
                "yaw_max_moment_nm",
            ):
                self.assertGreaterEqual(config.getfloat(section, option), 0.1)
                self.assertLessEqual(config.getfloat(section, option), 20.0)
        for option in (
            "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm"
        ):
            self.assertGreaterEqual(config.getfloat("control", option), 0.1)
            self.assertLessEqual(config.getfloat("control", option), 20.0)
        # Roll/pitch gains are operator tuning persisted by the Pi.  Loading
        # the config already rejects non-finite/negative gains; do not pin a
        # water-tuned value to an obsolete fixture number here.
        for axis in ("roll", "pitch"):
            for suffix in ("kp", "ki", "kd"):
                self.assertGreaterEqual(
                    config.getfloat("pid", f"{axis}_{suffix}"), 0.0
                )
        self.assertGreaterEqual(
            config.getfloat("control", "pitch_heave_feedforward_scale"), 0.0
        )
        self.assertLessEqual(
            config.getfloat("control", "pitch_heave_feedforward_scale"), 1.0
        )


if __name__ == "__main__":
    unittest.main()
