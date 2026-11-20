import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pi_runtime"))

from controller import Controller


class DepthHoldFilterDeadbandTests(unittest.TestCase):
    def setUp(self):
        self.controller = Controller(config_path=ROOT / "config.ini")
        self.mode = {"heading_hold": False, "depth_hold": True, "speed_hold": False}

    def test_deadband_returns_zero_correction_without_a_fixed_midpoint(self):
        auto, inhibited = self.controller.compute_control(
            {"depth": 0.100}, self.mode, {"depth_m": 0.110}, {"depth": True}
        )
        self.assertEqual(auto[2], 0.0)
        self.assertEqual(inhibited, [])

    def test_target_depth_with_downward_speed_commands_immediate_braking(self):
        self.controller.depth_controller.sample_time = 0.0
        auto, inhibited = self.controller.compute_control(
            {"depth": 0.520, "vertical_speed": 0.4},
            self.mode,
            {"depth_m": 0.520},
            {"depth": True},
        )
        self.assertLess(auto[2], 0.0)
        self.assertEqual(inhibited, [])

    def test_deadband_preserves_learned_vertical_bias_when_settled(self):
        self.controller.depth_controller.sample_time = 0.0
        self.controller.depth_controller.integral = -3.0
        auto, _ = self.controller.compute_control(
            {"depth": 0.520, "vertical_speed": 0.0},
            self.mode,
            {"depth_m": 0.520},
            {"depth": True},
        )
        self.assertEqual(auto[2], -3.0)
        self.assertEqual(self.controller.depth_controller.integral, -3.0)

    def test_error_outside_deadband_produces_a_vertical_correction(self):
        self.controller.compute_control(
            {"depth": 0.100}, self.mode, {"depth_m": 0.100}, {"depth": True}
        )
        self.controller.depth_controller.last_time -= 1.0
        auto, _ = self.controller.compute_control(
            {"depth": 0.100}, self.mode, {"depth_m": 0.140}, {"depth": True}
        )
        self.assertGreater(auto[2], 0.0)

    def test_fast_profile_is_bounded_and_vertical_speed_damps_output(self):
        self.assertEqual(self.controller.depth_pid_params, (55.0, 4.0, 50.0))
        self.assertEqual(self.controller.depth_controller.output_limit, 60.0)
        self.assertEqual(self.controller.depth_hold_filter_alpha, 0.55)
        self.assertEqual(
            self.controller.depth_release_brake_exit_speed_mps, 0.03
        )

        still = Controller(config_path=ROOT / "config.ini")
        descending = Controller(config_path=ROOT / "config.ini")
        still.depth_controller.sample_time = 0.0
        descending.depth_controller.sample_time = 0.0
        still_output, _ = still.compute_control(
            {"depth": 0.0, "vertical_speed": 0.0},
            self.mode, {"depth_m": 1.0}, {"depth": True},
        )
        descending_output, _ = descending.compute_control(
            {"depth": 0.0, "vertical_speed": 0.5},
            self.mode, {"depth_m": 1.0}, {"depth": True},
        )
        self.assertGreater(still_output[2], descending_output[2])
        self.assertLessEqual(abs(still_output[2]), 60.0)

    def test_depth_integrator_does_not_wind_up_behind_force_limit(self):
        self.controller.depth_controller.sample_time = 0.0
        for _ in range(20):
            output, _ = self.controller.compute_control(
                {"depth": 0.0, "vertical_speed": 0.0},
                self.mode, {"depth_m": 10.0}, {"depth": True},
            )
        self.assertEqual(output[2], 60.0)
        self.assertEqual(self.controller.depth_controller.integral, 0.0)

    def test_manual_heave_freezes_integral_and_release_brakes_at_new_depth(self):
        self.controller.depth_controller.sample_time = 0.0
        self.controller.depth_controller.integral = -5.0

        for _ in range(20):
            self.controller.compute_control(
                {"depth": 1.2, "vertical_speed": -0.4},
                self.mode,
                {"depth_m": 0.5},
                {"depth": True},
                depth_manual_override=True,
            )

        self.assertEqual(self.controller.depth_controller.integral, -5.0)
        released, inhibited = self.controller.compute_control(
            {"depth": 1.2, "vertical_speed": -0.4},
            self.mode,
            {"depth_m": 1.2},
            {"depth": True},
        )
        self.assertGreater(released[2], 0.0)
        self.assertEqual(inhibited, [])

    def test_release_does_not_apply_integral_that_reinforces_upward_inertia(self):
        self.controller.depth_controller.sample_time = 0.0
        self.controller.depth_controller.integral = -10.0

        self.controller.compute_control(
            {"depth": 1.2, "vertical_speed": -0.25},
            self.mode,
            {"depth_m": 1.2},
            {"depth": True},
            depth_manual_override=True,
        )
        released, _ = self.controller.compute_control(
            {"depth": 1.2, "vertical_speed": -0.25},
            self.mode,
            {"depth_m": 1.2},
            {"depth": True},
        )

        # Kd=50 should contribute the full +12.5 N downward brake.  The learned
        # -10 N static bias remains stored, but must not cancel that brake while
        # the vehicle is still coasting upward.
        self.assertAlmostEqual(released[2], 12.5)
        self.assertEqual(self.controller.depth_controller.integral, -10.0)

    def test_release_brake_is_ready_before_the_next_pid_sample(self):
        self.controller.depth_controller.integral = -10.0

        self.controller.compute_control(
            {"depth": 1.2, "vertical_speed": -0.25},
            self.mode,
            {"depth_m": 1.2},
            {"depth": True},
            depth_manual_override=True,
        )
        # Do not advance last_time.  This reproduces a stick release occurring
        # between the depth PID's 0.1 s samples.
        released, _ = self.controller.compute_control(
            {"depth": 1.2, "vertical_speed": -0.25},
            self.mode,
            {"depth_m": 1.2},
            {"depth": True},
        )

        self.assertAlmostEqual(released[2], 12.5)
        self.assertEqual(self.controller.depth_controller.integral, -10.0)

    def test_release_keeps_integral_when_it_already_opposes_downward_inertia(self):
        self.controller.depth_controller.sample_time = 0.0
        self.controller.depth_controller.integral = -10.0

        self.controller.compute_control(
            {"depth": 1.2, "vertical_speed": 0.2},
            self.mode,
            {"depth_m": 1.2},
            {"depth": True},
            depth_manual_override=True,
        )
        released, _ = self.controller.compute_control(
            {"depth": 1.2, "vertical_speed": 0.2},
            self.mode,
            {"depth_m": 1.2},
            {"depth": True},
        )

        self.assertAlmostEqual(released[2], -20.0)
        self.assertEqual(self.controller.depth_controller.integral, -10.0)

    def test_static_integral_returns_after_release_speed_has_settled(self):
        self.controller.depth_controller.sample_time = 0.0
        self.controller.depth_controller.integral = -10.0

        self.controller.compute_control(
            {"depth": 1.2, "vertical_speed": -0.25},
            self.mode,
            {"depth_m": 1.2},
            {"depth": True},
            depth_manual_override=True,
        )
        self.controller.compute_control(
            {"depth": 1.2, "vertical_speed": -0.25},
            self.mode,
            {"depth_m": 1.2},
            {"depth": True},
        )
        settled, _ = self.controller.compute_control(
            {"depth": 1.2, "vertical_speed": 0.0},
            self.mode,
            {"depth_m": 1.2},
            {"depth": True},
        )

        self.assertAlmostEqual(settled[2], -10.0)
        self.assertEqual(self.controller.depth_controller.integral, -10.0)


if __name__ == "__main__":
    unittest.main()
