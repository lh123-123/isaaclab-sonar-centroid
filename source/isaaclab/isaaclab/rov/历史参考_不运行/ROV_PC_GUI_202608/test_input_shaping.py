import unittest

from input_shaping import apply_response_curve, remap_deadzone, select_cardinal_axis


class SelectCardinalAxisTests(unittest.TestCase):
    def test_up_down_sectors_keep_only_y(self):
        self.assertEqual(select_cardinal_axis(0.2, 0.8), (0.0, 0.8))
        self.assertEqual(select_cardinal_axis(-0.3, -0.9), (0.0, -0.9))

    def test_left_right_sectors_keep_only_x(self):
        self.assertEqual(select_cardinal_axis(0.8, 0.2), (0.8, 0.0))
        self.assertEqual(select_cardinal_axis(-0.9, -0.3), (-0.9, 0.0))

    def test_exact_diagonal_is_one_action(self):
        self.assertEqual(select_cardinal_axis(0.6, -0.6), (0.0, -0.6))

    def test_non_finite_input_fails_closed(self):
        self.assertEqual(select_cardinal_axis(float("nan"), 0.5), (0.0, 0.0))


class ResponseCurveTests(unittest.TestCase):
    def test_mid_travel_is_significantly_gentler(self):
        mid_travel = remap_deadzone(0.5)
        self.assertAlmostEqual(mid_travel, 0.4117647059)
        self.assertLess(apply_response_curve(mid_travel, 2.2), 0.15)

    def test_full_travel_and_direction_are_preserved(self):
        self.assertEqual(apply_response_curve(1.0, 2.2), 1.0)
        self.assertEqual(apply_response_curve(-1.0, 2.2), -1.0)

    def test_invalid_exponent_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_response_curve(0.5, 0.9)


if __name__ == "__main__":
    unittest.main()
