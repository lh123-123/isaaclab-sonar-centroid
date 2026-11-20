import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pi_runtime"))

from controller import Controller  # noqa: E402
from secure_protocol import validate_control_payload  # noqa: E402
from test_vertical_force_sign import configured_allocator  # noqa: E402
from yaw_feedback import YawFeedbackEstimator  # noqa: E402


def clean_mode():
    return {"heading_hold": False, "depth_hold": False, "speed_hold": False}


def clean_target():
    return {"heading_deg": 0.0, "depth_m": 0.0, "speed_mps": 0.0}


def current_yaw(yaw, gyro_z=0.0):
    return {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": float(yaw),
        "gyro_x": 0.0,
        "gyro_y": 0.0,
        "gyro_z": float(gyro_z),
    }


class YawFeedbackEstimatorTests(unittest.TestCase):
    def test_integrates_body_rate_when_absolute_yaw_is_stuck(self):
        estimator = YawFeedbackEstimator(
            stale_seconds=0.3,
            rate_threshold_dps=0.5,
            reanchor_deg=0.5,
        )

        estimator.update(95.0, 0.0, 100.0)
        estimator.update(95.0, 1.0, 100.1)
        estimator.update(95.0, 1.0, 100.2)
        yaw, fallback = estimator.update(95.0, 1.0, 100.31)

        self.assertTrue(fallback)
        self.assertAlmostEqual(yaw, 95.11, places=6)

        yaw, fallback = estimator.update(95.0, 1.0, 100.41)
        self.assertTrue(fallback)
        self.assertAlmostEqual(yaw, 95.21, places=6)

    def test_reanchors_when_absolute_yaw_recovers(self):
        estimator = YawFeedbackEstimator(stale_seconds=0.2)
        estimator.update(95.0, 0.0, 100.0)
        estimator.update(95.0, 1.0, 100.1)
        estimator.update(95.0, 1.0, 100.2)
        estimator.update(95.0, 1.0, 100.3)

        yaw, fallback = estimator.update(96.0, 0.0, 100.4)

        self.assertFalse(fallback)
        self.assertEqual(yaw, 96.0)

    def test_stationary_euler_yaw_does_not_enter_fallback(self):
        estimator = YawFeedbackEstimator(stale_seconds=0.2)
        estimator.update(95.0, 0.0, 100.0)
        yaw, fallback = estimator.update(95.0, 0.2, 100.3)

        self.assertFalse(fallback)
        self.assertEqual(yaw, 95.0)

    def test_sustained_slow_turn_is_integrated_without_false_static_trigger(self):
        estimator = YawFeedbackEstimator(stale_seconds=0.2)
        estimator.update(95.0, 0.0, 100.0)
        estimator.update(95.0, 0.2, 100.1)
        yaw, fallback = estimator.update(95.0, 0.2, 100.3)
        self.assertTrue(fallback)
        self.assertAlmostEqual(yaw, 95.04, places=6)


class YawHoldControllerTests(unittest.TestCase):
    def make_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.ini"
            shutil.copyfile(ROOT / "config.ini", path)
            return Controller(config_path=path)

    def test_yaw_hold_uses_the_pi_captured_target(self):
        controller = self.make_controller()

        first, _ = controller.compute_control(
            current_yaw(35.0), clean_mode(), clean_target(),
            {"imu": True, "depth": False},
            attitude_hold={"roll": False, "pitch": False, "yaw": True},
            attitude_targets={"roll": 0.0, "pitch": 0.0, "yaw": 35.0},
        )
        controller.yaw_controller.sample_time = 0.0
        second, _ = controller.compute_control(
            current_yaw(45.0), clean_mode(), clean_target(),
            {"imu": True, "depth": False},
            attitude_hold={"roll": False, "pitch": False, "yaw": True},
            attitude_targets={"roll": 0.0, "pitch": 0.0, "yaw": 35.0},
        )

        self.assertEqual(first[3:5], (0.0, 0.0))
        self.assertEqual(first[5], 0.0)
        self.assertLess(second[5], 0.0)

    def test_captured_yaw_target_uses_the_short_wraparound_error(self):
        controller = self.make_controller()
        controller.yaw_controller.sample_time = 0.0
        controller.compute_control(
            current_yaw(-179.0), clean_mode(), clean_target(),
            {"imu": True, "depth": False},
            attitude_hold={"roll": False, "pitch": False, "yaw": True},
            attitude_targets={"roll": 0.0, "pitch": 0.0, "yaw": 179.0},
        )
        output, _ = controller.compute_control(
            current_yaw(-179.0), clean_mode(), clean_target(),
            {"imu": True, "depth": False},
            attitude_hold={"roll": False, "pitch": False, "yaw": True},
            attitude_targets={"roll": 0.0, "pitch": 0.0, "yaw": 179.0},
        )
        self.assertLess(output[5], 0.0)
        self.assertLess(abs(output[5]), 1.0)

    def test_positive_yaw_rate_is_damped_with_negative_moment(self):
        controller = self.make_controller()
        controller.yaw_controller.sample_time = 0.0
        controller.compute_control(
            current_yaw(0.0, gyro_z=8.0), clean_mode(), clean_target(),
            {"imu": True, "depth": False},
            attitude_hold={"roll": False, "pitch": False, "yaw": True},
        )
        output, _ = controller.compute_control(
            current_yaw(0.0, gyro_z=8.0), clean_mode(), clean_target(),
            {"imu": True, "depth": False},
            attitude_hold={"roll": False, "pitch": False, "yaw": True},
        )
        self.assertLess(output[5], 0.0)

    def test_yaw_pid_has_its_own_runtime_type_snapshot_and_limit(self):
        controller = self.make_controller()
        controller.set_pid_runtime(6, (0.25, 0.01, 0.09))
        snapshot = controller.get_pid_snapshot()

        self.assertEqual(
            tuple(snapshot["yaw"][key] for key in ("kp", "ki", "kd")),
            (0.25, 0.01, 0.09),
        )
        self.assertEqual(snapshot["yaw"]["max_moment_nm"], 10.0)


class YawHoldProtocolTests(unittest.TestCase):
    def test_protocol_accepts_three_axis_hold_and_yaw_pid_type(self):
        hold = {"roll": True, "pitch": False, "yaw": True}
        pid = {"type": 6, "kp": 0.2, "ki": 0.01, "kd": 0.08}
        self.assertEqual(
            validate_control_payload({"attitude_hold": hold}),
            {"attitude_hold": hold},
        )
        self.assertEqual(
            validate_control_payload({"pid_set": pid}),
            {"pid_set": pid},
        )


class YawHorizontalAllocationTests(unittest.TestCase):
    def test_yaw_and_translation_share_only_the_four_diagonal_thrusters(self):
        allocator = configured_allocator()
        result = allocator.allocate(
            tau_x=20.0, tau_y=8.0, M_z=2.0, write_output=False
        )

        self.assertTrue(np.allclose(result.nominal_thrusts[4:8], 0.0))
        self.assertAlmostEqual(result.achieved_tau[0], 20.0, places=6)
        self.assertAlmostEqual(result.achieved_tau[1], 8.0, places=6)
        self.assertAlmostEqual(result.achieved_tau[5], 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
