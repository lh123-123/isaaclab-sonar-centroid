import shutil
import math
import sys
import tempfile
import threading
import time
import types
import unittest
from collections import deque
from pathlib import Path
from unittest import mock


sys.modules.setdefault("serial", mock.Mock())
joystick_handler = types.ModuleType("joystick_handler")
joystick_handler.JoystickHandler = object
joystick_handler.NEUTRAL_JOYSTICK = {
    "tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0,
    "M_x": 0.0, "M_y": 0.0, "M_z": 0.0,
}
sys.modules.setdefault("joystick_handler", joystick_handler)
thrust_allocator = types.ModuleType("thrust_allocator")
thrust_allocator.THRUSTER_NAMES = tuple(f"thruster_{index}" for index in range(8))
thrust_allocator.ThrustAllocator = object
sys.modules.setdefault("thrust_allocator", thrust_allocator)
save = types.ModuleType("save")
save.is_saving = lambda: False
save.close_session = lambda: None
sys.modules.setdefault("save", save)
sys.path.insert(0, str(Path(__file__).resolve().parent / "pi_runtime"))

from attitude_math import (  # noqa: E402
    IDENTITY_QUATERNION,
    average_quaternions,
    attitude_from_raw_euler,
    from_euler_deg,
    inverse,
    mount_from_axis_captures,
    mount_from_pitch_yaw_captures,
    multiply,
    quaternion_distance_deg,
    raw_euler_to_vehicle_axes,
    rotate_vector,
    to_euler_deg,
    vehicle_body_rate_from_raw,
)
from config_loader import load_config  # noqa: E402
from main import ROVController  # noqa: E402


class Attitude3DMountMathTests(unittest.TestCase):
    def assertAnglesNear(self, actual, expected):
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name], value, places=6, msg=name)

    def test_mount_rotation_recovers_vehicle_axes_from_a_tilted_imu(self):
        mount = from_euler_deg(22.0, -17.0, 31.0)
        vehicle = from_euler_deg(12.0, -25.0, 48.0)
        raw = multiply(vehicle, inverse(mount))
        corrected, recovered = attitude_from_raw_euler(
            to_euler_deg(raw), mount, IDENTITY_QUATERNION
        )
        self.assertAnglesNear(corrected, to_euler_deg(vehicle))
        self.assertAnglesNear(to_euler_deg(recovered), to_euler_deg(vehicle))

    def test_options_reference_zeroes_all_axes_without_euler_offsets(self):
        mount = from_euler_deg(18.0, -11.0, 9.0)
        current = from_euler_deg(-31.0, 24.0, 97.0)
        raw = multiply(current, inverse(mount))
        corrected, _ = attitude_from_raw_euler(to_euler_deg(raw), mount, current)
        self.assertAnglesNear(corrected, {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})

    def test_verified_sensor_axis_convention_is_applied_before_mount_rotation(self):
        raw = {"roll": 13.0, "pitch": -21.0, "yaw": 47.0}
        corrected, _ = attitude_from_raw_euler(
            raw, IDENTITY_QUATERNION, IDENTITY_QUATERNION, (1.0, -1.0, -1.0)
        )
        self.assertAnglesNear(
            corrected,
            to_euler_deg(raw_euler_to_vehicle_axes(raw, (1.0, -1.0, -1.0))),
        )

    def test_two_pure_axis_captures_recover_mount_without_assuming_heading(self):
        mount = from_euler_deg(22.0, -17.0, 31.0)
        baseline_vehicle = from_euler_deg(7.0, -4.0, 123.0)
        raw_baseline = multiply(baseline_vehicle, inverse(mount))
        raw_roll = multiply(
            multiply(baseline_vehicle, from_euler_deg(32.0, 0.0, 0.0)),
            inverse(mount),
        )
        raw_pitch = multiply(
            multiply(baseline_vehicle, from_euler_deg(0.0, 27.0, 0.0)),
            inverse(mount),
        )
        recovered_mount = mount_from_axis_captures(
            multiply(inverse(raw_baseline), raw_roll),
            multiply(inverse(raw_baseline), raw_pitch),
        )
        reference = multiply(raw_baseline, recovered_mount)
        relative = multiply(
            inverse(reference), multiply(raw_pitch, recovered_mount)
        )
        self.assertAnglesNear(to_euler_deg(relative), {"roll": 0.0, "pitch": 27.0, "yaw": 0.0})

    def test_pitch_and_level_yaw_captures_recover_an_arbitrary_diagonal_mount(self):
        mount = from_euler_deg(37.0, -23.0, 58.0)
        baseline_vehicle = from_euler_deg(0.0, 0.0, 137.0)
        raw_baseline = multiply(baseline_vehicle, inverse(mount))
        raw_pitch = multiply(
            multiply(baseline_vehicle, from_euler_deg(0.0, 31.0, 0.0)),
            inverse(mount),
        )
        raw_yaw = multiply(
            multiply(baseline_vehicle, from_euler_deg(0.0, 0.0, 28.0)),
            inverse(mount),
        )

        recovered = mount_from_pitch_yaw_captures(
            multiply(inverse(raw_baseline), raw_pitch),
            multiply(inverse(raw_baseline), raw_yaw),
        )
        self.assertLess(quaternion_distance_deg(recovered, mount), 1e-6)

        x_axis = rotate_vector(recovered, (1.0, 0.0, 0.0))
        y_axis = rotate_vector(recovered, (0.0, 1.0, 0.0))
        z_axis = rotate_vector(recovered, (0.0, 0.0, 1.0))
        cross_xy = (
            x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
            x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
            x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0],
        )
        for actual, expected in zip(cross_xy, z_axis):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_pitch_yaw_calibration_averages_noisy_samples(self):
        mount = from_euler_deg(-29.0, 16.0, 73.0)
        baseline_vehicle = from_euler_deg(0.0, 0.0, -82.0)

        def noisy_average(vehicle_pose):
            ideal = multiply(vehicle_pose, inverse(mount))
            samples = []
            for roll_noise, pitch_noise, yaw_noise in (
                (-0.35, 0.20, -0.15),
                (0.25, -0.10, 0.30),
                (-0.10, -0.20, 0.05),
                (0.30, 0.15, -0.25),
                (-0.15, 0.05, 0.10),
            ):
                samples.append(multiply(
                    from_euler_deg(roll_noise, pitch_noise, yaw_noise), ideal
                ))
            return average_quaternions(samples)

        raw_baseline = noisy_average(baseline_vehicle)
        raw_pitch = noisy_average(
            multiply(baseline_vehicle, from_euler_deg(0.0, 30.0, 0.0))
        )
        raw_yaw = noisy_average(
            multiply(baseline_vehicle, from_euler_deg(0.0, 0.0, 26.0))
        )
        recovered = mount_from_pitch_yaw_captures(
            multiply(inverse(raw_baseline), raw_pitch),
            multiply(inverse(raw_baseline), raw_yaw),
        )
        self.assertLess(quaternion_distance_deg(recovered, mount), 0.5)

    def test_pitch_yaw_capture_enforces_20_to_45_degrees_and_axis_separation(self):
        with self.assertRaisesRegex(ValueError, "captured_rotation_too_small"):
            mount_from_pitch_yaw_captures(
                from_euler_deg(0.0, 19.9, 0.0),
                from_euler_deg(0.0, 0.0, 30.0),
            )
        with self.assertRaisesRegex(ValueError, "captured_rotation_too_large"):
            mount_from_pitch_yaw_captures(
                from_euler_deg(0.0, 30.0, 0.0),
                from_euler_deg(0.0, 0.0, 45.1),
            )
        angle = math.radians(30.0)
        half = angle / 2.0
        axis_20_degrees_from_pitch = (
            0.0,
            math.cos(math.radians(20.0)),
            math.sin(math.radians(20.0)),
        )
        bad_yaw_delta = (
            math.cos(half),
            *(component * math.sin(half) for component in axis_20_degrees_from_pitch),
        )
        with self.assertRaisesRegex(ValueError, "captured_axes_too_close"):
            mount_from_pitch_yaw_captures(
                from_euler_deg(0.0, 30.0, 0.0), bad_yaw_delta
            )

    def test_gyro_vector_is_rotated_from_imu_axes_into_vehicle_body_axes(self):
        # A +90 degree body->IMU yaw mount expresses body +X as IMU +Y.
        mount = from_euler_deg(0.0, 0.0, 90.0)
        body_rate = vehicle_body_rate_from_raw(
            (0.0, 12.0, 0.0),
            mount,
            (1.0, 1.0, 1.0),
            mount_calibrated=True,
        )
        self.assertAlmostEqual(body_rate[0], 12.0, places=6)
        self.assertAlmostEqual(body_rate[1], 0.0, places=6)
        self.assertAlmostEqual(body_rate[2], 0.0, places=6)

    def test_gyro_legacy_signs_are_applied_without_a_mount_guess(self):
        body_rate = vehicle_body_rate_from_raw(
            (3.0, 4.0, 5.0),
            IDENTITY_QUATERNION,
            (1.0, -1.0, -1.0),
            mount_calibrated=False,
        )
        self.assertEqual(body_rate, (3.0, -4.0, -5.0))


def controller_for_mount(config_path):
    controller = ROVController.__new__(ROVController)
    controller.state_lock = threading.RLock()
    controller.thrusters_armed = False
    controller.control_enabled = False
    controller.attitude_hold = {"roll": False, "pitch": False}
    controller.imu_freshness = 0.35
    controller._last_imu_update = time.monotonic()
    controller._latest_raw_euler = {"roll": 18.0, "pitch": -11.0, "yaw": 37.0}
    controller.imu_3d_mount_calibrated = False
    controller.imu_mount_quaternion = IDENTITY_QUATERNION
    controller.imu_reference_quaternion = IDENTITY_QUATERNION
    controller.imu_roll_inverted = False
    controller.imu_pitch_inverted = True
    controller.imu_yaw_inverted = True
    controller.imu_roll_offset_deg = 0.0
    controller.imu_pitch_offset_deg = 0.0
    controller.imu_yaw_offset_deg = 0.0
    controller.sensor_state = {"roll": 18.0, "pitch": 11.0, "yaw": -37.0}
    controller.attitude_control_mapping_validated = True
    controller.controller = mock.Mock()
    controller.config_path = str(config_path)
    controller.config = load_config(config_path)
    controller.last_command_ack = {}
    return controller


class Attitude3DMountPersistenceTests(unittest.TestCase):
    def test_runtime_capture_averages_a_stable_window_and_reports_progress(self):
        source = Path(__file__).resolve().parent / "config.ini"
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            shutil.copyfile(source, config_path)
            controller = controller_for_mount(config_path)
            now = time.monotonic()
            controller._last_imu_update = now
            baseline = controller._raw_imu_quaternion(
                controller._latest_raw_euler
            )
            controller._recent_raw_quaternions = deque(maxlen=20)
            for index, noise in enumerate((-0.20, 0.15, -0.10, 0.05, 0.10)):
                controller._recent_raw_quaternions.append((
                    now - index * 0.02,
                    multiply(from_euler_deg(noise, 0.0, 0.0), baseline),
                ))

            with mock.patch("main.time.monotonic", return_value=now):
                controller._calibrate_attitude_mount()

            self.assertFalse(controller.attitude_control_mapping_validated)
            self.assertFalse(load_config(config_path).getboolean(
                "control", "attitude_control_mapping_validated"
            ))
            self.assertEqual(controller.last_command_ack["sample_count"], 5)
            self.assertLess(controller.last_command_ack["sample_spread_deg"], 1.5)
            progress = controller._attitude_mount_calibration_snapshot()
            self.assertTrue(progress["active"])
            self.assertTrue(progress["baseline_captured"])
            self.assertFalse(progress["pitch_captured"])
            self.assertFalse(progress["yaw_captured"])
            self.assertEqual(progress["required_angle_min_deg"], 20.0)
            self.assertEqual(progress["required_angle_max_deg"], 45.0)

    def test_runtime_capture_rejects_a_pose_that_is_still_moving(self):
        source = Path(__file__).resolve().parent / "config.ini"
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            shutil.copyfile(source, config_path)
            controller = controller_for_mount(config_path)
            now = time.monotonic()
            controller._last_imu_update = now
            baseline = controller._raw_imu_quaternion(
                controller._latest_raw_euler
            )
            controller._recent_raw_quaternions = deque(maxlen=20)
            for index, motion in enumerate((-4.0, -2.0, 0.0, 2.0, 4.0)):
                controller._recent_raw_quaternions.append((
                    now - index * 0.02,
                    multiply(from_euler_deg(motion, 0.0, 0.0), baseline),
                ))

            with mock.patch("main.time.monotonic", return_value=now):
                with self.assertRaisesRegex(ValueError, "pose_not_stable"):
                    controller._calibrate_attitude_mount()

    def test_mount_calibration_persists_rotation_and_zeroes_live_attitude(self):
        source = Path(__file__).resolve().parent / "config.ini"
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            shutil.copyfile(source, config_path)
            controller = controller_for_mount(config_path)
            controller._calibrate_attitude_mount()
            baseline = raw_euler_to_vehicle_axes(
                controller._latest_raw_euler, (1.0, -1.0, -1.0)
            )
            def raw_from_vehicle_quaternion(quaternion):
                value = to_euler_deg(quaternion)
                return {"roll": value["roll"], "pitch": -value["pitch"], "yaw": -value["yaw"]}
            controller._latest_raw_euler = raw_from_vehicle_quaternion(
                multiply(baseline, from_euler_deg(0.0, 24.0, 0.0))
            )
            controller._capture_attitude_mount_axis("pitch")
            controller._latest_raw_euler = raw_from_vehicle_quaternion(
                multiply(baseline, from_euler_deg(0.0, 0.0, 28.0))
            )
            controller._capture_attitude_mount_axis("yaw")
            controller._latest_raw_euler = raw_from_vehicle_quaternion(baseline)
            controller._finalize_attitude_mount()
            saved = load_config(config_path)
            self.assertTrue(saved.getboolean("hardware", "imu_3d_mount_calibrated"))
            self.assertTrue(controller.last_command_ack["ok"])
            self.assertEqual(
                controller.last_command_ack["message"],
                "attitude_mount_calibrated_pitch_yaw",
            )
            for value in controller.sensor_state.values():
                self.assertAlmostEqual(value, 0.0, places=6)
            controller._calibrate_attitude_zero()
            self.assertEqual(controller.last_command_ack["message"], "attitude_zero_calibrated_3d")

    def test_mount_calibration_applies_the_verified_axis_convention_first(self):
        source = Path(__file__).resolve().parent / "config.ini"
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            shutil.copyfile(source, config_path)
            controller = controller_for_mount(config_path)
            controller._calibrate_attitude_mount()
            expected = raw_euler_to_vehicle_axes(
                controller._latest_raw_euler, (1.0, -1.0, -1.0)
            )
            for actual, value in zip(
                controller._attitude_mount_capture["baseline"], expected
            ):
                self.assertAlmostEqual(actual, value, places=9)


if __name__ == "__main__":
    unittest.main()
