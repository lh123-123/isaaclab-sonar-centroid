import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
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

from config_loader import load_config  # noqa: E402
from main import ROVController  # noqa: E402
from secure_protocol import ProtocolError, validate_control_payload  # noqa: E402
from attitude_math import IDENTITY_QUATERNION  # noqa: E402


def controller_for_attitude_zero(
    config_path, *, armed=False, fresh=True, mount_calibrated=True,
    hold=None, control_enabled=False,
):
    controller = ROVController.__new__(ROVController)
    controller.state_lock = threading.RLock()
    controller.thrusters_armed = armed
    controller.control_enabled = control_enabled
    controller.attitude_hold = hold or {"roll": False, "pitch": False}
    controller.imu_freshness = 0.35
    controller._last_imu_update = time.monotonic() if fresh else 0.0
    controller._latest_raw_euler = {"roll": 8.0, "pitch": -5.0, "yaw": 175.0}
    controller.imu_roll_inverted = False
    controller.imu_pitch_inverted = True
    controller.imu_yaw_inverted = True
    controller.imu_roll_offset_deg = 0.0
    controller.imu_pitch_offset_deg = 0.0
    controller.imu_yaw_offset_deg = 0.0
    controller.imu_3d_mount_calibrated = mount_calibrated
    controller.imu_mount_quaternion = IDENTITY_QUATERNION
    controller.imu_reference_quaternion = IDENTITY_QUATERNION
    controller.sensor_state = {"roll": 8.0, "pitch": 5.0, "yaw": -175.0}
    controller.config_path = str(config_path)
    controller.config = load_config(config_path)
    controller.last_command_ack = {}
    return controller


class AttitudeZeroCalibrationTests(unittest.TestCase):
    def test_legacy_roll_and_yaw_are_both_bounded_to_signed_half_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            shutil.copyfile(Path(__file__).resolve().parent / "config.ini", config_path)
            controller = controller_for_attitude_zero(
                config_path, mount_calibrated=False
            )
            attitude = controller._referenced_attitude(
                {"roll": 541.0, "pitch": 0.0, "yaw": -541.0}
            )
            self.assertEqual(attitude["roll"], -179.0)
            self.assertEqual(attitude["yaw"], -179.0)

    def test_zero_persists_the_vehicle_reference_and_updates_live_telemetry(self):
        source_config = Path(__file__).resolve().parent / "config.ini"
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            shutil.copyfile(source_config, config_path)
            controller = controller_for_attitude_zero(config_path)

            controller._calibrate_attitude_zero()

            persisted = load_config(config_path)
            saved_reference = tuple(
                persisted.getfloat("hardware", f"imu_reference_quat_{name}")
                for name in ("w", "x", "y", "z")
            )
            self.assertNotEqual(saved_reference, IDENTITY_QUATERNION)
            self.assertEqual(controller.sensor_state, {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})
            self.assertTrue(controller.last_command_ack["ok"])
            self.assertTrue(controller.last_command_ack["persisted"])
            self.assertFalse(controller.attitude_control_mapping_validated)
            self.assertFalse(persisted.getboolean(
                "control", "attitude_control_mapping_validated"
            ))

    def test_zero_rejects_unsafe_state_unmounted_sensor_or_stale_imu(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            shutil.copyfile(Path(__file__).resolve().parent / "config.ini", config_path)
            with self.assertRaisesRegex(ValueError, "locked"):
                controller_for_attitude_zero(config_path, armed=True)._calibrate_attitude_zero()
            with self.assertRaisesRegex(ValueError, "not_fresh"):
                controller_for_attitude_zero(config_path, fresh=False)._calibrate_attitude_zero()
            with self.assertRaisesRegex(ValueError, "mount_uncalibrated"):
                controller_for_attitude_zero(
                    config_path, mount_calibrated=False
                )._calibrate_attitude_zero()
            with self.assertRaisesRegex(ValueError, "holds_must_be_off"):
                controller_for_attitude_zero(
                    config_path, hold={"roll": True, "pitch": False}
                )._calibrate_attitude_zero()

    def test_protocol_accepts_only_a_boolean_standalone_flag(self):
        self.assertEqual(
            validate_control_payload({"attitude_zero_calibrate": True}),
            {"attitude_zero_calibrate": True},
        )
        with self.assertRaisesRegex(ProtocolError, "true or false"):
            validate_control_payload({"attitude_zero_calibrate": 1})
        self.assertEqual(
            validate_control_payload({"attitude_mount_calibrate": True}),
            {"attitude_mount_calibrate": True},
        )
        for field in (
            "attitude_mount_capture_roll",
            "attitude_mount_capture_pitch",
            "attitude_mount_capture_yaw",
            "attitude_mount_finalize",
        ):
            self.assertEqual(validate_control_payload({field: True}), {field: True})

    def test_protocol_accepts_only_boolean_mapping_validation(self):
        self.assertEqual(
            validate_control_payload({"attitude_mapping_validated": True}),
            {"attitude_mapping_validated": True},
        )
        for invalid in (1, 0, "true", None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ProtocolError, "true or false"):
                    validate_control_payload({
                        "attitude_mapping_validated": invalid,
                    })


if __name__ == "__main__":
    unittest.main()
