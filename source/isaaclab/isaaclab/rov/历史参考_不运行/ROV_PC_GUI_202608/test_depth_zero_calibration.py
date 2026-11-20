import math
import sys
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

from main import ROVController
from secure_protocol import ProtocolError, validate_control_payload


class FakeDepthSensor:
    def __init__(self, pressure=0.0042, depth=0.0918, temperature=25.0):
        self.reference = None
        self.pressure = pressure
        self.depth = depth
        self.temperature = temperature

    def set_surface_reference(self, pressure_mpa):
        self.reference = pressure_mpa

    def get_pressure(self):
        return self.pressure

    def get_depth(self):
        return self.depth

    def get_temperature(self):
        return self.temperature


def controller_for_zero(*, armed=False, enabled=False, fresh=True):
    controller = ROVController.__new__(ROVController)
    controller.state_lock = threading.RLock()
    controller.thrusters_armed = armed
    controller.control_enabled = enabled
    controller.depth_freshness = 1.25
    controller._last_depth_update = time.monotonic() if fresh else 0.0
    controller._last_depth_pressure_update = time.monotonic() if fresh else 0.0
    controller._depth_fault_reason = ""
    controller.sensor_state = {"pressure_mpa": 0.0042, "depth": 0.0918}
    controller.depth_sensor = FakeDepthSensor()
    controller.depth_raw_zero_mpa = 0.0033
    controller._prev_depth = 0.0918
    controller._prev_depth_time = time.monotonic() - 0.5
    controller._vert_speed = 0.2
    controller.last_command_ack = {}
    return controller


class DepthZeroCalibrationTests(unittest.TestCase):
    def test_calibration_updates_only_runtime_reference_and_resets_depth_state(self):
        controller = controller_for_zero()

        controller._calibrate_depth_zero()

        self.assertEqual(controller.depth_sensor.reference, 0.0042)
        self.assertEqual(controller.depth_raw_zero_mpa, 0.0042)
        self.assertEqual(controller.sensor_state["depth"], 0.0)
        self.assertEqual(controller._vert_speed, 0.0)
        self.assertTrue(controller.last_command_ack["ok"])
        self.assertEqual(controller.last_command_ack["message"], "depth_zero_calibrated")
        self.assertFalse(controller.last_command_ack["persisted"])

    def test_calibration_requires_locked_thrusters_and_fresh_pressure(self):
        with self.assertRaisesRegex(ValueError, "locked"):
            controller_for_zero(armed=True)._calibrate_depth_zero()
        with self.assertRaisesRegex(ValueError, "not_fresh"):
            controller_for_zero(fresh=False)._calibrate_depth_zero()

    def test_calibration_accepts_fresh_pressure_when_old_depth_is_out_of_range(self):
        controller = controller_for_zero()
        controller.sensor_state.update({"pressure_mpa": 0.2500, "depth": 25.0})
        controller._last_depth_update = 0.0
        controller._depth_fault_reason = "depth_out_of_range"

        controller._calibrate_depth_zero()

        self.assertEqual(controller.depth_sensor.reference, 0.2500)
        self.assertEqual(controller.sensor_state["depth"], 0.0)
        self.assertGreater(controller._last_depth_update, 0.0)
        self.assertEqual(controller._depth_fault_reason, "")

    def test_out_of_range_depth_still_marks_raw_pressure_fresh_for_zeroing(self):
        controller = controller_for_zero()
        controller.depth_sensor = FakeDepthSensor(pressure=0.2500, depth=25.0)
        controller._last_depth_pressure_update = 0.0
        controller._last_depth_update = 0.0
        controller._depth_rejected_samples = 0
        controller._update_depth_state(time.monotonic())

        self.assertGreater(controller._last_depth_pressure_update, 0.0)
        self.assertEqual(controller._last_depth_update, 0.0)
        self.assertEqual(controller._depth_fault_reason, "depth_out_of_range")
        controller._calibrate_depth_zero()
        self.assertEqual(controller.sensor_state["depth"], 0.0)

    def test_protocol_accepts_only_boolean_calibration_request(self):
        self.assertEqual(
            validate_control_payload({"depth_zero_calibrate": True}),
            {"depth_zero_calibrate": True},
        )
        with self.assertRaisesRegex(ProtocolError, "true or false"):
            validate_control_payload({"depth_zero_calibrate": 1})


if __name__ == "__main__":
    unittest.main()
