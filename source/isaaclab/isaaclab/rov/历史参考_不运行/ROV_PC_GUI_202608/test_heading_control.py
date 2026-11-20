import sys
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent / "pi_runtime"))

from Heading_Control import HeadingControl  # noqa: E402
from pc_gui import ROVGui  # noqa: E402


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class HeadingControlTests(unittest.TestCase):
    def test_yaw_moment_has_same_sign_as_shortest_heading_error(self):
        controller = HeadingControl(Kp=0.5, Ki=0.0, Kd=0.0, sample_time=0.0)
        controller.set_course(10.0)
        self.assertEqual(controller.update(0.0), 5.0)

        controller.set_course(350.0)
        self.assertEqual(controller.update(10.0), -5.0)

    def test_output_is_bounded_for_large_error(self):
        controller = HeadingControl(Kp=0.5, sample_time=0.0, output_limit=5.0)
        controller.set_course(180.0)
        self.assertEqual(controller.update(0.0), -5.0)

    def test_pid_update_clears_stale_output(self):
        controller = HeadingControl(Kp=0.5, sample_time=0.0)
        controller.set_course(20.0)
        self.assertEqual(controller.update(0.0), 5.0)
        controller.set_pid((0.5, 0.0, 0.0))
        self.assertEqual(controller.output, 0.0)

    def test_boundary_delta_is_the_short_two_degree_rotation(self):
        # This is the exact circular delta used by HeadingControl's derivative
        # term when feedback crosses -180/180.
        self.assertEqual(HeadingControl.shortest_error(-179.0, 179.0), 2.0)

    def test_course_is_exposed_in_signed_angle_range(self):
        controller = HeadingControl()
        controller.set_course(270.0)
        self.assertEqual(controller.exp_course, -90.0)


class HeadingCoordinateTests(unittest.TestCase):
    def make_gui(self, *, inverted=True, offset=0.0):
        gui = ROVGui.__new__(ROVGui)
        gui._imu_roll_inverted = False
        gui._imu_pitch_inverted = True
        gui._imu_yaw_inverted = True
        gui._imu_heading_inverted = inverted
        gui._imu_yaw_offset_deg = 0.0
        gui._imu_heading_offset_deg = offset
        return gui

    def test_vehicle_heading_round_trips_to_pi_raw_heading(self):
        gui = self.make_gui(inverted=True, offset=0.0)
        vehicle = gui._vehicle_imu({"heading": 7.5})["heading"]
        self.assertAlmostEqual(vehicle, 352.5)
        self.assertAlmostEqual(gui._raw_heading_from_vehicle(vehicle), 7.5)

    def test_heading_offset_round_trip(self):
        gui = self.make_gui(inverted=True, offset=15.0)
        vehicle = gui._vehicle_imu({"heading": 40.0})["heading"]
        self.assertAlmostEqual(gui._raw_heading_from_vehicle(vehicle), 40.0)

    def test_one_click_depth_does_not_enable_unvalidated_heading_loop(self):
        gui = self.make_gui(inverted=True, offset=0.0)
        gui._emergency_stop_latched = False
        gui._confirmed_armed_telemetry = lambda: {
            "depth": 0.3,
            "sensor_health": {"depth_valid": True},
            "imu": {"heading": 7.5},
        }
        gui._tgt_d_var = Var("0")
        gui._depth_en, gui._speed_en = Var(False), Var(False)
        gui._log_event = lambda _message: None
        gui._cmd_start = mock.Mock()

        gui._cmd_hover()

        self.assertEqual(gui._tgt_d_var.get(), "0.300")
        self.assertTrue(gui._depth_en.get())
        gui._cmd_start.assert_called_once_with()

    def test_gui_autonomous_command_never_enables_heading_hold(self):
        gui = self.make_gui(inverted=True, offset=0.0)
        gui._emergency_stop_latched = False
        gui._estop_pending_target = None
        gui._confirmed_armed_telemetry = lambda: {
            "depth": 0.3,
            "sensor_health": {"depth_valid": True},
            "control_link_ok": True,
            "armed": True,
            "rearm_required": False,
            "emergency_stop": False,
        }
        gui._depth_en, gui._speed_en = Var(True), Var(False)
        gui._tgt_d_var, gui._tgt_s_var = Var("0.3"), Var("0")
        gui._d_kp, gui._d_ki, gui._d_kd = Var("60"), Var("0"), Var("0")
        gui._save_var = Var(False)
        gui._depth_target_min_m = -0.2
        gui._depth_target_max_m = 10.0
        gui._depth_target_max_step_m = 0.75
        gui._log_event = lambda _message: None
        gui.client = mock.Mock()
        gui.client.get_telemetry_marker.return_value = ("pi", 1)
        gui._wait_for_pid_ack = mock.Mock()
        gui._wait_for_autonomous_enabled = mock.Mock()

        with mock.patch("pc_gui.messagebox.showwarning") as warning:
            gui._cmd_start()

        warning.assert_not_called()
        command_calls = [
            call for call in gui.client.send_cmd.call_args_list
            if "mode" in call.kwargs
        ]
        self.assertTrue(command_calls)
        self.assertTrue(all(
            call.kwargs["mode"]["heading_hold"] is False
            and call.kwargs["target"]["heading_deg"] == 0.0
            for call in command_calls
        ))


if __name__ == "__main__":
    unittest.main()
