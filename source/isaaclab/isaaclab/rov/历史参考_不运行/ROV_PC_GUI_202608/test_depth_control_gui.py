import unittest
from unittest import mock

from pc_gui import (
    ROVGui,
    format_depth_display,
    format_pressure_status_text,
    validate_depth_request,
    valid_depth_measurement,
)


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def healthy_telemetry(depth=0.4):
    return {
        "depth": depth,
        "pressure_mpa": 0.0042,
        "sensor_health": {"depth_valid": True, "pressure_valid": True},
        "control_link_ok": True,
        "armed": True,
        "rearm_required": False,
        "emergency_stop": False,
    }


class DepthValidationTests(unittest.TestCase):
    def test_invalid_direct_reading_is_visible_but_not_valid_for_control(self):
        telemetry = healthy_telemetry(12.3456)
        telemetry["sensor_health"].update({
            "depth_valid": False,
            "depth_fault": "depth_out_of_range",
        })
        self.assertEqual(format_depth_display(telemetry), "12.346 m")
        with self.assertRaises(ValueError):
            valid_depth_measurement(telemetry)

    def test_valid_depth_keeps_numeric_display(self):
        self.assertEqual(format_depth_display(healthy_telemetry(0.4567)), "0.457 m")

    def test_stale_pressure_does_not_display_default_zero_as_real_pressure(self):
        telemetry = healthy_telemetry(0.0)
        telemetry["pressure_mpa"] = 0.0
        telemetry["sensor_health"]["pressure_valid"] = False

        self.assertEqual(format_pressure_status_text(telemetry), " · 原始压力 --")

    def test_fresh_pressure_keeps_numeric_status_text(self):
        self.assertEqual(
            format_pressure_status_text(healthy_telemetry()),
            " · 原始压力 0.004200 MPa",
        )

    def test_valid_request_returns_numeric_target_error_and_gains(self):
        request = validate_depth_request(
            healthy_telemetry(0.4), "0.55", "120", "10", "0",
            minimum_depth=-0.2, maximum_depth=10.0, maximum_step=0.75,
        )
        self.assertAlmostEqual(request["target_depth"], 0.55)
        self.assertAlmostEqual(request["depth_error"], 0.15)
        self.assertEqual(request["pid"], (120.0, 10.0, 0.0))

    def test_stale_depth_is_rejected(self):
        telemetry = healthy_telemetry()
        telemetry["sensor_health"]["depth_valid"] = False
        with self.assertRaisesRegex(ValueError, "超时"):
            valid_depth_measurement(telemetry)

    def test_nonfinite_and_oversized_step_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "有限数字"):
            validate_depth_request(
                healthy_telemetry(), "nan", 120, 10, 0,
                minimum_depth=-0.2, maximum_depth=10.0, maximum_step=0.75,
            )
        with self.assertRaisesRegex(ValueError, "超过单次允许"):
            validate_depth_request(
                healthy_telemetry(0.1), 1.0, 120, 10, 0,
                minimum_depth=-0.2, maximum_depth=10.0, maximum_step=0.75,
            )

    def test_negative_pid_gain_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            validate_depth_request(
                healthy_telemetry(), 0.5, 120, -1, 0,
                minimum_depth=-0.2, maximum_depth=10.0, maximum_step=0.75,
            )


class StartOrderingTests(unittest.TestCase):
    def test_depth_pid_is_sent_before_final_enable(self):
        gui = ROVGui.__new__(ROVGui)
        gui._emergency_stop_latched = False
        gui._confirmed_armed_telemetry = lambda: healthy_telemetry(0.4)
        gui._depth_en = Var(True)
        gui._speed_en = Var(False)
        gui._tgt_d_var = Var("0.5")
        gui._tgt_s_var = Var("0")
        gui._d_kp, gui._d_ki, gui._d_kd = Var("120"), Var("10"), Var("0")
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
        calls = gui.client.send_cmd.call_args_list
        self.assertEqual(calls[0].kwargs, {"enable": False})
        self.assertEqual(calls[1].kwargs["pid_set"]["type"], 1)
        self.assertNotIn("enable", calls[1].kwargs)
        self.assertTrue(calls[-1].kwargs["enable"])
        gui._wait_for_pid_ack.assert_called_once()
        gui._wait_for_autonomous_enabled.assert_called_once()

    def test_pid_rejection_never_sends_enable_true(self):
        gui = ROVGui.__new__(ROVGui)
        gui._emergency_stop_latched = False
        gui._confirmed_armed_telemetry = lambda: healthy_telemetry(0.4)
        gui._depth_en, gui._speed_en = Var(True), Var(False)
        gui._tgt_d_var, gui._tgt_s_var = Var("0.5"), Var("0")
        gui._d_kp, gui._d_ki, gui._d_kd = Var("60"), Var("0"), Var("0")
        gui._save_var = Var(False)
        gui._depth_target_min_m = -0.2
        gui._depth_target_max_m = 10.0
        gui._depth_target_max_step_m = 0.75
        gui._log_event = lambda _message: None
        gui.client = mock.Mock()
        gui.client.get_telemetry_marker.return_value = ("pi", 1)
        gui._wait_for_pid_ack = mock.Mock(side_effect=OSError("PID 写入失败"))
        gui._wait_for_autonomous_enabled = mock.Mock()

        with mock.patch("pc_gui.messagebox.showwarning") as warning:
            gui._cmd_start()

        warning.assert_called_once()
        self.assertFalse(any(
            call.kwargs.get("enable") is True
            for call in gui.client.send_cmd.call_args_list
        ))
        gui._wait_for_autonomous_enabled.assert_not_called()


class OneKeyDepthTests(unittest.TestCase):
    def test_shortcut_enables_depth_only(self):
        gui = ROVGui.__new__(ROVGui)
        gui._emergency_stop_latched = False
        gui._confirmed_armed_telemetry = lambda: healthy_telemetry(0.63)
        gui._tgt_d_var = Var("0")
        gui._depth_en = Var(False)
        gui._speed_en = Var(True)
        gui._log_event = mock.Mock()
        gui._cmd_start = mock.Mock()

        with mock.patch("pc_gui.messagebox.showwarning") as warning:
            gui._cmd_hover()

        warning.assert_not_called()
        self.assertEqual(gui._tgt_d_var.get(), "0.630")
        self.assertTrue(gui._depth_en.get())
        self.assertFalse(gui._speed_en.get())
        gui._cmd_start.assert_called_once()


class ControllerDepthShortcutTests(unittest.TestCase):
    def test_square_turns_off_only_depth_hold_when_it_is_running(self):
        gui = ROVGui.__new__(ROVGui)
        gui._emergency_stop_latched = False
        gui._depth_en = Var(True)
        gui._depth_status_var = Var("")
        gui._save_var = Var(False)
        gui._log_event = mock.Mock()
        gui.client = mock.Mock()
        gui.client.get_telemetry.return_value = healthy_telemetry(0.63) | {
            "cmd_enabled": True,
            "mode": {"heading_hold": False, "depth_hold": True, "speed_hold": False},
            "target": {"heading_deg": 25.0, "depth_m": 0.63, "speed_mps": 0.0},
        }

        gui._toggle_depth_hold()

        gui.client.send_cmd.assert_called_once_with(
            enable=False,
            mode={"heading_hold": False, "depth_hold": False, "speed_hold": False},
            target={"heading_deg": 25.0, "depth_m": 0.63, "speed_mps": 0.0},
            save_data=False,
        )
        self.assertFalse(gui._depth_en.get())

    def test_vertical_stick_release_uses_pi_captured_depth_hold_target(self):
        gui = ROVGui.__new__(ROVGui)
        gui._tgt_d_var = Var("0.500")
        gui._depth_status_var = Var("")
        gui._log_event = mock.Mock()
        gui.client = mock.Mock()
        gui.client.get_telemetry.return_value = healthy_telemetry(0.638) | {
            "cmd_enabled": True,
            "mode": {"heading_hold": True, "depth_hold": True, "speed_hold": False},
            "target": {"heading_deg": 25.0, "depth_m": 0.500, "speed_mps": 0.0},
        }

        gui._update_depth_target_after_manual_vertical_move()

        self.assertEqual(gui._tgt_d_var.get(), "0.638")
        gui.client.send_cmd.assert_not_called()

    def test_vertical_motion_queues_one_refresh_on_release(self):
        gui = ROVGui.__new__(ROVGui)
        gui._vertical_manual_active = False
        gui._vertical_target_update_requests = __import__("queue").SimpleQueue()

        gui._track_manual_vertical_motion(3.0)
        gui._track_manual_vertical_motion(3.0)
        gui._track_manual_vertical_motion(0.0)

        self.assertIsNone(gui._vertical_target_update_requests.get_nowait())
        self.assertTrue(gui._vertical_target_update_requests.empty())


class WaterSurfaceZeroTests(unittest.TestCase):
    def test_surface_zero_sends_only_the_dedicated_command_when_locked(self):
        gui = ROVGui.__new__(ROVGui)
        gui.client = mock.Mock()
        gui.client.get_telemetry.return_value = healthy_telemetry(0.08) | {
            "armed": False, "cmd_enabled": False,
        }
        gui._depth_status_var = Var("")
        gui._log_event = mock.Mock()

        with mock.patch("pc_gui.messagebox.showwarning") as warning:
            gui._calibrate_depth_zero()

        warning.assert_not_called()
        gui.client.send_cmd.assert_called_once_with(depth_zero_calibrate=True)
        self.assertIn("等待 Pi 确认", gui._depth_status_var.get())

    def test_surface_zero_is_rejected_while_armed(self):
        gui = ROVGui.__new__(ROVGui)
        gui.client = mock.Mock()
        gui.client.get_telemetry.return_value = healthy_telemetry(0.08)
        gui._depth_status_var = Var("")
        gui._log_event = mock.Mock()

        with mock.patch("pc_gui.messagebox.showwarning") as warning:
            gui._calibrate_depth_zero()

        warning.assert_called_once()
        gui.client.send_cmd.assert_not_called()

    def test_position_zero_accepts_fresh_pressure_when_depth_is_invalid(self):
        gui = ROVGui.__new__(ROVGui)
        gui.client = mock.Mock()
        telemetry = healthy_telemetry(25.0) | {
            "armed": False,
            "cmd_enabled": False,
            "pressure_mpa": 0.2500,
        }
        telemetry["sensor_health"] = {
            "depth_valid": False,
            "depth_fault": "depth_out_of_range",
            "pressure_valid": True,
        }
        gui.client.get_telemetry.return_value = telemetry
        gui._depth_status_var = Var("")
        gui._log_event = mock.Mock()

        with mock.patch("pc_gui.messagebox.showwarning") as warning:
            gui._calibrate_depth_zero()

        warning.assert_not_called()
        gui.client.send_cmd.assert_called_once_with(depth_zero_calibrate=True)


if __name__ == "__main__":
    unittest.main()
