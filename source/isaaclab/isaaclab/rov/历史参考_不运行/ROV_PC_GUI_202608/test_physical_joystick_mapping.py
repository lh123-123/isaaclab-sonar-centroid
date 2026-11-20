import unittest
import queue
from unittest import mock

from pc_gui import (
    DpadAttitudeToggleController,
    PhysicalJoystick,
    ROVGui,
    ROVPoseView,
    step_gripper_jog_position,
)
from secure_protocol import validate_control_payload


class _FakeJoystick:
    def __init__(self, axes, buttons=None):
        self._axes = axes
        self._buttons = buttons or {}

    def get_axis(self, index):
        return self._axes[index]

    def get_button(self, index):
        return self._buttons.get(index, 0)

    def get_numhats(self):
        return 0


class _FakeButtonReader:
    def __init__(self):
        self.buttons = {}

    def is_button_pressed(self, index):
        return bool(self.buttons.get(index, 0))


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def _reader_for_axes(axes, buttons=None):
    reader = PhysicalJoystick.__new__(PhysicalJoystick)
    reader._available = True
    reader._joy = _FakeJoystick(axes, buttons)
    reader._fx_scale = 1.0
    reader._fy_scale = 1.0
    reader._fz_scale = 1.0
    reader._n_scale = 1.0
    reader._m_scale = 1.0
    reader._stick_response_exponent = 2.2
    reader._trigger_response_exponent = 2.2
    return reader


def _gui_for_face_buttons():
    gui = ROVGui.__new__(ROVGui)
    gui._physical_joy = _FakeButtonReader()
    gui._face_button_down = {
        button: False for button in gui._FACE_BUTTON_ACTIONS
    }
    gui._face_button_requests = queue.SimpleQueue()
    return gui


class PhysicalJoystickMappingTests(unittest.TestCase):
    def test_r3_gripper_jog_reaches_full_open_at_the_faster_rate(self):
        open_position = 1.0
        self.assertAlmostEqual(
            step_gripper_jog_position(0.0, 1, 0.0, open_position), 0.05
        )
        self.assertEqual(
            step_gripper_jog_position(open_position, 1, 0.0, open_position),
            open_position,
        )
        self.assertEqual(
            step_gripper_jog_position(0.01, -1, 0.0, open_position), 0.0
        )
        position = 0.0
        for _ in range(20):
            position = step_gripper_jog_position(
                position, 1, 0.0, open_position
            )
        self.assertEqual(position, 1.0)

    def test_r3_opens_and_l3_closes(self):
        self.assertEqual(ROVGui._GRIPPER_OPEN_JOG_BUTTON, 8)
        self.assertEqual(ROVGui._GRIPPER_CLOSE_JOG_BUTTON, 7)

    @mock.patch("pc_gui.update_config_values")
    def test_rb_close_angle_also_updates_the_live_lower_limit(self, save_config):
        gui = ROVGui.__new__(ROVGui)
        gui._servo_angle_min = 0.0
        gui._servo_angle_max = 180.0
        gui._servo_position_var = _Value(0.1)
        gui._gripper_open_angle_var = _Value("120")
        gui._gripper_closed_angle_var = _Value("30")
        gui._gripper_open_position = 1.0
        gui._gripper_closed_position = 0.0
        gui._servo_angle_var = _Value("")
        gui._servo_lower_angle_var = _Value("")
        gui._servo_status_var = _Value("")
        gui._emergency_stop_latched = False
        gui._send_servo_target = mock.Mock()

        result = gui._apply_gripper_positions()

        self.assertEqual(result, "break")
        self.assertAlmostEqual(gui._gripper_open_position, 2.0 / 3.0)
        self.assertAlmostEqual(gui._gripper_closed_position, 1.0 / 6.0)
        self.assertAlmostEqual(gui._servo_position_var.get(), 1.0 / 6.0)
        self.assertEqual(gui._servo_lower_angle_var.get(), "30°")
        gui._send_servo_target.assert_called_once_with(
            1.0 / 6.0, immediate=False
        )
        save_config.assert_called_once()
        saved = save_config.call_args.args[2]
        self.assertAlmostEqual(saved["closed_position"], 1.0 / 6.0)
        self.assertAlmostEqual(saved["failsafe_position"], 1.0 / 6.0)
        self.assertIn("下限/RB关 30.0°", gui._servo_status_var.value)

    def test_gripper_toggle_entries_reject_closed_position_above_open_position(self):
        gui = ROVGui.__new__(ROVGui)
        gui._servo_angle_min = 0.0
        gui._servo_angle_max = 180.0
        gui._gripper_open_angle_var = _Value("30")
        gui._gripper_closed_angle_var = _Value("40")
        gui._gripper_open_position = 0.3
        gui._gripper_closed_position = 0.1
        gui._servo_status_var = _Value("")

        result = gui._apply_gripper_positions()

        self.assertEqual(result, "break")
        self.assertEqual(gui._gripper_open_position, 0.3)
        self.assertEqual(gui._gripper_closed_position, 0.1)
        self.assertIn("闭合角度", gui._servo_status_var.value)

    def test_servo_target_has_no_periodic_resend(self):
        self.assertFalse(hasattr(ROVGui, "_refresh_servo_target"))
        self.assertFalse(hasattr(ROVGui, "_on_servo_slider"))

    def test_servo_target_retries_are_bounded_and_stop_after_fresh_ack(self):
        class Root:
            def __init__(self):
                self.callbacks = []

            def after(self, delay, callback):
                self.callbacks.append((delay, callback))

        class Client:
            def __init__(self):
                self.sent = []
                self.telemetry = {}

            def get_telemetry_marker(self):
                return ("pi", 10)

            def get_telemetry(self):
                return self.telemetry

            def send_cmd(self, **kwargs):
                self.sent.append(kwargs)

        gui = ROVGui.__new__(ROVGui)
        gui.root = Root()
        gui.client = Client()
        gui._servo_command_generation = 0
        gui._emergency_stop_latched = False
        gui._gripper_closed_position = 0.1

        gui._send_servo_target(0.4)
        self.assertEqual(gui.client.sent, [{
            "servo_position": 0.4,
            "servo_min_position": 0.1,
            "servo_immediate": False,
        }])
        self.assertEqual(len(gui.root.callbacks), 2)

        gui.root.callbacks[0][1]()
        self.assertEqual(len(gui.client.sent), 2)

        gui.client.telemetry = {
            "_rx_marker": ("pi", 11),
            "servo": {"target_position": 0.4},
            "servo_ack": {
                "ok": True,
                "message": "servo_target_updated",
                "servo_target_position": 0.4,
            },
        }
        gui.root.callbacks[1][1]()
        self.assertEqual(len(gui.client.sent), 2)

    def test_protocol_accepts_dynamic_servo_minimum(self):
        payload = validate_control_payload({
            "servo_position": 2.0 / 3.0,
            "servo_min_position": 1.0 / 6.0,
            "servo_immediate": True,
        })
        self.assertAlmostEqual(payload["servo_min_position"], 1.0 / 6.0)

    def test_lb_mode_button_queues_one_rising_edge(self):
        gui = _gui_for_face_buttons()
        gui._performance_button_down = False
        gui._performance_mode_requests = queue.SimpleQueue()

        gui._physical_joy.buttons = {gui._PERFORMANCE_MODE_BUTTON: 1}
        gui._sample_performance_mode_button(allow_toggle=True)
        gui._sample_performance_mode_button(allow_toggle=True)
        self.assertIsNone(gui._performance_mode_requests.get_nowait())
        with self.assertRaises(queue.Empty):
            gui._performance_mode_requests.get_nowait()

    def test_r3_gui_jog_uses_the_full_open_endpoint(self):
        gui = ROVGui.__new__(ROVGui)
        gui._gripper_jog_direction = 1
        gui._emergency_stop_latched = False
        gui._servo_position_var = _Value(0.98)
        gui._servo_status_var = _Value("")
        gui._servo_angle_min = 0.0
        gui._servo_angle_max = 180.0
        gui._gripper_closed_position = 0.0
        gui._gripper_open_position = 0.4594594595  # RB remains unchanged.
        gui._gripper_jog_open_position = 1.0
        gui._gripper_jog_step = 0.05
        gui._gripper_close_on_next_press = False
        gui._servo_angle_var = _Value("")
        gui._send_servo_target = mock.Mock()

        gui._apply_gripper_jog_step()

        self.assertEqual(gui._servo_position_var.get(), 1.0)
        gui._send_servo_target.assert_called_once_with(1.0, immediate=False)
        self.assertIn("R3", gui._servo_status_var.value)
        self.assertFalse(gui._gripper_close_on_next_press)

    def test_l3_jog_does_not_overwrite_the_last_full_toggle_action(self):
        gui = ROVGui.__new__(ROVGui)
        gui._gripper_jog_direction = -1
        gui._emergency_stop_latched = False
        gui._servo_position_var = _Value(0.50)
        gui._servo_status_var = _Value("")
        gui._servo_angle_min = 0.0
        gui._servo_angle_max = 180.0
        gui._gripper_closed_position = 0.0
        gui._gripper_open_position = 2.0 / 3.0
        gui._gripper_jog_open_position = 1.0
        gui._gripper_jog_step = 0.05
        gui._gripper_close_on_next_press = True
        gui._servo_angle_var = _Value("")
        gui._send_servo_target = mock.Mock()

        gui._apply_gripper_jog_step()

        self.assertTrue(gui._gripper_close_on_next_press)

    def test_full_open_then_jog_still_makes_the_next_toggle_close(self):
        gui = ROVGui.__new__(ROVGui)
        gui._emergency_stop_latched = False
        gui._gripper_close_on_next_press = False
        gui._gripper_open_position = 2.0 / 3.0
        gui._gripper_closed_position = 1.0 / 18.0
        gui._gripper_jog_open_position = 1.0
        gui._gripper_jog_step = 0.05
        gui._servo_angle_min = 0.0
        gui._servo_angle_max = 180.0
        gui._servo_position_var = _Value(gui._gripper_closed_position)
        gui._servo_status_var = _Value("")
        gui._servo_angle_var = _Value("")
        gui._send_servo_target = mock.Mock()
        gui._log_event = mock.Mock()

        gui._toggle_gripper()
        gui._gripper_jog_direction = -1
        gui._apply_gripper_jog_step()
        gui._toggle_gripper()

        self.assertEqual(
            gui._send_servo_target.call_args_list[-1],
            mock.call(gui._gripper_closed_position, immediate=True),
        )
        self.assertFalse(gui._gripper_close_on_next_press)
        self.assertEqual(gui._expanded_gripper_next_text(), "下次开")

    def test_l3_r3_held_state_selects_one_jog_direction(self):
        gui = ROVGui.__new__(ROVGui)
        gui._physical_joy = _FakeButtonReader()
        gui._gripper_jog_direction = 0

        gui._physical_joy.buttons = {gui._GRIPPER_OPEN_JOG_BUTTON: 1}
        gui._sample_gripper_jog(allow_jog=True)
        self.assertEqual(gui._gripper_jog_direction, 1)

        gui._physical_joy.buttons = {gui._GRIPPER_CLOSE_JOG_BUTTON: 1}
        gui._sample_gripper_jog(allow_jog=True)
        self.assertEqual(gui._gripper_jog_direction, -1)

        gui._physical_joy.buttons = {
            gui._GRIPPER_OPEN_JOG_BUTTON: 1,
            gui._GRIPPER_CLOSE_JOG_BUTTON: 1,
        }
        gui._sample_gripper_jog(allow_jog=True)
        self.assertEqual(gui._gripper_jog_direction, 0)

        gui._physical_joy.buttons = {gui._GRIPPER_OPEN_JOG_BUTTON: 1}
        gui._sample_gripper_jog(allow_jog=False)
        self.assertEqual(gui._gripper_jog_direction, 0)

    def test_pose_view_uses_the_exact_bounded_telemetry_triplet(self):
        view = ROVPoseView.__new__(ROVPoseView)
        view._target_angles = [0.0, 0.0, 0.0]
        view._display_angles = [0.0, 0.0, 0.0]
        view._connected = False
        view._dirty = False

        view.set_pose(541.0, 12.5, -541.0, connected=True)

        self.assertEqual(view._target_angles, [-179.0, 12.5, 179.0])
        self.assertEqual(view._display_angles, view._target_angles)
        self.assertTrue(view._connected)
        self.assertTrue(view._dirty)

    def test_pose_view_wrap_boundary_never_animates_through_181_degrees(self):
        view = ROVPoseView.__new__(ROVPoseView)
        view._target_angles = [179.8, 0.0, 179.8]
        view._display_angles = [179.8, 0.0, 179.8]
        view._connected = True
        view._dirty = False

        view.set_pose(-179.0, 0.0, -179.0, connected=True)

        self.assertEqual(view._display_angles, [-179.0, 0.0, -179.0])

    def test_left_stick_controls_lateral_and_right_stick_controls_yaw(self):
        # Left-right input on each stick must use the requested channels only.
        reader = _reader_for_axes([0.9, 0.0, 0.8, 0.0, -1.0, -1.0])
        fx, fy, fz, yaw, pitch = reader.read()
        self.assertEqual((fx, fz, pitch), (0.0, 0.0, 0.0))
        self.assertGreater(fy, 0.0)
        self.assertGreater(yaw, 0.0)

    def test_each_stick_keeps_only_its_dominant_cardinal_direction(self):
        # A diagonal push is still exactly one requested motion per stick.
        reader = _reader_for_axes([0.2, -0.9, 0.3, -0.8, -1.0, -1.0])
        fx, fy, fz, yaw, pitch = reader.read()
        self.assertGreater(fx, 0.0)
        self.assertEqual(fy, 0.0)
        # Vehicle body +Z points down.  Pushing the stick forward/up requests
        # ascent, therefore its force command must be negative Fz.
        self.assertLess(fz, 0.0)
        self.assertEqual(yaw, 0.0)
        self.assertEqual(pitch, 0.0)

    def test_right_stick_up_is_negative_body_fz_and_down_is_positive(self):
        up = _reader_for_axes([0.0, 0.0, 0.0, -1.0, -1.0, -1.0]).read()
        down = _reader_for_axes([0.0, 0.0, 0.0, 1.0, -1.0, -1.0]).read()

        self.assertLess(up[2], 0.0)
        self.assertGreater(down[2], 0.0)

    def test_dpad_is_reported_separately_from_the_analog_stick(self):
        reader = _reader_for_axes(
            [0.9, 0.0, 0.0, 0.0, -1.0, -1.0],
            buttons={PhysicalJoystick._DPAD_UP_BUTTON: 1},
        )
        fx, fy, fz, yaw, pitch = reader.read()
        self.assertGreater(fy, 0.0)
        self.assertEqual((fx, fz, yaw, pitch), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(reader.read_dpad(), (0.0, 1.0))

    def test_dpad_right_is_reported_as_one_cardinal_direction(self):
        reader = _reader_for_axes(
            [0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
            buttons={PhysicalJoystick._DPAD_RIGHT_BUTTON: 1},
        )
        fx, fy, fz, yaw, pitch = reader.read()
        self.assertEqual((fx, fy, fz, yaw, pitch), (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(reader.read_dpad(), (1.0, 0.0))

    def test_mid_travel_uses_the_progressive_curve_for_sticks_and_triggers(self):
        reader = _reader_for_axes([0.5, 0.0, 0.0, 0.0, 0.0, -1.0])
        fx, fy, fz, yaw, pitch = reader.read()
        self.assertEqual((fx, fz, yaw), (0.0, 0.0, 0.0))
        self.assertEqual(fy, 0.1)
        self.assertEqual(pitch, 0.1)

    def test_dpad_directions_emit_one_action_edge_without_force_values(self):
        toggles = DpadAttitudeToggleController()
        self.assertEqual(
            toggles.update((0.0, 1.0), allow_trigger=True),
            "pitch_hold_toggle",
        )
        self.assertIsNone(toggles.update((0.0, 1.0), allow_trigger=True))
        self.assertIsNone(toggles.update((0.0, 0.0), allow_trigger=True))
        self.assertEqual(
            toggles.update((1.0, 0.0), allow_trigger=True),
            "roll_hold_toggle",
        )
        self.assertIsNone(toggles.update((0.0, 0.0), allow_trigger=True))
        self.assertEqual(
            toggles.update((-1.0, 0.0), allow_trigger=True),
            "yaw_hold_toggle",
        )
        self.assertIsNone(toggles.update((0.0, 0.0), allow_trigger=True))
        self.assertEqual(
            toggles.update((0.0, -1.0), allow_trigger=True),
            "timer_toggle",
        )
        self.assertIsNone(toggles.update((0.0, -1.0), allow_trigger=True))

    def test_inhibited_dpad_requires_release_before_mode_edge(self):
        toggles = DpadAttitudeToggleController()
        self.assertIsNone(toggles.update((0.0, 1.0), allow_trigger=False))
        self.assertIsNone(toggles.update((0.0, 1.0), allow_trigger=True))
        toggles.update((0.0, 0.0), allow_trigger=True)
        self.assertEqual(
            toggles.update((0.0, 1.0), allow_trigger=True),
            "pitch_hold_toggle",
        )

    def test_face_button_press_edges_queue_the_expected_actions(self):
        gui = _gui_for_face_buttons()
        for button, expected_action in gui._FACE_BUTTON_ACTIONS.items():
            gui._physical_joy.buttons = {button: 1}
            gui._sample_face_buttons(allow_actions=True)
            self.assertEqual(gui._face_button_requests.get_nowait(), expected_action)

            # A held button must not repeatedly enqueue at the 40 Hz rate.
            gui._sample_face_buttons(allow_actions=True)
            self.assertTrue(gui._face_button_requests.empty())
            gui._physical_joy.buttons = {}
            gui._sample_face_buttons(allow_actions=True)

    def test_pc_protocol_accepts_the_options_attitude_zero_command(self):
        self.assertEqual(
            validate_control_payload({"attitude_zero_calibrate": True}),
            {"attitude_zero_calibrate": True},
        )

    def test_face_actions_call_the_safe_gui_operations(self):
        gui = ROVGui.__new__(ROVGui)
        gui._emergency_stop_latched = False
        actions = []
        gui._toggle_arm = lambda: actions.append("arm")
        gui._toggle_emergency_stop = lambda: actions.append("estop")
        gui._toggle_depth_hold = lambda: actions.append("depth_toggle")
        gui._calibrate_depth_zero = lambda: actions.append("surface_zero")
        gui._calibrate_attitude_zero = lambda: actions.append("attitude_zero")
        gui._toggle_attitude_hold = lambda axis: actions.append(f"{axis}_hold")

        class _VideoView:
            def toggle_expanded_view(self):
                actions.append("video_toggle")

        gui._video_view = _VideoView()

        class _Timer:
            def _toggle_run(self):
                actions.append("timer_toggle")

        gui._timer = _Timer()

        class _Client:
            def get_telemetry(self):
                return {"cmd_enabled": False}

        gui.client = _Client()
        for action in (
            "arm", "estop", "depth_toggle", "video_toggle", "surface_zero",
            "attitude_zero", "pitch_hold_toggle", "roll_hold_toggle",
            "yaw_hold_toggle", "timer_toggle",
        ):
            gui._handle_face_button_action(action)
        self.assertEqual(
            actions,
            [
                "arm", "estop", "depth_toggle", "video_toggle", "surface_zero",
                "attitude_zero", "pitch_hold", "roll_hold", "yaw_hold",
                "timer_toggle",
            ],
        )

    def test_pi_referenced_yaw_is_normalized_to_signed_range(self):
        gui = ROVGui.__new__(ROVGui)
        gui._imu_roll_inverted = False
        gui._imu_pitch_inverted = True
        gui._imu_yaw_inverted = True
        gui._imu_heading_inverted = True
        gui._imu_yaw_offset_deg = 0.0
        gui._imu_heading_offset_deg = 0.0
        vehicle = gui._vehicle_imu({"roll": 6.0, "pitch": -2.0, "yaw": 185.0, "heading": 0.0})
        self.assertEqual(vehicle["yaw"], -175.0)

    def test_gui_defensively_bounds_all_pi_euler_angles_to_signed_range(self):
        gui = ROVGui.__new__(ROVGui)
        gui._imu_heading_inverted = False
        gui._imu_heading_offset_deg = 0.0
        vehicle = gui._vehicle_imu({
            "roll": 541.0, "pitch": 541.0, "yaw": -541.0, "heading": 290.0,
        })
        self.assertEqual(vehicle["roll"], -179.0)
        self.assertEqual(vehicle["pitch"], -179.0)
        self.assertEqual(vehicle["yaw"], 179.0)
        self.assertEqual(vehicle["heading"], 290.0)

    def test_independent_imu_display_alignment_matches_the_bench_result(self):
        gui = ROVGui.__new__(ROVGui)
        gui._imu_roll_inverted = False
        gui._imu_pitch_inverted = True
        gui._imu_yaw_inverted = True
        gui._imu_heading_inverted = True
        gui._imu_yaw_offset_deg = 0.0
        gui._imu_heading_offset_deg = 0.0
        vehicle = gui._vehicle_imu({
            "roll": 8.0, "pitch": -5.0, "yaw": 185.0, "heading": 350.0,
        })
        self.assertEqual(vehicle, {
            "roll": 8.0, "pitch": -5.0, "yaw": -175.0, "heading": 10.0,
        })

    def test_attitude_zero_requires_locked_fresh_pi_telemetry(self):
        gui = ROVGui.__new__(ROVGui)
        events = []
        gui._log_event = events.append

        class _Client:
            def get_telemetry(self):
                return {
                    "armed": False,
                    "cmd_enabled": False,
                    "imu": {"roll": 8.0, "pitch": -5.0, "yaw": 175.0},
                    "sensor_health": {"imu_valid": True},
                    "attitude_3d_mount_calibrated": True,
                    "attitude_hold": {"roll": False, "pitch": False},
                }

            def send_cmd(self, **kwargs):
                self.command = kwargs

        gui.client = _Client()
        with mock.patch("pc_gui.messagebox.askyesno", return_value=True):
            gui._calibrate_attitude_zero()
        self.assertEqual(gui.client.command, {"attitude_zero_calibrate": True})
        self.assertIn("全姿态置零", events[0])

    def test_attitude_zero_rejects_active_hold_or_uncalibrated_mount(self):
        gui = ROVGui.__new__(ROVGui)
        events = []
        gui._log_event = events.append

        class _Client:
            command = None

            def __init__(self, calibrated, hold):
                self.calibrated = calibrated
                self.hold = hold

            def get_telemetry(self):
                return {
                    "armed": False,
                    "cmd_enabled": False,
                    "imu": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                    "sensor_health": {"imu_valid": True},
                    "attitude_3d_mount_calibrated": self.calibrated,
                    "attitude_hold": self.hold,
                }

            def send_cmd(self, **kwargs):
                self.command = kwargs

        gui.client = _Client(False, {"roll": False, "pitch": False})
        gui._calibrate_attitude_zero()
        self.assertIsNone(gui.client.command)
        self.assertIn("安装标定", events[-1])

        gui.client = _Client(True, {"roll": False, "pitch": True})
        gui._calibrate_attitude_zero()
        self.assertIsNone(gui.client.command)
        self.assertIn("姿态锁定", events[-1])


if __name__ == "__main__":
    unittest.main()
