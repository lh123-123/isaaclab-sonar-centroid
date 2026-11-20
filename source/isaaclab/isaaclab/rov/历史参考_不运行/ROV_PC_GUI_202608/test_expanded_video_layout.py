import threading
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image

from pc_gui import (
    CompetitionTimer,
    ROVGui,
    VideoStream,
    expanded_attitude_hold_texts,
    expanded_status_side_width,
    expanded_canvas_to_video_point,
    expanded_mission_map_layout,
    expanded_video_layout,
    load_video_annotations,
    save_video_annotations,
    split_motion_command,
    video_point_to_expanded_canvas,
    video_cover_crop,
)


class ExpandedVideoLayoutTests(unittest.TestCase):
    def test_wide_window_is_filled_edge_to_edge(self):
        left, top, width, height = expanded_video_layout(2048, 1188, 640, 480)

        self.assertEqual((left, top, width, height), (0, 0, 2048, 1188))

    def test_narrower_window_is_also_filled_edge_to_edge(self):
        left, top, width, height = expanded_video_layout(1280, 720, 640, 480)

        self.assertEqual((left, top, width, height), (0, 0, 1280, 720))

    def test_cover_crop_fills_wide_panel_without_stretching(self):
        self.assertEqual(
            video_cover_crop(1280, 360, 1280, 720),
            (0, 180, 1280, 360),
        )

    def test_expanded_status_anchors_keep_their_existing_side_width(self):
        self.assertEqual(expanded_status_side_width(1280), 180)
        self.assertEqual(expanded_status_side_width(600), 140)

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            expanded_video_layout(0, 720, 640, 480)

    def test_mission_map_is_small_bottom_left_and_clear_of_image_centre(self):
        left, top, width, height = expanded_mission_map_layout(
            1920, 1080, 1822, 863
        )

        self.assertEqual(left, 16)
        self.assertEqual(top + height, 1080 - 16)
        self.assertLess(left + width, 1920 * 0.30)
        self.assertLess(height, 1080 * 0.25)

    def test_mission_map_asset_is_high_resolution(self):
        map_path = Path(__file__).with_name("assets") / "mission_map_overlay.png"

        with Image.open(map_path) as image:
            self.assertGreaterEqual(image.width, 1800)
            self.assertGreaterEqual(image.height, 850)

    def test_annotation_point_round_trips_through_cover_crop(self):
        point = expanded_canvas_to_video_point(320, 180, 1280, 720, 640, 480)

        self.assertAlmostEqual(point[0], 0.25)
        self.assertAlmostEqual(point[1], 0.3125)
        self.assertEqual(
            video_point_to_expanded_canvas(point, 1280, 720, 640, 480),
            (320.0, 180.0),
        )

    def test_expanded_and_embedded_canvases_both_receive_each_frame(self):
        stream = VideoStream.__new__(VideoStream)
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        embedded_canvas = MagicMock()
        embedded_canvas.winfo_width.return_value = 640
        embedded_canvas.winfo_height.return_value = 480
        expanded_canvas = MagicMock()
        expanded_window = MagicMock()
        expanded_window.winfo_exists.return_value = True
        stream._lock = threading.Lock()
        stream._frame = frame
        stream._last_frame = None
        stream._last_fps_time = 0.0
        stream._frame_count = 1
        stream._fps = 0.0
        stream._fps_label = MagicMock()
        stream.canvas = embedded_canvas
        stream.running = True
        stream.cap = object()
        stream._placeholder_drawn = False
        stream._expanded_window = expanded_window
        stream._expanded_canvas = expanded_canvas
        stream._render_frame = MagicMock()

        with patch("pc_gui.time.monotonic", return_value=1.0):
            stream.update_display()

        calls = stream._render_frame.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0].args[0], embedded_canvas)
        self.assertIs(calls[0].args[1], frame)
        self.assertFalse(calls[0].kwargs.get("draw_depth", False))
        self.assertIs(calls[1].args[0], expanded_canvas)
        self.assertIs(calls[1].args[1], frame)
        self.assertTrue(calls[1].kwargs["draw_depth"])

    def test_controller_toggle_opens_then_closes_the_expanded_video(self):
        stream = VideoStream.__new__(VideoStream)
        stream._expanded_window = None
        stream._expanded_canvas = None
        stream._open_expanded_view = MagicMock()
        stream._close_expanded_view = MagicMock()

        stream.toggle_expanded_view()
        stream._open_expanded_view.assert_called_once_with()
        stream._close_expanded_view.assert_not_called()

        stream._expanded_window = MagicMock()
        stream._expanded_window.winfo_exists.return_value = True
        stream.toggle_expanded_view()
        stream._close_expanded_view.assert_called_once_with()

    def test_two_annotation_clicks_create_one_persistent_video_line(self):
        stream = VideoStream.__new__(VideoStream)
        stream._annotation_mode = True
        stream._annotation_pending_point = None
        stream._annotation_lines = []
        stream._last_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        stream._expanded_canvas = MagicMock()
        stream._expanded_canvas.winfo_width.return_value = 1280
        stream._expanded_canvas.winfo_height.return_value = 720
        stream._render_expanded_frame = MagicMock()
        stream._persist_annotations = MagicMock()

        stream._on_expanded_annotation_click(MagicMock(x=200, y=300))
        stream._on_expanded_annotation_click(MagicMock(x=900, y=500))

        self.assertIsNone(stream._annotation_pending_point)
        self.assertEqual(
            stream._annotation_lines,
            [((200 / 1280, 300 / 720), (900 / 1280, 500 / 720))],
        )
        stream._persist_annotations.assert_called_once_with()

    def test_fast_double_click_still_supplies_the_second_point(self):
        stream = VideoStream.__new__(VideoStream)
        stream._annotation_mode = True
        stream._annotation_pending_point = (0.25, 0.25)
        stream._annotation_lines = []
        stream._last_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        stream._expanded_canvas = MagicMock()
        stream._expanded_canvas.winfo_width.return_value = 1280
        stream._expanded_canvas.winfo_height.return_value = 720
        stream._render_expanded_frame = MagicMock()
        stream._persist_annotations = MagicMock()

        result = stream._on_expanded_double_click(MagicMock(x=960, y=540))

        self.assertEqual(result, "break")
        self.assertEqual(
            stream._annotation_lines,
            [((0.25, 0.25), (0.75, 0.75))],
        )

    def test_r_is_debounced_and_t_clears_lines(self):
        stream = VideoStream.__new__(VideoStream)
        stream._annotation_mode = False
        stream._annotation_pending_point = None
        stream._annotation_lines = [((0.1, 0.2), (0.3, 0.4))]
        stream._annotation_pressed_keys = set()
        stream._expanded_canvas = MagicMock()
        stream._render_expanded_frame = MagicMock()
        stream._persist_annotations = MagicMock()

        event_r = MagicMock(keysym="r")
        stream._on_expanded_annotation_key_press(event_r)
        stream._on_expanded_annotation_key_press(event_r)
        self.assertTrue(stream._annotation_mode)
        stream._on_expanded_annotation_key_release(event_r)
        stream._on_expanded_annotation_key_press(event_r)
        self.assertFalse(stream._annotation_mode)

        stream._on_expanded_annotation_key_release(event_r)
        stream._on_expanded_annotation_key_press(MagicMock(keysym="t"))
        self.assertEqual(stream._annotation_lines, [])
        stream._persist_annotations.assert_called_once_with()

    def test_annotation_is_redrawn_at_one_millimetre_width(self):
        stream = VideoStream.__new__(VideoStream)
        stream._annotation_mode = False
        stream._annotation_pending_point = None
        stream._annotation_lines = [((0.1, 0.2), (0.8, 0.7))]
        canvas = MagicMock()
        canvas.winfo_fpixels.return_value = 3.78

        stream._draw_expanded_annotations(canvas, 1280, 720, 1280, 720)

        line_call = canvas.create_line.call_args
        self.assertEqual(line_call.kwargs["fill"], "#FF8F8F")
        self.assertAlmostEqual(line_call.kwargs["width"], 3.78)

    def test_annotations_survive_a_save_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_annotations.json"
            lines = [((0.1, 0.2), (0.8, 0.7))]

            save_video_annotations(path, lines)

            self.assertEqual(load_video_annotations(path), lines)

    def test_corrupt_annotation_file_fails_closed_to_no_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_annotations.json"
            path.write_text('{"version":1,"lines":[[[-1,0],[1,1]]]}', encoding="utf-8")

            self.assertEqual(load_video_annotations(path), [])


class ExpandedMotionTextTests(unittest.TestCase):
    def test_motion_label_and_action_are_split(self):
        label, action, color = split_motion_command("运动指令：前进")

        self.assertEqual(label, "运动指令：")
        self.assertEqual(action, "前进")
        self.assertEqual(color, "#FFD166")

    def test_unarmed_action_is_red(self):
        label, action, color = split_motion_command("运动指令：未解锁")

        self.assertEqual(label, "运动指令：")
        self.assertEqual(action, "未解锁")
        self.assertEqual(color, "#FF4D4F")


class ExpandedAttitudeHoldTextTests(unittest.TestCase):
    def test_pitch_roll_then_yaw_are_reported_from_pi_telemetry(self):
        pitch_text, roll_text, yaw_text = expanded_attitude_hold_texts({
            "attitude_hold": {"pitch": True, "roll": False, "yaw": True},
        })

        self.assertEqual(pitch_text, "锁俯仰：开启")
        self.assertEqual(roll_text, "锁横滚：关闭")
        self.assertEqual(yaw_text, "锁偏转：开启")

    def test_missing_hold_telemetry_is_unknown(self):
        self.assertEqual(
            expanded_attitude_hold_texts({}),
            ("锁俯仰：未知", "锁横滚：未知", "锁偏转：未知"),
        )

    def test_expanded_status_keeps_hold_rows_between_depth_and_motion(self):
        gui = ROVGui.__new__(ROVGui)
        gui.client = MagicMock()
        gui.client.get_telemetry.return_value = {
            "cmd_enabled": True,
            "mode": {"depth_hold": True},
            "target": {"depth_m": 0.82},
            "imu": {"roll": -4.5, "pitch": 7.25, "yaw": -32.0},
            "attitude_hold": {
                "pitch": False, "roll": True, "yaw": True,
                "yaw_control_deg": -32.0,
                "yaw_target_deg": -28.5,
            },
            "armed": True,
            "control_link_ok": True,
            "joy": {},
        }
        gui._expanded_video_depth_text = MagicMock(return_value="深度 1.000 m")

        self.assertEqual(
            gui._expanded_video_status(),
            (
                "深度 1.000 m", "目标深度 0.820 m", "定深：已开启",
                "实时 Yaw -32.0°", "目标 Yaw -28.5°",
                "俯仰 +7.2°", "横滚 -4.5°",
                "锁俯仰：关闭", "锁横滚：开启", "锁偏转：开启",
                "运动指令：定深保持",
            ),
        )

    def test_legacy_three_row_provider_gets_three_safe_unknown_rows(self):
        stream = VideoStream.__new__(VideoStream)
        stream._status_provider = lambda: (
            "深度 1.000 m", "定深：已开启", "运动指令：静止",
        )
        stream._depth_provider = None

        self.assertEqual(
            stream._expanded_status_texts(),
            (
                "深度 1.000 m", "目标深度 --", "定深：已开启",
                "实时 Yaw --°", "目标 Yaw --°",
                "俯仰 --°", "横滚 --°",
                "锁俯仰：未知", "锁横滚：未知", "锁偏转：未知",
                "运动指令：静止",
            ),
        )

    def test_hold_rows_share_green_left_alignment_without_overlapping_motion(self):
        stream = VideoStream.__new__(VideoStream)
        stream._status_provider = lambda: (
            "深度 1.000 m", "目标深度 0.820 m", "定深：已开启",
            "实时 Yaw -32.0°", "目标 Yaw -28.5°",
            "俯仰 +7.2°", "横滚 -4.5°",
            "锁俯仰：开启", "锁横滚：关闭", "锁偏转：开启",
            "运动指令：静止",
        )
        stream._depth_provider = None
        stream._timer_provider = lambda: ("00:00", "正计时 · 待开始")
        stream._gripper_next_provider = lambda: "下次关"
        canvas = MagicMock()
        canvas.winfo_width.return_value = 1280
        canvas.winfo_height.return_value = 720
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("pc_gui.ImageTk.PhotoImage", return_value=object()):
            stream._render_frame(canvas, frame, draw_depth=True)

        canvas.create_rectangle.assert_not_called()
        text_calls = {
            call.kwargs["text"]: call
            for call in canvas.create_text.call_args_list
        }
        depth_call = text_calls["定深：已开启"]
        current_depth_call = text_calls["深度 1.000 m"]
        target_depth_call = text_calls["目标深度 0.820 m"]
        yaw_value_call = text_calls["实时 Yaw -32.0°"]
        yaw_target_call = text_calls["目标 Yaw -28.5°"]
        pitch_value_call = text_calls["俯仰 +7.2°"]
        roll_value_call = text_calls["横滚 -4.5°"]
        gripper_next_call = text_calls["下次关"]
        pitch_call = text_calls["锁俯仰：开启"]
        roll_call = text_calls["锁横滚：关闭"]
        yaw_call = text_calls["锁偏转：开启"]
        motion_call = text_calls["运动指令："]

        self.assertEqual(
            [depth_call.args[0], pitch_call.args[0], roll_call.args[0], yaw_call.args[0]],
            [depth_call.args[0]] * 4,
        )
        self.assertEqual(
            [depth_call.kwargs["fill"], pitch_call.kwargs["fill"], roll_call.kwargs["fill"], yaw_call.kwargs["fill"]],
            ["#4FE09B"] * 4,
        )
        self.assertLess(depth_call.args[1], pitch_call.args[1])
        self.assertLess(pitch_call.args[1], roll_call.args[1])
        self.assertLess(roll_call.args[1], yaw_call.args[1])
        self.assertLess(yaw_call.args[1], motion_call.args[1])
        self.assertLess(current_depth_call.args[1], target_depth_call.args[1])
        self.assertLess(target_depth_call.args[1], yaw_value_call.args[1])
        self.assertLess(yaw_value_call.args[1], yaw_target_call.args[1])
        self.assertLess(yaw_target_call.args[1], depth_call.args[1])
        self.assertLess(depth_call.args[1], pitch_call.args[1])
        self.assertEqual(yaw_target_call.kwargs["fill"], target_depth_call.kwargs["fill"])
        self.assertEqual(yaw_target_call.kwargs["font"], target_depth_call.kwargs["font"])
        for numeric_call in (
            current_depth_call, target_depth_call,
            yaw_value_call, yaw_target_call,
            pitch_value_call, roll_value_call,
        ):
            self.assertEqual(
                numeric_call.kwargs["width"], 0,
                "numeric HUD rows must stay on one line at Windows DPI scaling",
            )
        self.assertEqual(pitch_value_call.args[0], roll_value_call.args[0])
        self.assertLess(pitch_value_call.args[1], roll_value_call.args[1])
        self.assertEqual(gripper_next_call.args[0], roll_value_call.args[0])
        self.assertGreater(gripper_next_call.args[1], roll_value_call.args[1])
        self.assertLess(pitch_value_call.args[0], 180)
        self.assertGreater(current_depth_call.args[0], 1100)


class CompetitionTimerTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.timer = CompetitionTimer(self.root)

    def tearDown(self):
        self.root.destroy()

    def test_ten_minute_countdown_does_not_finish_when_started(self):
        self.timer._set_countdown(600)
        self.assertEqual(self.timer._time_var.get(), "10:00")

        with patch("pc_gui.time.monotonic", side_effect=(100.0, 101.0)):
            self.timer._toggle_run()
            self.timer.tick()

        self.assertTrue(self.timer._running)
        self.assertEqual(self.timer._time_var.get(), "09:59")

    def test_panel_and_expanded_timer_have_no_decimal_part(self):
        self.timer._set_countdown(600)
        time_text, state_text = self.timer.expanded_status()

        self.assertEqual(time_text, "10:00")
        self.assertNotIn(".", time_text)
        self.assertEqual(state_text, "倒计时 10 min · 待开始")

    def test_reset_returns_to_zeroed_count_up_mode(self):
        self.timer._set_countdown(600)
        self.timer._reset()

        self.assertEqual(self.timer._mode, "up")
        self.assertEqual(self.timer._time_var.get(), "00:00")
        self.assertEqual(self.timer.expanded_status(), ("00:00", "正计时 · 待开始"))

    def test_start_after_finished_restarts_selected_countdown(self):
        self.timer._set_countdown(600)
        self.timer._elapsed = 600.0
        self.timer._update_display()

        with patch("pc_gui.time.monotonic", return_value=100.0):
            self.timer._toggle_run()

        self.assertTrue(self.timer._running)
        self.assertEqual(self.timer._elapsed, 0.0)

    def test_repeated_toggle_pauses_and_resumes_without_resetting_elapsed_time(self):
        with patch("pc_gui.time.monotonic", side_effect=(100.0, 101.5, 200.0, 201.0)):
            self.timer._toggle_run()
            self.timer.tick()
            self.timer._toggle_run()
            paused_elapsed = self.timer._elapsed
            self.timer._toggle_run()
            self.timer.tick()

        self.assertAlmostEqual(paused_elapsed, 1.5)
        self.assertTrue(self.timer._running)
        self.assertAlmostEqual(self.timer._elapsed, 2.5)


if __name__ == "__main__":
    unittest.main()
