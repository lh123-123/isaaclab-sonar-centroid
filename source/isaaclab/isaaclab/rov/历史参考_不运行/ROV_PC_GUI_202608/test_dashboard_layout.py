import inspect
import tkinter as tk
import time
import unittest

from pc_gui import (
    CompetitionTimer,
    DASHBOARD_REFRESH_INTERVAL_MS,
    FAST_LABEL_REFRESH_INTERVAL_MS,
    POSE_RENDER_INTERVAL_MS,
    ROVGui,
    TrendChart,
    fit_main_window_geometry,
)


class DashboardLayoutTests(unittest.TestCase):
    def test_heavy_dashboard_work_is_slower_than_fast_labels(self):
        self.assertEqual(FAST_LABEL_REFRESH_INTERVAL_MS, 50)
        self.assertEqual(DASHBOARD_REFRESH_INTERVAL_MS, 200)
        self.assertGreaterEqual(POSE_RENDER_INTERVAL_MS, 100)

    def test_tool_window_layout_is_resolved_before_showing(self):
        class Window:
            def __init__(self):
                self.calls = []

            def update_idletasks(self):
                self.calls.append("layout")

            def deiconify(self):
                self.calls.append("show")

            def lift(self):
                self.calls.append("raise")

            def focus_set(self):
                self.calls.append("focus")

        window = Window()
        ROVGui._show_tool_window(window)

        self.assertEqual(window.calls, ["layout", "show", "raise", "focus"])

    def test_pi_restart_forces_fresh_parameter_readback(self):
        gui = ROVGui.__new__(ROVGui)
        gui._last_pi_telemetry_session = "old-session"
        gui._pid_synced_from_pi = True
        gui._motion_tuning_synced_from_pi = True
        gui._attitude_tuning_synced_from_pi = True

        self.assertTrue(gui._reset_pi_sync_on_session_change({
            "_rx_marker": ("new-session", 0),
        }))
        self.assertFalse(gui._pid_synced_from_pi)
        self.assertFalse(gui._motion_tuning_synced_from_pi)
        self.assertFalse(gui._attitude_tuning_synced_from_pi)

        gui._pid_synced_from_pi = True
        self.assertFalse(gui._reset_pi_sync_on_session_change({
            "_rx_marker": ("new-session", 1),
        }))
        self.assertTrue(gui._pid_synced_from_pi)

    def test_main_window_fits_inside_current_screen_instead_of_clipping_right_column(self):
        self.assertEqual(fit_main_window_geometry(1920, 1080), (1560, 900))
        self.assertEqual(fit_main_window_geometry(1708, 1069), (1560, 900))
        width, height = fit_main_window_geometry(1366, 768)
        self.assertLessEqual(width, 1366)
        self.assertLessEqual(height, 768)

    def test_detailed_pages_do_not_compress_main_dashboard(self):
        constructor = inspect.getsource(ROVGui.__init__)
        self.assertIn("self._build_settings_windows()", constructor)
        self.assertIn("self._build_main_action_bar(self._content)", constructor)
        self.assertNotIn("self._build_command_panel(self._content)", constructor)
        self.assertNotIn("self._build_joystick_panel(self._right_column)", constructor)

    def test_main_bar_keeps_safety_controls_visible(self):
        source = inspect.getsource(ROVGui._build_main_action_bar)
        self.assertIn('text="⚙ 自主控制"', source)
        self.assertNotIn("自主控制与参数", source)
        for control in (
            "self._arm_btn",
            "self._estop_btn",
            "self._start_btn",
            "self._hover_btn",
            "self._safety_detail",
        ):
            self.assertIn(control, source)

    def test_main_bar_repeats_pi_authoritative_attitude_hold_status(self):
        main_source = inspect.getsource(ROVGui._build_main_action_bar)
        command_source = inspect.getsource(ROVGui._build_command_panel)
        for quick_status in (
            "self._pitch_hold_quick_light",
            "self._roll_hold_quick_light",
            "self._yaw_hold_quick_light",
        ):
            self.assertIn(quick_status, main_source)
            self.assertNotIn(quick_status, command_source)
        self.assertIn('("pitch", "俯仰自动", "十字键↑")', command_source)
        self.assertIn('("roll", "横滚自动", "十字键→")', command_source)
        self.assertIn('("yaw", "偏转自动", "十字键←")', command_source)
        self.assertIn('setattr(self, f"_{axis}_hold_btn", button)', command_source)
        self.assertIn("width=20", command_source)

    def test_main_bar_owns_quick_depth_and_servo_controls(self):
        main_source = inspect.getsource(ROVGui._build_main_action_bar)
        command_source = inspect.getsource(ROVGui._build_command_panel)
        for label in (
            "深度保持", "目标深度 m", "取当前值", "当前位置置零", "舵机 CH8",
        ):
            self.assertIn(label, main_source)
            self.assertNotIn(label, command_source)
        self.assertIn("self._main_yaw_target_var", main_source)
        self.assertIn('value="目标Yaw：--°"', main_source)
        self.assertLess(
            main_source.index("self._main_yaw_target_var"),
            main_source.index("servo_controls ="),
        )
        self.assertIn("_gripper_open_angle_var", main_source)
        self.assertIn("_gripper_closed_angle_var", main_source)
        self.assertIn("_apply_gripper_positions", main_source)
        self.assertNotIn("ttk.Scale", main_source)
        self.assertNotIn("_on_servo_slider", main_source)

    def test_autonomy_rows_remove_live_yaw_labels_and_align_pid_columns(self):
        source = inspect.getsource(ROVGui._build_command_panel)

        self.assertNotIn("实时修正 Yaw", source)
        self.assertNotIn("锁定目标 Yaw", source)
        self.assertIn("row_attitude.grid", source)
        self.assertIn('text="松手锁定≤"', source)
        self.assertIn("self._yaw_release_rate_threshold_var", source)

    def test_center_is_wider_and_right_status_column_is_compact(self):
        constructor = inspect.getsource(ROVGui.__init__)
        self.assertIn("grid_columnconfigure(0, weight=3, minsize=210)", constructor)
        self.assertIn("grid_columnconfigure(1, weight=10, minsize=520)", constructor)
        self.assertIn("grid_columnconfigure(2, weight=1, minsize=280)", constructor)

    def test_wide_button_groups_wrap_instead_of_forcing_dashboard_off_screen(self):
        timer_source = inspect.getsource(CompetitionTimer.__init__)
        telemetry_source = inspect.getsource(ROVGui._build_telemetry_panel)
        self.assertIn("row=0, column=0", timer_source)
        self.assertIn("row=1, column=0", timer_source)
        self.assertIn("row=0, column=0", telemetry_source)
        self.assertIn("row=1, column=0", telemetry_source)
        self.assertIn("wraplength=300", telemetry_source)

    def test_left_column_has_depth_and_all_attitude_trends(self):
        build_source = inspect.getsource(ROVGui._build_left_panels)
        refresh_source = inspect.getsource(ROVGui._refresh_display_once)
        for chart in (
            "_depth_trend",
            "_roll_trend",
            "_pitch_trend",
            "_yaw_trend",
            "_heading_trend",
        ):
            self.assertIn(chart, build_source)
            self.assertIn(chart, refresh_source)

    def test_trends_use_fixed_absolute_physical_ranges(self):
        source = inspect.getsource(ROVGui._build_left_panels)
        for bounds in (
            "minimum=self._depth_target_min_m, maximum=self._depth_target_max_m",
            "minimum=-180.0, maximum=180.0",
            "minimum=-90.0, maximum=90.0",
            "minimum=0.0, maximum=360.0",
        ):
            self.assertIn(bounds, source)
        self.assertEqual(
            source.count("minimum=-180.0, maximum=180.0"), 2
        )

    def test_realtime_telemetry_sits_below_compact_thruster_card(self):
        constructor = inspect.getsource(ROVGui.__init__)
        thruster_call = "self._build_thruster_panel(self._right_column)"
        telemetry_call = "self._build_telemetry_panel(self._right_column)"
        self.assertIn(telemetry_call, constructor)
        self.assertLess(constructor.index(thruster_call), constructor.index(telemetry_call))
        thruster_source = inspect.getsource(ROVGui._build_thruster_panel)
        self.assertIn("frame.pack(fill=tk.X", thruster_source)
        self.assertIn("display_size=190", thruster_source)
        self.assertNotIn("frame.pack(fill=tk.BOTH, expand=True)", thruster_source)

    def test_right_telemetry_replaces_angular_environment_metrics_with_log(self):
        constructor = inspect.getsource(ROVGui.__init__)
        telemetry_source = inspect.getsource(ROVGui._build_telemetry_panel)
        refresh_source = inspect.getsource(ROVGui._refresh_display_once)
        self.assertIn("self._build_event_log(frame)", telemetry_source)
        self.assertNotIn("角速度与环境", telemetry_source)
        self.assertNotIn("self._build_event_log(self._content)", constructor)
        for obsolete in ("_lbl_gx", "_lbl_gy", "_lbl_gz", "_lbl_temp"):
            self.assertNotIn(obsolete, refresh_source)

    def test_event_log_uses_chinese_safe_font_and_keeps_a_compact_readable_height(self):
        source = inspect.getsource(ROVGui._build_event_log)
        self.assertIn('font=("Microsoft YaHei UI", 9)', source)
        self.assertIn("text_frame, width=1, height=3", source)
        self.assertIn("spacing1=0", source)
        self.assertIn("spacing3=1", source)
        self.assertIn("frame.pack(fill=tk.BOTH, expand=True", source)
        self.assertIn("text_frame.pack(fill=tk.BOTH, expand=True)", source)

    def test_mount_instructions_stay_single_line_to_reserve_log_space(self):
        source = inspect.getsource(ROVGui._build_telemetry_panel)
        self.assertIn("水平→上仰20–45°→水平右偏20–45°", source)
        self.assertIn('font=("Microsoft YaHei UI", 7)', source)

    def test_expandable_canvases_do_not_force_their_tk_default_width(self):
        source = inspect.getsource(TrendChart.__init__)
        self.assertIn("self, width=1, bg=UI[\"surface_alt\"], height=48", source)

    def test_tool_windows_are_content_sized(self):
        source = inspect.getsource(ROVGui._build_settings_windows)
        self.assertNotIn("self._build_telemetry_panel(telemetry_column)", source)
        self.assertIn('self._autonomy_window.geometry("1220x640")', source)
        self.assertIn('self._autonomy_window.minsize(1080, 640)', source)
        self.assertIn('self._joystick_window.geometry("960x135")', source)

    def test_clicking_inside_tool_window_does_not_focus_main_window(self):
        source = inspect.getsource(ROVGui._clear_text_focus_on_background_click)
        self.assertIn("winfo_toplevel()", source)
        self.assertIn("target.focus_set()", source)

    def test_front_rear_labels_are_short_and_direction_specific(self):
        source = inspect.getsource(ROVGui._build_command_panel)
        for label in ("上升差%", "下降差%", "前进差%", "后退差%"):
            self.assertIn(f'"{label}"', source)
        self.assertNotIn("\u5de6\u53f3\u524d\u540e\u5dee", source)
        self.assertNotIn("\u524d\u8fdb\u5782\u63a8前后差", source)

    def test_rejected_attitude_limit_edit_keeps_input_for_retry(self):
        source = inspect.getsource(ROVGui._apply_attitude_tuning_from_entry)
        self.assertIn("self._attitude_tuning_dirty = True", source)
        self.assertNotIn("self._restore_attitude_tuning_entries()", source)
        self.assertIn("self._log_event", source)


class TrendChartFixedRangeTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self):
        self.root.destroy()

    def test_chart_stores_an_explicit_fixed_range(self):
        chart = TrendChart(
            self.root, "Yaw", "°", "#123456", minimum=-180.0, maximum=180.0,
        )
        self.assertEqual(chart._value_min, -180.0)
        self.assertEqual(chart._value_max, 180.0)

    def test_invalid_fixed_range_is_rejected(self):
        with self.assertRaises(ValueError):
            TrendChart(
                self.root, "bad", "", "#123456", minimum=1.0, maximum=1.0,
            )

    def test_redraw_uses_one_polyline_for_all_samples(self):
        class Canvas:
            def __init__(self):
                self.line_calls = []

            def winfo_width(self):
                return 300

            def winfo_height(self):
                return 80

            def delete(self, _tag):
                pass

            def create_polygon(self, *_args, **_kwargs):
                pass

            def create_line(self, *args, **kwargs):
                self.line_calls.append((args, kwargs))

            def create_text(self, *_args, **_kwargs):
                pass

        chart = TrendChart.__new__(TrendChart)
        chart.canvas = Canvas()
        chart._window = 30.0
        chart._value_min = -180.0
        chart._value_max = 180.0
        chart._fill_color = "#eeeeee"
        chart._color = "#123456"
        now = time.monotonic()
        chart._data = [
            (now - 5.0 + index * 0.05, float(index % 20))
            for index in range(100)
        ]

        chart.redraw()

        self.assertEqual(len(chart.canvas.line_calls), 1)
        points, options = chart.canvas.line_calls[0]
        self.assertEqual(len(points), 200)
        self.assertEqual(options["fill"], "#123456")


if __name__ == "__main__":
    unittest.main()
