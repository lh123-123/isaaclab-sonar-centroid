#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest import mock

import tkinter as tk

import pc_gui
from pc_gui import ROVGui, UI, validate_attitude_calibration_snapshot


class _Button:
    def __init__(self):
        self.options = {}

    def configure(self, **options):
        self.options.update(options)


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def _gui_without_tk_window():
    gui = ROVGui.__new__(ROVGui)
    gui._roll_hold_btn = _Button()
    gui._pitch_hold_btn = _Button()
    gui._yaw_hold_btn = _Button()
    gui._roll_hold_quick_light = _Button()
    gui._pitch_hold_quick_light = _Button()
    gui._yaw_hold_quick_light = _Button()
    gui._attitude_mapping_btn = _Button()
    gui._attitude_status_var = _Var()
    gui._main_yaw_target_var = _Var()
    gui._last_attitude_hold_rejection_ack = None
    gui._last_attitude_calibration_sync = None
    gui._attitude_tuning = {
        "roll_max_moment_nm": 3.0,
        "pitch_max_moment_nm": 4.0,
        "yaw_max_moment_nm": 4.0,
        "yaw_release_rate_threshold_dps": 3.0,
    }
    gui._attitude_tuning_synced_from_pi = True
    gui._attitude_tuning_dirty = False
    gui._roll_moment_limit_var = _Var("3")
    gui._pitch_moment_limit_var = _Var("4")
    gui._yaw_moment_limit_var = _Var("4")
    gui._yaw_release_rate_threshold_var = _Var("3")
    gui._performance_mode = "balance"
    gui._performance_profiles = {
        "balance": {
            "roll_max_moment_nm": 4.0,
            "pitch_max_moment_nm": 3.0,
            "yaw_max_moment_nm": 4.0,
        },
        "rage": {
            "roll_max_moment_nm": 4.0,
            "pitch_max_moment_nm": 3.5,
            "yaw_max_moment_nm": 4.0,
        },
    }
    gui.events = []
    gui._log_event = gui.events.append
    return gui


class AttitudeHoldGuiStateTests(unittest.TestCase):
    @staticmethod
    def _calibration_snapshot(mapping=False):
        return {
            "imu_3d_mount_calibrated": True,
            "mapping_validated": mapping,
            "imu_mount_quat_w": 1.0, "imu_mount_quat_x": 0.0,
            "imu_mount_quat_y": 0.0, "imu_mount_quat_z": 0.0,
            "imu_reference_quat_w": 0.70710678,
            "imu_reference_quat_x": 0.0,
            "imu_reference_quat_y": 0.0,
            "imu_reference_quat_z": 0.70710678,
        }

    def test_mapping_button_follows_pi_not_local_state(self):
        gui = _gui_without_tk_window()
        gui._refresh_attitude_hold_state({
            "attitude_hold": {
                "roll": False, "pitch": False,
                "mapping_validated": True,
                "reason": "operator_disabled",
            }
        })
        self.assertIn("已验收", gui._attitude_mapping_btn.options["text"])
        self.assertEqual(gui._attitude_mapping_btn.options["state"], tk.NORMAL)

        gui._refresh_attitude_hold_state(None)
        self.assertIn("状态未知", gui._attitude_mapping_btn.options["text"])
        self.assertEqual(gui._attitude_mapping_btn.options["state"], tk.DISABLED)

    def test_attitude_tuning_wait_ignores_old_rejection_until_ack_changes(self):
        gui = _gui_without_tk_window()
        old_ack = {
            "ok": False,
            "message": "command_rejected",
            "error": "unrelated_old_error",
            "timestamp": 1.0,
        }
        requested = {
            "roll_max_moment_nm": 3.0,
            "pitch_max_moment_nm": 3.0,
            "yaw_max_moment_nm": 3.0,
            "yaw_release_rate_threshold_dps": 3.0,
        }
        packets = iter((
            {
                "_rx_marker": ("session", 11),
                "command_ack": dict(old_ack),
                "attitude_tuning": dict(gui._attitude_tuning),
            },
            {
                "_rx_marker": ("session", 12),
                "command_ack": {
                    "ok": True,
                    "message": "attitude_tuning_saved",
                    "timestamp": 2.0,
                    "attitude_tuning": dict(requested),
                },
                "attitude_tuning": dict(requested),
            },
        ))

        class _Client:
            def get_telemetry(self):
                return next(packets)

        gui.client = _Client()
        actual = gui._wait_for_attitude_tuning_ack(
            requested,
            ("session", 10),
            baseline_ack=old_ack,
            timeout=0.1,
        )

        self.assertEqual(actual, requested)

    def test_rejected_attitude_tuning_keeps_user_input_for_retry(self):
        gui = _gui_without_tk_window()
        gui._roll_moment_limit_var.set("3")
        gui._pitch_moment_limit_var.set("3")
        gui._yaw_moment_limit_var.set("3")
        gui._yaw_release_rate_threshold_var.set("3")
        status = []
        gui._set_parameter_status = status.append

        class _Client:
            def get_telemetry(self):
                return {"armed": True, "cmd_enabled": False}

        gui.client = _Client()
        gui._apply_attitude_tuning_from_entry()

        self.assertEqual(gui._roll_moment_limit_var.get(), "3")
        self.assertEqual(gui._pitch_moment_limit_var.get(), "3")
        self.assertEqual(gui._yaw_moment_limit_var.get(), "3")
        self.assertIn("请先锁定推进器", status[-1])

    @mock.patch("pc_gui.update_config_values")
    def test_attitude_tuning_submit_uses_active_profile_moment_limits(
        self, _update_config
    ):
        gui = _gui_without_tk_window()
        gui._roll_moment_limit_var.set("9")
        gui._pitch_moment_limit_var.set("9")
        gui._yaw_moment_limit_var.set("9")
        gui._yaw_release_rate_threshold_var.set("4")
        gui._set_parameter_status = mock.Mock()
        gui._finish_parameter_entry = mock.Mock()

        class _Client:
            sent = []

            @staticmethod
            def get_telemetry():
                return {
                    "armed": False,
                    "cmd_enabled": False,
                    "performance_mode": "balance",
                    "command_ack": {},
                }

            @staticmethod
            def get_telemetry_marker():
                return ("session", 1)

            def send_cmd(self, **payload):
                self.sent.append(payload)

        gui.client = _Client()
        gui._wait_for_attitude_tuning_ack = (
            lambda requested, *_args, **_kwargs: dict(requested)
        )

        self.assertTrue(gui._apply_attitude_tuning_from_entry())
        self.assertEqual(gui.client.sent[-1]["attitude_tuning"], {
            "roll_max_moment_nm": 4.0,
            "pitch_max_moment_nm": 3.0,
            "yaw_max_moment_nm": 4.0,
            "yaw_release_rate_threshold_dps": 4.0,
        })

    def test_attitude_tuning_sync_does_not_overwrite_dirty_input(self):
        gui = _gui_without_tk_window()
        gui._attitude_tuning_synced_from_pi = False
        gui._attitude_tuning_dirty = True
        gui._roll_moment_limit_var.set("3")
        gui._pitch_moment_limit_var.set("3")
        gui._yaw_moment_limit_var.set("3")
        gui._yaw_release_rate_threshold_var.set("3")

        gui._sync_attitude_tuning_from_pi({
            "attitude_tuning": {
                "roll_max_moment_nm": 4.0,
                "pitch_max_moment_nm": 4.0,
                "yaw_max_moment_nm": 4.0,
                "yaw_release_rate_threshold_dps": 4.0,
            }
        })

        self.assertEqual(gui._roll_moment_limit_var.get(), "3")
        self.assertEqual(gui._pitch_moment_limit_var.get(), "3")
        self.assertEqual(gui._yaw_moment_limit_var.get(), "3")
        self.assertEqual(gui._yaw_release_rate_threshold_var.get(), "3")

    def test_corrected_yaw_and_magnetic_heading_remain_separate_pi_values(self):
        gui = _gui_without_tk_window()
        gui._imu_heading_inverted = True
        gui._imu_heading_offset_deg = 0.0

        actual = gui._vehicle_imu({
            "roll": -4.0,
            "pitch": 7.0,
            "yaw": 91.0,
            "yaw_control": -32.0,
            "heading": 12.0,
            "heading_vehicle": 287.5,
        })

        self.assertEqual(actual["yaw"], -32.0)
        self.assertEqual(actual["heading"], 287.5)

    def test_calibration_snapshot_accepts_only_normalized_finite_quaternions(self):
        good = self._calibration_snapshot()
        sections = validate_attitude_calibration_snapshot(good)
        self.assertEqual(sections["hardware"]["imu_3d_mount_calibrated"], "true")
        self.assertEqual(
            sections["control"]["attitude_control_mapping_validated"], "false"
        )
        for field, invalid in (
            ("imu_mount_quat_w", float("nan")),
            ("imu_reference_quat_w", 0.2),
            ("mapping_validated", 1),
        ):
            with self.subTest(field=field):
                bad = dict(good)
                bad[field] = invalid
                with self.assertRaises(ValueError):
                    validate_attitude_calibration_snapshot(bad)

    def test_valid_pi_calibration_is_written_locally_once(self):
        gui = _gui_without_tk_window()

        class _Config:
            config_path = "local-config.ini"

            def __init__(self):
                self.values = {}

            def get(self, section, key, fallback=None):
                return self.values.get((section, key), fallback)

            def set(self, section, key, value):
                self.values[(section, key)] = value

        local_config = _Config()
        telemetry = {
            "attitude_calibration_config": self._calibration_snapshot(True),
        }
        with (
            mock.patch.object(pc_gui, "APP_CONFIG", local_config),
            mock.patch.object(pc_gui, "update_config_sections") as update,
        ):
            self.assertTrue(gui._sync_attitude_calibration_from_pi(telemetry))
            self.assertFalse(gui._sync_attitude_calibration_from_pi(telemetry))

        update.assert_called_once()
        self.assertEqual(update.call_args.args[0], "local-config.ini")
        self.assertEqual(
            local_config.get("control", "attitude_control_mapping_validated"),
            "true",
        )
    def test_fresh_pi_telemetry_is_the_only_source_of_button_state(self):
        gui = _gui_without_tk_window()

        gui._refresh_attitude_hold_state({
            "attitude_hold": {
                "roll": True,
                "pitch": False,
                "yaw": True,
                "yaw_control_deg": -31.25,
                "yaw_target_deg": -35.5,
                "yaw_control_source": "referenced_euler_yaw",
                "reason": "operator_enabled",
            }
        })

        self.assertIn("横滚自动：开", gui._roll_hold_btn.options["text"])
        self.assertEqual(gui._roll_hold_btn.options["bg"], "#FFFFFF")
        self.assertEqual(gui._roll_hold_btn.options["fg"], "#000000")
        self.assertEqual(gui._roll_hold_btn.options["state"], tk.NORMAL)
        self.assertIn("俯仰自动：关", gui._pitch_hold_btn.options["text"])
        self.assertIn("偏转自动：开", gui._yaw_hold_btn.options["text"])
        self.assertEqual(gui._pitch_hold_btn.options["bg"], "#FFFFFF")
        self.assertEqual(gui._roll_hold_quick_light.options["text"], "● 横滚锁：开")
        self.assertEqual(gui._pitch_hold_quick_light.options["text"], "● 俯仰锁：关")
        self.assertEqual(gui._yaw_hold_quick_light.options["text"], "● 偏转锁：开")
        self.assertEqual(gui._roll_hold_quick_light.options["foreground"], UI["green"])
        self.assertEqual(gui._pitch_hold_quick_light.options["foreground"], UI["muted"])
        self.assertEqual(gui._yaw_hold_quick_light.options["foreground"], UI["green"])
        self.assertEqual(gui._main_yaw_target_var.value, "目标Yaw：-35.5°")
        self.assertIn("Pi 原因：操作员已开启", gui._attitude_status_var.value)

    def test_telemetry_loss_clears_stale_green_state_to_unknown(self):
        gui = _gui_without_tk_window()
        gui._refresh_attitude_hold_state({
            "attitude_hold": {
                "roll": True,
                "pitch": True,
                "yaw": True,
                "reason": "operator_enabled",
            }
        })

        gui._refresh_attitude_hold_state(None)

        for button in (gui._roll_hold_btn, gui._pitch_hold_btn, gui._yaw_hold_btn):
            self.assertIn("状态未知", button.options["text"])
            self.assertEqual(button.options["bg"], "#FFFFFF")
            self.assertEqual(button.options["disabledforeground"], "#000000")
            self.assertEqual(button.options["state"], tk.DISABLED)
        for light, axis in (
            (gui._roll_hold_quick_light, "横滚"),
            (gui._pitch_hold_quick_light, "俯仰"),
            (gui._yaw_hold_quick_light, "偏转"),
        ):
            self.assertEqual(light.options["text"], f"● {axis}锁：未知")
            self.assertEqual(light.options["foreground"], UI["muted"])
        self.assertIn("遥测中断", gui._attitude_status_var.value)
        self.assertEqual(gui._main_yaw_target_var.value, "目标Yaw：--°")
        self.assertNotIn("已开启", gui._attitude_status_var.value)

    def test_yaw_release_settling_is_shown_as_inertia_wait(self):
        gui = _gui_without_tk_window()

        gui._refresh_attitude_hold_state({
            "attitude_hold": {
                "roll": False,
                "pitch": False,
                "yaw": True,
                "yaw_control_deg": 40.0,
                "yaw_target_deg": 38.0,
                "yaw_control_source": "referenced_euler_yaw",
                "yaw_release_settling": True,
                "reason": "operator_enabled",
            }
        })

        self.assertIn("惯性等待", gui._yaw_hold_btn.options["text"])
        self.assertIn("惯性等待", gui._yaw_hold_quick_light.options["text"])
        self.assertEqual(gui._yaw_hold_quick_light.options["foreground"], UI["amber"])
        self.assertIn("偏转 惯性等待", gui._attitude_status_var.value)

    def test_pi_fault_reason_replaces_previous_enabled_wording(self):
        gui = _gui_without_tk_window()

        gui._refresh_attitude_hold_state({
            "attitude_hold": {
                "roll": False,
                "pitch": False,
                "reason": "imu_invalid",
            }
        })

        self.assertIn("IMU 无效，Pi 已关闭", gui._attitude_status_var.value)
        self.assertNotIn("操作员已开启", gui._attitude_status_var.value)

    def test_manual_override_is_not_painted_as_active_pid_control(self):
        gui = _gui_without_tk_window()

        gui._refresh_attitude_hold_state({
            "attitude_hold": {
                "roll": True,
                "pitch": True,
                "roll_manual_override": False,
                "pitch_manual_override": True,
                "reason": "operator_enabled",
            }
        })

        self.assertIn("横滚自动：开", gui._roll_hold_btn.options["text"])
        self.assertEqual(gui._roll_hold_btn.options["bg"], "#FFFFFF")
        self.assertIn("俯仰自动：人工接管", gui._pitch_hold_btn.options["text"])
        self.assertEqual(gui._pitch_hold_btn.options["bg"], "#FFFFFF")
        self.assertEqual(
            gui._pitch_hold_quick_light.options["text"], "● 俯仰锁：人工接管"
        )
        self.assertEqual(gui._pitch_hold_quick_light.options["foreground"], UI["amber"])
        self.assertIn("俯仰 人工接管", gui._attitude_status_var.value)

    def test_unvalidated_mapping_is_an_explicit_autonomy_gate(self):
        gui = _gui_without_tk_window()

        gui._refresh_attitude_hold_state({
            "attitude_hold": {
                "roll": False,
                "pitch": False,
                "mapping_validated": False,
                "reason": "startup_disabled",
            }
        })

        self.assertIn("姿态推进器/IMU 方向映射未验收", gui._attitude_status_var.value)

    def test_pi_hold_rejection_error_is_shown_and_logged_once(self):
        gui = _gui_without_tk_window()
        telemetry = {
            "attitude_hold": {
                "roll": False,
                "pitch": False,
                "mapping_validated": False,
                "reason": "startup_disabled",
            },
            "command_ack": {
                "ok": False,
                "message": "command_rejected",
                "error": "attitude_hold_rejected_mapping_unvalidated",
                "timestamp": 123.0,
            },
        }

        gui._refresh_attitude_hold_state(telemetry)
        gui._refresh_attitude_hold_state(telemetry)

        self.assertIn("姿态锁定被 Pi 拒绝", gui._attitude_status_var.value)
        self.assertIn("姿态推进器/IMU 方向映射未验收", gui._attitude_status_var.value)
        self.assertEqual(len(gui.events), 1)
        self.assertIn("映射未验收", gui.events[0])

    def test_toggle_preflight_reports_mapping_gate_without_sending(self):
        gui = _gui_without_tk_window()

        class _Client:
            def __init__(self):
                self.sent = []

            def get_telemetry(self):
                return {
                    "armed": True,
                    "control_link_ok": True,
                    "emergency_stop": False,
                    "sensor_health": {"imu_valid": True},
                    "attitude_3d_mount_calibrated": True,
                    "attitude_hold": {
                        "roll": False,
                        "pitch": False,
                        "mapping_validated": False,
                    },
                }

            def send_cmd(self, **command):
                self.sent.append(command)

        gui.client = _Client()
        gui._toggle_attitude_hold("roll")

        self.assertEqual(gui.client.sent, [])
        self.assertIn("姿态推进器/IMU 方向映射未验收", gui._attitude_status_var.value)
        self.assertIn("横滚锁定未开启", gui.events[-1])

    def test_yaw_toggle_sends_all_three_independent_hold_switches(self):
        gui = _gui_without_tk_window()

        class _Client:
            def __init__(self):
                self.sent = []

            def get_telemetry(self):
                return {
                    "armed": True,
                    "control_link_ok": True,
                    "emergency_stop": False,
                    "sensor_health": {"imu_valid": True},
                    "attitude_3d_mount_calibrated": True,
                    "attitude_hold": {
                        "roll": True,
                        "pitch": False,
                        "yaw": False,
                        "mapping_validated": True,
                    },
                    "command_ack": {},
                }

            def get_telemetry_marker(self):
                return ("session", 1)

            def send_cmd(self, **command):
                self.sent.append(command)

        gui.client = _Client()
        gui._set_parameter_status = lambda _message: None
        gui._wait_for_attitude_hold_ack = (
            lambda requested, *_args, **_kwargs: dict(requested)
        )

        gui._toggle_attitude_hold("yaw")

        self.assertEqual(gui.client.sent, [{
            "attitude_hold": {"roll": True, "pitch": False, "yaw": True}
        }])
        self.assertIn("偏转自动调整已开", gui.events[-1])

    def test_wait_ignores_a_repeated_old_rejection_until_ack_changes(self):
        gui = _gui_without_tk_window()
        old_ack = {
            "ok": False,
            "message": "command_rejected",
            "error": "unrelated_old_error",
            "timestamp": 1.0,
        }
        packets = iter((
            {
                "_rx_marker": ("session", 11),
                "command_ack": dict(old_ack),
                "attitude_hold": {"roll": False, "pitch": False},
            },
            {
                "_rx_marker": ("session", 12),
                "command_ack": {
                    "ok": True,
                    "message": "attitude_hold_updated",
                    "timestamp": 2.0,
                },
                "attitude_hold": {"roll": True, "pitch": False},
            },
        ))

        class _Client:
            def get_telemetry(self):
                return next(packets)

        gui.client = _Client()
        actual = gui._wait_for_attitude_hold_ack(
            {"roll": True, "pitch": False},
            ("session", 10),
            baseline_ack=old_ack,
            timeout=0.1,
        )
        self.assertTrue(actual["roll"])


if __name__ == "__main__":
    unittest.main()
