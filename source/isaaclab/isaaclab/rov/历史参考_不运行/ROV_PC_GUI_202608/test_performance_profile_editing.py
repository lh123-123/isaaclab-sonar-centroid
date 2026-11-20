import unittest
from unittest import mock

from pc_gui import ROVGui, validate_performance_profile


PROFILE_SCALARS = {
    "hover": "hover_force_per_thruster_n",
    "ascent_trim": "vertical_ascent_front_rear_trim_percent",
    "descent_trim": "vertical_descent_front_rear_trim_percent",
    "forward_trim": "forward_front_rear_trim_percent",
    "backward_trim": "backward_front_rear_trim_percent",
    "left_lateral": "left_lateral_surge_trim_percent",
    "right_lateral": "right_lateral_surge_trim_percent",
    "roll_moment": "roll_max_moment_nm",
    "pitch_moment": "pitch_max_moment_nm",
    "yaw_moment": "yaw_max_moment_nm",
}


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Status(_Var):
    pass


def _profile(value, *, mode="balance"):
    return {
        "mode": mode,
        "horizontal_limit_fraction": value,
        "vertical_limit_fraction": value,
        "hover_force_per_thruster_n": min(value, 0.5),
        "vertical_ascent_front_rear_trim_percent": value,
        "vertical_descent_front_rear_trim_percent": -value,
        "forward_front_rear_trim_percent": value,
        "backward_front_rear_trim_percent": -value,
        "left_lateral_surge_trim_percent": -value,
        "right_lateral_surge_trim_percent": value,
        "roll_max_moment_nm": max(value, 0.1),
        "pitch_max_moment_nm": max(value, 0.1),
        "yaw_max_moment_nm": max(value, 0.1),
        "roll_pid": {"kp": value, "ki": value, "kd": value},
        "pitch_pid": {"kp": value, "ki": value, "kd": value},
        "yaw_pid": {"kp": value, "ki": value, "kd": value},
    }


def _entries(value):
    result = {
        "horizontal": _Var(value),
        "vertical": _Var(value),
    }
    result.update({key: _Var(value) for key in PROFILE_SCALARS})
    for axis in ("roll", "pitch", "yaw"):
        for field in ("kp", "ki", "kd"):
            result[f"{axis}_{field}"] = _Var(value)
    return result


class _Client:
    def __init__(self):
        self.sent = []

    def get_telemetry_marker(self):
        return 10

    def send_cmd(self, **payload):
        self.sent.append(payload)


class PerformanceProfileEditingTests(unittest.TestCase):
    def make_gui(self):
        gui = ROVGui.__new__(ROVGui)
        gui._performance_profiles = {
            "balance": _profile(0.2, mode="balance"),
            "rage": _profile(0.3, mode="rage"),
        }
        gui._performance_profile_entries = {
            "balance": _entries("0"),
            "rage": _entries("0"),
        }
        gui._performance_profile_dirty = {"balance": False, "rage": False}
        gui._performance_status_var = _Status("")
        gui._parameter_status_var = _Status("")
        gui.client = _Client()
        return gui

    def test_complete_profile_validation_includes_mode_owned_motion_and_moments(self):
        profile = validate_performance_profile(
            "rage",
            "40",
            "30",
            {
                "roll": ("0.32", "0.02", "0.11"),
                "pitch": ("0.38", "0.015", "0.16"),
                "yaw": ("0.27", "0.01", "0.11"),
            },
            {
                "hover_force_per_thruster_n": "0",
                "vertical_ascent_front_rear_trim_percent": "0",
                "vertical_descent_front_rear_trim_percent": "0",
                "forward_front_rear_trim_percent": "8",
                "backward_front_rear_trim_percent": "-8",
                "left_lateral_surge_trim_percent": "-5",
                "right_lateral_surge_trim_percent": "-6",
            },
            {
                "roll_max_moment_nm": "4",
                "pitch_max_moment_nm": "3.5",
                "yaw_max_moment_nm": "4",
            },
        )

        self.assertEqual(profile["horizontal_limit_fraction"], 0.4)
        self.assertEqual(profile["vertical_limit_fraction"], 0.3)
        self.assertEqual(profile["forward_front_rear_trim_percent"], 8.0)
        self.assertEqual(profile["backward_front_rear_trim_percent"], -8.0)
        self.assertEqual(profile["left_lateral_surge_trim_percent"], -5.0)
        self.assertEqual(profile["right_lateral_surge_trim_percent"], -6.0)
        self.assertEqual(profile["pitch_max_moment_nm"], 3.5)

    def test_pi_refresh_does_not_overwrite_profile_being_edited(self):
        gui = self.make_gui()
        gui._performance_profile_entries["balance"]["forward_trim"].set("typed")
        gui._performance_profile_dirty["balance"] = True

        remote_balance = _profile(0.25, mode="balance")
        remote_rage = _profile(0.4, mode="rage")
        gui._sync_performance_profiles_from_pi(
            {"balance": remote_balance, "rage": remote_rage}
        )

        self.assertEqual(
            gui._performance_profile_entries["balance"]["forward_trim"].get(),
            "typed",
        )
        self.assertEqual(
            gui._performance_profile_entries["rage"]["forward_trim"].get(),
            "0.4",
        )
        self.assertEqual(gui._performance_profiles["balance"], remote_balance)

    def test_lateral_surge_trim_rejects_unsafe_entry(self):
        with self.assertRaises(ValueError):
            validate_performance_profile(
                "balance",
                "20",
                "20",
                {
                    "roll": ("0.25", "0.02", "0.09"),
                    "pitch": ("0.3", "0.02", "0.14"),
                    "yaw": ("0.21", "0.01", "0.09"),
                },
                {"left_lateral_surge_trim_percent": "100.1"},
            )

    def test_set_entries_updates_every_mode_owned_field(self):
        gui = self.make_gui()
        profile = _profile(0.25, mode="balance")

        gui._set_performance_profile_entries("balance", profile)

        entries = gui._performance_profile_entries["balance"]
        self.assertEqual(entries["horizontal"].get(), "25.0")
        self.assertEqual(entries["hover"].get(), "0.25")
        self.assertEqual(entries["descent_trim"].get(), "-0.25")
        self.assertEqual(entries["left_lateral"].get(), "-0.25")
        self.assertEqual(entries["right_lateral"].get(), "0.25")
        self.assertEqual(entries["pitch_moment"].get(), "0.25")
        self.assertEqual(entries["yaw_kd"].get(), "0.25")

    @mock.patch("pc_gui.update_config_sections")
    def test_enter_applies_and_persists_complete_profile(self, update_sections):
        gui = self.make_gui()
        entries = gui._performance_profile_entries["rage"]
        entries["horizontal"].set("40")
        entries["vertical"].set("30")
        entries["hover"].set("0")
        entries["ascent_trim"].set("0")
        entries["descent_trim"].set("0")
        entries["forward_trim"].set("8")
        entries["backward_trim"].set("-8")
        entries["left_lateral"].set("-5")
        entries["right_lateral"].set("-6")
        entries["roll_moment"].set("4")
        entries["pitch_moment"].set("3.5")
        entries["yaw_moment"].set("4")
        for axis in ("roll", "pitch", "yaw"):
            for field in ("kp", "ki", "kd"):
                entries[f"{axis}_{field}"].set("0.1")
        gui._wait_for_performance_profile_ack = lambda requested, _marker: requested

        result = gui._apply_performance_profile_from_entries("rage")

        self.assertTrue(result)
        sent = gui.client.sent[-1]["performance_profile"]
        self.assertEqual(sent["forward_front_rear_trim_percent"], 8.0)
        self.assertEqual(sent["pitch_max_moment_nm"], 3.5)
        self.assertEqual(sent["left_lateral_surge_trim_percent"], -5.0)
        self.assertEqual(sent["right_lateral_surge_trim_percent"], -6.0)
        saved = update_sections.call_args.args[1]["mode_rage"]
        self.assertEqual(saved["backward_front_rear_trim_percent"], -8.0)
        self.assertEqual(saved["left_lateral_surge_trim_percent"], -5.0)
        self.assertEqual(saved["yaw_max_moment_nm"], 4.0)

    def test_escape_restores_authoritative_values_and_releases_edit_lock(self):
        gui = self.make_gui()
        gui._performance_profile_entries["balance"]["hover"].set("typed")
        gui._performance_profile_dirty["balance"] = True

        result = gui._cancel_performance_profile_edit("balance")

        self.assertEqual(result, "break")
        self.assertFalse(gui._performance_profile_dirty["balance"])
        self.assertEqual(
            gui._performance_profile_entries["balance"]["hover"].get(), "0.2"
        )

    def test_enter_focuses_non_entry_parent_after_success(self):
        class _Parent:
            focused = False

            def focus_set(self):
                self.focused = True

        class _Widget:
            def __init__(self):
                self.master = _Parent()
                self.selection_cleared = False

            def selection_clear(self):
                self.selection_cleared = True

        event = type("_Event", (), {"widget": _Widget()})()

        ROVGui._finish_parameter_entry(event)

        self.assertTrue(event.widget.selection_cleared)
        self.assertTrue(event.widget.master.focused)

    def test_ctrl_s_saves_both_profiles_global_yaw_threshold_and_depth_pid(self):
        gui = self.make_gui()
        gui._apply_performance_profile_from_entries = mock.Mock(return_value=True)
        gui._apply_attitude_tuning_from_entry = mock.Mock(return_value=True)
        gui._pid_values_from_entries = mock.Mock(return_value=(55.0, 4.0, 50.0))
        gui._wait_for_pid_ack = mock.Mock()
        gui._record_applied_pid = mock.Mock()
        gui._log_event = mock.Mock()

        with mock.patch("pc_gui.update_config_values"):
            result = gui._save_current_parameters()

        self.assertEqual(result, "break")
        self.assertEqual(
            gui._apply_performance_profile_from_entries.call_args_list,
            [mock.call("balance"), mock.call("rage")],
        )
        gui._apply_attitude_tuning_from_entry.assert_called_once_with()
        self.assertEqual(gui._pid_values_from_entries.call_args_list, [mock.call(1)])
        self.assertEqual(gui.client.sent[-1]["pid_save"]["type"], 1)


if __name__ == "__main__":
    unittest.main()
