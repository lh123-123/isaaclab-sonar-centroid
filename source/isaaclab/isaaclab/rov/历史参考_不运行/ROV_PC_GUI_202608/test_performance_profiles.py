import configparser
import copy
import importlib.util
import inspect
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from config_loader import config_fingerprint, load_config
from pc_gui import ROVGui
from secure_protocol import ProtocolError, validate_control_payload
from test_motion_tuning import ROVController, make_allocator


ROOT = Path(__file__).resolve().parent
EXTENDED_FIELDS = {
    "hover_force_per_thruster_n",
    "vertical_ascent_front_rear_trim_percent",
    "vertical_descent_front_rear_trim_percent",
    "forward_front_rear_trim_percent",
    "backward_front_rear_trim_percent",
    "roll_max_moment_nm",
    "pitch_max_moment_nm",
    "yaw_max_moment_nm",
    "left_lateral_surge_trim_percent",
    "right_lateral_surge_trim_percent",
}
LATERAL_FIELDS = {
    "left_lateral_surge_trim_percent",
    "right_lateral_surge_trim_percent",
}


def load_pi_protocol():
    spec = importlib.util.spec_from_file_location(
        "performance_profile_pi_protocol", ROOT / "pi_runtime" / "secure_protocol.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def profile(mode="rage"):
    return {
        "mode": mode,
        "horizontal_limit_fraction": 0.4 if mode == "rage" else 0.2,
        "vertical_limit_fraction": 0.3 if mode == "rage" else 0.2,
        "hover_force_per_thruster_n": 0.0,
        "vertical_ascent_front_rear_trim_percent": 0.0,
        "vertical_descent_front_rear_trim_percent": 0.0,
        "forward_front_rear_trim_percent": 8.0 if mode == "rage" else 0.0,
        "backward_front_rear_trim_percent": -8.0 if mode == "rage" else 0.0,
        "left_lateral_surge_trim_percent": -5.0 if mode == "balance" else 0.0,
        "right_lateral_surge_trim_percent": -5.0 if mode == "balance" else 0.0,
        "roll_max_moment_nm": 4.0,
        "pitch_max_moment_nm": 3.5 if mode == "rage" else 3.0,
        "yaw_max_moment_nm": 4.0,
        "roll_pid": {"kp": 0.32, "ki": 0.02, "kd": 0.11},
        "pitch_pid": {"kp": 0.38, "ki": 0.015, "kd": 0.16},
        "yaw_pid": {"kp": 0.27, "ki": 0.01, "kd": 0.11},
    }


class PerformanceProfileProtocolTests(unittest.TestCase):
    def test_pc_and_pi_accept_complete_and_legacy_profiles(self):
        pi_protocol = load_pi_protocol()
        complete = profile()
        legacy = {
            key: value
            for key, value in complete.items()
            if key not in EXTENDED_FIELDS
        }
        previous_complete = {
            key: value
            for key, value in complete.items()
            if key not in LATERAL_FIELDS
        }

        for validator in (
            validate_control_payload,
            pi_protocol.validate_control_payload,
        ):
            self.assertEqual(
                validator({"performance_profile": complete}),
                {"performance_profile": complete},
            )
            self.assertEqual(
                validator({"performance_profile": legacy}),
                {"performance_profile": legacy},
            )
            self.assertEqual(
                validator({"performance_profile": previous_complete}),
                {"performance_profile": previous_complete},
            )

    def test_pc_and_pi_reject_incomplete_or_unsafe_complete_profiles(self):
        pi_protocol = load_pi_protocol()
        invalid_profiles = [
            {key: value for key, value in profile().items() if key != "yaw_max_moment_nm"},
            dict(profile(), horizontal_limit_fraction=float("nan")),
            dict(profile(), forward_front_rear_trim_percent=float("inf")),
            dict(profile(), left_lateral_surge_trim_percent=100.1),
            dict(profile(), vertical_limit_fraction=0.51),
            dict(
                profile(),
                vertical_limit_fraction=0.1,
                hover_force_per_thruster_n=6.0,
            ),
            dict(profile(), hover_force_per_thruster_n=10.1),
            dict(profile(), roll_max_moment_nm=20.1),
        ]

        for validator in (
            validate_control_payload,
            pi_protocol.validate_control_payload,
        ):
            for invalid in invalid_profiles:
                with self.subTest(validator=validator.__module__, fields=invalid.keys()):
                    with self.assertRaises(pi_protocol.ProtocolError if validator is pi_protocol.validate_control_payload else ProtocolError):
                        validator({"performance_profile": invalid})


class PerformanceProfileConfigTests(unittest.TestCase):
    def test_config_fingerprint_detects_parameters_but_excludes_shared_key(self):
        first = configparser.ConfigParser(interpolation=None)
        first.read(ROOT / "config.ini", encoding="utf-8")
        second = copy.deepcopy(first)
        second.set("safety", "shared_key", "x" * 64)
        self.assertEqual(config_fingerprint(first), config_fingerprint(second))
        second.set("control", "performance_mode", "rage")
        self.assertNotEqual(config_fingerprint(first), config_fingerprint(second))

    def test_operational_profiles_are_explicit_and_preserve_limits_and_pids(self):
        config = load_config(ROOT / "config.ini")
        balance = config["mode_balance"]
        rage = config["mode_rage"]

        self.assertEqual(balance.getfloat("horizontal_limit_fraction"), 0.2)
        self.assertEqual(balance.getfloat("vertical_limit_fraction"), 0.2)
        self.assertEqual(rage.getfloat("horizontal_limit_fraction"), 0.4)
        self.assertEqual(rage.getfloat("vertical_limit_fraction"), 0.3)
        self.assertEqual(rage.getfloat("pitch_max_moment_nm"), 3.5)
        self.assertEqual(balance.getfloat("left_lateral_surge_trim_percent"), -16.0)
        self.assertEqual(balance.getfloat("right_lateral_surge_trim_percent"), -20.0)
        for section in (balance, rage):
            for field in EXTENDED_FIELDS:
                self.assertIn(field, section)
            for field in (
                "vertical_ascent_front_rear_trim_percent",
                "vertical_descent_front_rear_trim_percent",
                "forward_front_rear_trim_percent",
                "backward_front_rear_trim_percent",
            ):
                self.assertLessEqual(abs(section.getfloat(field)), 100.0)
        self.assertEqual(balance.getfloat("roll_kp"), 0.25)
        self.assertEqual(rage.getfloat("yaw_kd"), 0.11)

    def test_config_loader_keeps_legacy_profile_sections_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.ini"
            config = configparser.ConfigParser()
            config.read(ROOT / "config.ini", encoding="utf-8")
            for section in ("mode_balance", "mode_rage"):
                for field in EXTENDED_FIELDS:
                    config.remove_option(section, field)
            with path.open("w", encoding="utf-8") as handle:
                config.write(handle)

            loaded = load_config(path)

        self.assertTrue(loaded.has_section("mode_balance"))
        self.assertTrue(loaded.has_section("mode_rage"))


class PerformanceProfileRuntimeTests(unittest.TestCase):
    def make_controller(self):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "config.ini"
        shutil.copyfile(ROOT / "config.ini", path)
        return directory, path, ROVController(config_path=path)

    def test_pi_loads_complete_profiles_and_activates_every_owned_field(self):
        directory, _path, rov = self.make_controller()
        self.addCleanup(directory.cleanup)
        rov.allocator = make_allocator()

        rage = rov.performance_profiles["rage"]
        self.assertTrue(EXTENDED_FIELDS.issubset(rage))
        rov._activate_performance_profile(rage)

        self.assertEqual(rov.horizontal_limit_fraction, 0.4)
        self.assertEqual(rov.vertical_limit_fraction, 0.3)
        self.assertEqual(
            rov.forward_front_rear_trim_percent,
            rage["forward_front_rear_trim_percent"],
        )
        self.assertEqual(
            rov.backward_front_rear_trim_percent,
            rage["backward_front_rear_trim_percent"],
        )
        self.assertEqual(
            rov.left_lateral_surge_trim_percent,
            rage["left_lateral_surge_trim_percent"],
        )
        self.assertEqual(
            rov.right_lateral_surge_trim_percent,
            rage["right_lateral_surge_trim_percent"],
        )
        self.assertEqual(rov.controller.pitch_max_moment_nm, 3.5)
        self.assertEqual(rov.controller.get_pid_snapshot()["yaw"]["kp"], 0.27)

    def test_mode_switch_applies_each_complete_profile_without_cross_talk(self):
        directory, _path, rov = self.make_controller()
        self.addCleanup(directory.cleanup)
        rov.allocator = make_allocator()

        for mode in ("rage", "balance", "rage"):
            expected = rov.performance_profiles[mode]
            rov._apply_performance_mode(mode)

            self.assertEqual(rov.performance_mode, mode)
            for attribute, field in (
                ("horizontal_limit_fraction", "horizontal_limit_fraction"),
                ("vertical_limit_fraction", "vertical_limit_fraction"),
                ("vertical_hover_force_per_thruster_n", "hover_force_per_thruster_n"),
                ("vertical_ascent_front_rear_trim_percent", "vertical_ascent_front_rear_trim_percent"),
                ("vertical_descent_front_rear_trim_percent", "vertical_descent_front_rear_trim_percent"),
                ("forward_front_rear_trim_percent", "forward_front_rear_trim_percent"),
                ("backward_front_rear_trim_percent", "backward_front_rear_trim_percent"),
                ("left_lateral_surge_trim_percent", "left_lateral_surge_trim_percent"),
                ("right_lateral_surge_trim_percent", "right_lateral_surge_trim_percent"),
            ):
                self.assertEqual(getattr(rov, attribute), expected[field])
            self.assertEqual(rov.controller.roll_max_moment_nm, expected["roll_max_moment_nm"])
            self.assertEqual(rov.controller.pitch_max_moment_nm, expected["pitch_max_moment_nm"])
            self.assertEqual(rov.controller.yaw_max_moment_nm, expected["yaw_max_moment_nm"])
            self.assertEqual(
                rov.allocator.left_lateral_surge_trim_percent,
                expected["left_lateral_surge_trim_percent"],
            )
            self.assertEqual(
                rov.allocator.right_lateral_surge_trim_percent,
                expected["right_lateral_surge_trim_percent"],
            )
            snapshot = rov.controller.get_pid_snapshot()
            for axis in ("roll", "pitch", "yaw"):
                self.assertEqual(
                    {
                        field: snapshot[axis][field]
                        for field in ("kp", "ki", "kd")
                    },
                    expected[f"{axis}_pid"],
                )

    def test_legacy_config_does_not_raise_rage_vertical_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.ini"
            config = configparser.ConfigParser()
            config.read(ROOT / "config.ini", encoding="utf-8")
            config.remove_section("mode_balance")
            config.remove_section("mode_rage")
            with path.open("w", encoding="utf-8") as handle:
                config.write(handle)

            rov = ROVController(config_path=path)

        self.assertEqual(rov.vertical_limit_fraction, 0.3)
        self.assertEqual(
            rov.performance_profiles["rage"]["vertical_limit_fraction"], 0.3
        )

    def test_pi_persists_complete_profile_and_expands_legacy_request(self):
        directory, path, rov = self.make_controller()
        self.addCleanup(directory.cleanup)
        requested = copy.deepcopy(profile("balance"))
        requested["hover_force_per_thruster_n"] = 0.25
        requested["vertical_ascent_front_rear_trim_percent"] = -12.0
        requested["left_lateral_surge_trim_percent"] = -7.0

        actual = rov._apply_performance_profile(requested)
        saved = load_config(path)["mode_balance"]
        legacy = {
            key: value for key, value in profile("rage").items()
            if key not in EXTENDED_FIELDS
        }
        expanded = rov._validated_performance_profile(legacy)
        previous_complete = {
            key: value for key, value in profile("balance").items()
            if key not in LATERAL_FIELDS
        }
        expanded_previous = rov._validated_performance_profile(previous_complete)

        self.assertEqual(actual["hover_force_per_thruster_n"], 0.25)
        self.assertEqual(saved.getfloat("hover_force_per_thruster_n"), 0.25)
        self.assertEqual(
            saved.getfloat("vertical_ascent_front_rear_trim_percent"), -12.0
        )
        self.assertEqual(saved.getfloat("left_lateral_surge_trim_percent"), -7.0)
        self.assertTrue(EXTENDED_FIELDS.issubset(expanded))
        self.assertEqual(
            expanded_previous["left_lateral_surge_trim_percent"],
            rov.performance_profiles["balance"]["left_lateral_surge_trim_percent"],
        )


class PerformanceProfileLayoutTests(unittest.TestCase):
    def test_command_panel_uses_compact_entries_without_legacy_sliders(self):
        source = inspect.getsource(ROVGui._build_command_panel)

        self.assertNotIn("ttk.Scale(", source)
        self.assertNotIn("推力设置（倍率0–0.5）", source)
        for entry_key in (
            "hover",
            "ascent_trim",
            "descent_trim",
            "forward_trim",
            "backward_trim",
            "left_lateral",
            "right_lateral",
            "roll_moment",
            "pitch_moment",
            "yaw_moment",
        ):
            self.assertIn(f'"{entry_key}"', source)


if __name__ == "__main__":
    unittest.main()
