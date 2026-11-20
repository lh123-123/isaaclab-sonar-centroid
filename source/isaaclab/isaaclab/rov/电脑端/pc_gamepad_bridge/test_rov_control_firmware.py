"""Offline-only checks. These are not a full flight-controller build or SITL."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from firmware.prepare_control_bundle import COMMIT, ROOT
sys.path.insert(0, str(ROOT))
from prepare_independent_bundle import sources


class RovControlFirmwareTests(unittest.TestCase):
    def test_combined_activation_requires_exact_additions_and_unchanged_hardware(self):
        from firmware.activate_deadzone import same_parameters, CONTROL_PARAMETERS
        before = {'parameter_count': 1, 'parameters': {'FRAME_CONFIG': {'value': 2, 'type': 4}},
                  'autopilot_version': {'uid2': [1, 2, 3]}}
        after = json.loads(json.dumps(before))
        for name, default, _active, kind in CONTROL_PARAMETERS:
            after['parameters'][name] = {'value': default, 'type': kind}
        after['parameter_count'] = 11
        additions = {item[0] for item in CONTROL_PARAMETERS}
        self.assertEqual(same_parameters(before, after, additions), {})
        with self.assertRaisesRegex(RuntimeError, 'Unexpected firmware parameter additions'):
            same_parameters(before, after)  # old two-parameter activation must still reject
        after['parameters']['FRAME_CONFIG']['value'] = 1
        with self.assertRaisesRegex(RuntimeError, 'Pre-existing parameters changed'):
            same_parameters(before, after, additions)

    def test_prepared_source_is_exact_generator_output(self):
        original, updated = sources(ROOT / "baseline_dbe79216")
        bundle = ROOT / "prepared_tolerance_20261005"
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_commit"], COMMIT)
        self.assertEqual(manifest["status"], "OFFLINE_NOT_BUILT_NOT_FLASHED")
        self.assertEqual(set(updated), set(manifest["source_sha256"]))
        for path, text in updated.items():
            with self.subTest(path=path):
                self.assertEqual((bundle / "source" / path).read_text(encoding="utf-8"), text)
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), manifest["source_sha256"][path])
        for path in original:
            self.assertEqual(hashlib.sha256((ROOT / "baseline_dbe79216" / path).read_bytes()).hexdigest(),
                             manifest["baseline_sha256"][path])

    def test_patch_really_applies_and_reproduces_payloads_outside_project_git(self):
        # git apply from a subdirectory can silently ignore paths; check all
        # actual resulting files in an independent temporary source tree.
        _, updated = sources(ROOT / "baseline_dbe79216")
        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory) / "source"
            shutil.copytree(ROOT / "baseline_dbe79216", tree)
            patch = ROOT / "prepared_tolerance_20261005" / "ardusub_rov_control.patch"
            checked = subprocess.run(["git", "apply", "--check", str(patch)], cwd=tree,
                                     capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            applied = subprocess.run(["git", "apply", str(patch)], cwd=tree,
                                     capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
            self.assertEqual(applied.returncode, 0, applied.stdout + applied.stderr)
            for path, expected in updated.items():
                with self.subTest(path=path):
                    self.assertEqual((tree / path).read_text(encoding="utf-8"), expected)

    def test_stock_fallback_and_mode_integration_preserve_hardware_configuration(self):
        _, source_files = sources(ROOT / "baseline_dbe79216")
        for mode in ("stabilize", "althold"):
            text = source_files[f"ArduSub/mode_{mode}.cpp"]
            self.assertIn("sub.rov_pilot_enter()", text)
            self.assertIn("g2.rov_pilot.reset()", text)
            self.assertIn("if (!sub.rov_pilot_attitude(target_roll, target_pitch))", text)
            self.assertIn("pwm_to_angle_dz_trim", text)  # original disabled fallback
            self.assertIn("sub.rov_pilot_input_fresh() ? channel_forward", text)
            self.assertIn("sub.rov_pilot_input_fresh() ? channel_lateral", text)
        self.assertIn("g2.rov_pilot.shift_yaw", source_files["ArduSub/Attitude.cpp"])
        self.assertIn("last_message_ms != 0", source_files["ArduSub/mode_althold.cpp"])
        # Both feedback controllers remain ArduSub's, not a duplicate on PC/Pi.
        self.assertIn("position_control->D_update_controller()", source_files["ArduSub/mode_althold.cpp"])
        self.assertNotIn("SERVO1_FUNCTION", "".join(source_files.values()))
        motors = source_files["libraries/AP_Motors/AP_Motors6DOF.cpp"]
        self.assertIn('_motor_reverse[i] * (vertical[i] ? rv[i] : rh[i])', motors)
        self.assertIn('AP_GROUPINFO("ATT_PRIO", 16, AP_Motors6DOF, _attitude_priority, 0)', motors)
        self.assertIn('AP_SUBGROUPINFO(rov_pilot, "ROV_", 25,', source_files["ArduSub/Parameters.cpp"])

    def test_pending_parameters_are_separate_and_exported(self):
        name = "ROV回平与定深_仅新固件_待验证.params"
        project = (ROOT / name).read_text(encoding="utf-8")
        exported = (Path.home() / "Documents" / "QGroundControl" / "Parameters" / name).read_text(encoding="utf-8")
        self.assertEqual(project, exported)
        params = {line.split()[2]: float(line.split()[3]) for line in project.splitlines()
                  if line and not line.startswith("#")}
        self.assertEqual(set(params), {"ROV_EN", "ROV_T_RATE", "ROV_T_ACC", "ROV_Y_RATE", "ROV_Y_ACC",
                                      "ROV_Y_LEAD", "ROV_IN_TMO", "MOT_ATT_PRIO", "MOT_DZ_PWM", "MOT_DZ_IN"})
        self.assertEqual(params["ROV_EN"], 1)
        stock = (ROOT.parent / "ardusub_button_profile.params").read_text(encoding="utf-8")
        self.assertNotIn("ROV_EN", stock)
        self.assertNotIn("MOT_ATT_PRIO", stock)

    @unittest.skipUnless(shutil.which("g++"), "Host C++ compiler unavailable")
    def test_actual_target_and_allocation_kernels(self):
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "test_rov_control.exe"
            result = subprocess.run(["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                                     str(ROOT / "test_rov_control.cpp"), "-o", str(exe)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("10 ROV control scenario suites passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
