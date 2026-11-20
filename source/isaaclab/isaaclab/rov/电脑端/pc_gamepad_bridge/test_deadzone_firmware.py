"""Offline firmware artifact checks; never opens a vehicle connection."""
import shutil
import copy
import subprocess
import tempfile
import unittest
from pathlib import Path


FIRMWARE = Path(__file__).with_name("firmware")


class DeadzoneFirmwareTests(unittest.TestCase):
    def test_boot_pressure_recalibration_is_bounded_and_requires_reboot(self):
        from firmware.activate_deadzone import same_parameters
        before = {'parameter_count': 2,
                  'parameters': {'STAT_BOOTCNT': {'value': 73, 'type': 4},
                                 'BARO2_GND_PRESS': {'value': 99130, 'type': 9}},
                  'autopilot_version': {'uid2': [1]}}
        after = copy.deepcopy(before)
        after['parameter_count'] = 4
        after['parameters'].update({'MOT_DZ_PWM': {'value': 0, 'type': 4},
                                    'MOT_DZ_IN': {'value': 0.01, 'type': 9}})
        after['parameters']['STAT_BOOTCNT']['value'] = 74
        after['parameters']['BARO2_GND_PRESS']['value'] = 99350
        self.assertEqual(set(same_parameters(before, after)), {'STAT_BOOTCNT', 'BARO2_GND_PRESS'})
        for value in (102000, float('nan'), float('inf')):
            after['parameters']['BARO2_GND_PRESS']['value'] = value
            with self.assertRaisesRegex(RuntimeError, 'Pre-existing parameters changed'):
                same_parameters(before, after)
        after['parameters']['BARO2_GND_PRESS']['value'] = 99350
        after['parameters']['STAT_BOOTCNT']['value'] = 73
        with self.assertRaisesRegex(RuntimeError, 'Pre-existing parameters changed'):
            same_parameters(before, after)

    def test_deployment_only_allows_monotonic_read_only_runtime_counters(self):
        from firmware.activate_deadzone import same_parameters
        before = {'parameter_count': 2,
                  'parameters': {'STAT_BOOTCNT': {'value': 73, 'type': 4},
                                 'STAT_RUNTIME': {'value': 44576, 'type': 6}},
                  'autopilot_version': {'uid2': [1]}}
        after = copy.deepcopy(before)
        after['parameters'].update({'MOT_DZ_PWM': {'value': 0, 'type': 4},
                                    'MOT_DZ_IN': {'value': 0.01, 'type': 9}})
        after['parameters']['STAT_BOOTCNT']['value'] = 75
        after['parameters']['STAT_RUNTIME']['value'] = 45000
        after['parameter_count'] = 4
        self.assertEqual(set(same_parameters(before, after)), {'STAT_BOOTCNT', 'STAT_RUNTIME'})
        after['parameters']['STAT_RUNTIME']['value'] = 0
        with self.assertRaisesRegex(RuntimeError, 'Pre-existing parameters changed'):
            same_parameters(before, after)

    def test_deployment_requires_all_old_parameters_unchanged(self):
        from firmware.activate_deadzone import same_parameters
        before = {'parameter_count': 1, 'parameters': {'MOT_PWM_MIN': {'value': 1000, 'type': 4}},
                  'autopilot_version': {'uid2': [1, 2, 3]}}
        after = copy.deepcopy(before)
        after['parameters'].update({'MOT_DZ_PWM': {'value': 0, 'type': 4},
                                    'MOT_DZ_IN': {'value': 0.01, 'type': 9}})
        after['parameter_count'] = 3
        same_parameters(before, after)
        after['parameters']['MOT_PWM_MIN']['value'] = 1100
        with self.assertRaisesRegex(RuntimeError, 'Pre-existing parameters changed'):
            same_parameters(before, after)

    def test_deployment_rejects_incomplete_or_wrong_vehicle_snapshots(self):
        from firmware.activate_deadzone import same_parameters
        before = {'parameter_count': 0, 'parameters': {}, 'autopilot_version': {'uid2': [1]}}
        after = {'parameter_count': 2,
                 'parameters': {'MOT_DZ_PWM': {'value': 0, 'type': 4},
                                'MOT_DZ_IN': {'value': 0.01, 'type': 9}},
                 'autopilot_version': {'uid2': [2]}}
        with self.assertRaisesRegex(RuntimeError, 'Vehicle identity changed'):
            same_parameters(before, after)
        after['parameter_count'] = 3
        with self.assertRaisesRegex(RuntimeError, 'Incomplete parameter snapshot'):
            same_parameters(before, after)

    def test_patch_embeds_exact_tested_header(self):
        patch = (FIRMWARE / "ardusub_deadzone.patch").read_text(encoding="utf-8")
        new_file = patch.split("+++ b/libraries/AP_Motors/ROV_DeadzonePWM.h\n", 1)[1]
        payload = new_file.splitlines()[1:]
        self.assertTrue(all(line.startswith("+") for line in payload))
        embedded = "\n".join(line[1:] for line in payload) + "\n"
        self.assertEqual(embedded, (FIRMWARE / "ROV_DeadzonePWM.h").read_text(encoding="utf-8"))
        self.assertIn('AP_GROUPINFO("DZ_PWM", 14, AP_Motors6DOF, _startup_deadzone_pwm, 0)', patch)
        self.assertIn('AP_GROUPINFO("DZ_IN", 15, AP_Motors6DOF, _startup_deadzone_input, 0.01f)', patch)
        self.assertIn("ROV_DeadzonePWM::calculate(thrust_in", patch)
        self.assertNotIn("DO_MOTOR_TEST", patch)

    def test_custom_activation_parameters_are_separate_and_exported_consistently(self):
        name = "ROV启转补偿_仅自定义固件.params"
        project = (FIRMWARE / name).read_text(encoding="utf-8")
        exported = (Path.home() / "Documents" / "QGroundControl" / "Parameters" / name).read_text(encoding="utf-8")
        self.assertEqual(project, exported)
        params = {line.split()[2]: float(line.split()[3]) for line in project.splitlines()
                  if line and not line.startswith("#")}
        self.assertEqual(params, {"MOT_DZ_PWM": 50, "MOT_DZ_IN": 0.01})
        stock = FIRMWARE.parent / "ardusub_button_profile.params"
        self.assertNotIn("MOT_DZ_", stock.read_text(encoding="utf-8"))

    @unittest.skipUnless(shutil.which("g++"), "Host C++ compiler unavailable; firmware kernel unverified")
    def test_actual_cpp_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "deadzone_test.exe"
            build = subprocess.run(
                ["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                 str(FIRMWARE / "test_deadzone_pwm.cpp"), "-o", str(executable)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
