import json
import unittest
import tempfile
from copy import deepcopy
from pathlib import Path

from control_profile import (
    LocalActionController,
    button_mask,
    load_profile,
    resolve_button_conflicts,
    save_default_motion_gain,
    save_quick_targets,
)


class ControlProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile()

    def test_profile_is_intentionally_sparse(self):
        self.assertEqual(
            self.profile["button_forward_indices"],
            [0, 1],
        )

    def test_yaw_curve_validated_before_opening_control_and_legacy_default_preserved(self):
        self.assertEqual(self.profile['axis']['yaw_response_exponent'], .75)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            for value in (0, .1, float('nan'), float('inf'), True, '0.75', 3):
                candidate = deepcopy(self.profile)
                candidate['axis']['yaw_response_exponent'] = value
                path.write_text(json.dumps(candidate), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'yaw response'):
                    load_profile(path)
            del candidate['axis']['yaw_response_exponent']
            path.write_text(json.dumps(candidate), encoding='utf-8')
            self.assertNotIn('yaw_response_exponent', load_profile(path)['axis'])

    def test_gimbal_presets_use_unequal_discrete_steps(self):
        presets = self.profile["gimbal"]["presets_deg"]
        steps = [b - a for a, b in zip(presets, presets[1:])]
        self.assertGreater(len(set(steps)), 1)
        self.assertEqual(presets, [-80, -45, -20, 0, 20])

    def test_gimbal_changes_once_per_press(self):
        controller = LocalActionController(self.profile)
        up = 1 << self.profile["gimbal"]["up_button"]
        self.assertEqual(controller.update(up).gimbal_angle_deg, 20.0)
        self.assertIsNone(controller.update(up).gimbal_angle_deg)
        self.assertIsNone(controller.update(0).gimbal_angle_deg)
        self.assertEqual(controller.update(up).gimbal_angle_deg, 20.0)

    def test_gimbal_down_and_quick_left(self):
        controller = LocalActionController(self.profile)
        down = 1 << self.profile["gimbal"]["down_button"]
        quick = 1 << self.profile["gimbal"]["quick_button"]
        self.assertEqual(controller.update(down).gimbal_angle_deg, -20.0)
        controller.update(0)
        self.assertEqual(controller.update(down).gimbal_angle_deg, -45.0)
        controller.update(0)
        self.assertEqual(controller.update(quick).gimbal_angle_deg, -20.0)

    def test_right_servo_reset_and_diagonal_dpad(self):
        controller = LocalActionController(self.profile)
        result = controller.update(1 << 14)
        self.assertEqual(result.angle_servo_reset_pwm, 920)
        self.assertIsNone(result.gimbal_angle_deg)
        self.assertIsNone(controller.update(1 << 14).angle_servo_reset_pwm)
        controller.update(0)
        self.assertIsNone(controller.update((1 << 11) | (1 << 13)).gimbal_angle_deg)
        # Releasing one leg of the diagonal must not count as a new press.
        self.assertIsNone(controller.update(1 << 11).gimbal_angle_deg)
        controller.update(0)
        self.assertEqual(controller.update(1 << 11).gimbal_angle_deg, 20)
        controller.update(0)
        self.assertIsNone(controller.update((1 << 13) | (1 << 14)).angle_servo_reset_pwm)
        self.assertIsNone(controller.update(1 << 14).angle_servo_reset_pwm)

    def test_quick_angle_can_be_between_presets_and_step_from_it(self):
        profile = deepcopy(self.profile)
        profile["gimbal"]["quick_angle_deg"] = -27.5
        controller = LocalActionController(profile)
        self.assertEqual(controller.update(1 << 13).gimbal_angle_deg, -27.5)
        controller.update(0)
        self.assertEqual(controller.update(1 << 11).gimbal_angle_deg, -20)

    def test_quick_targets_persist_and_validate_without_touching_fc_params(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(self.profile), encoding="utf-8")
            save_quick_targets(path, -27.5, 925)
            saved = load_profile(path)
            self.assertEqual(saved["gimbal"]["quick_angle_deg"], -27.5)
            self.assertEqual(saved["servos"]["angle"]["reset_pwm"], 925)
            self.assertEqual(saved["pixhawk_button_parameters"], self.profile["pixhawk_button_parameters"])
            before = path.read_bytes()
            for angle, pwm in ((-91, 925), (float("nan"), 925), (-20, 499), (-20, 2501)):
                with self.assertRaises(ValueError):
                    save_quick_targets(path, angle, pwm)
                self.assertEqual(path.read_bytes(), before)

    def test_depth_and_stabilize_are_edges_and_y_wins(self):
        controller = LocalActionController(self.profile)
        self.assertEqual(controller.update(1 << 2).mode_action, "depth_toggle")
        self.assertIsNone(controller.update(1 << 2).mode_action)
        self.assertEqual(controller.update((1 << 2) | (1 << 3)).mode_action, "attitude_toggle")

    def test_disarm_suppresses_all_local_actions(self):
        controller = LocalActionController(self.profile)
        update = controller.update(button_mask((1, 2, 9, 14)))
        self.assertIsNone(update.mode_action)
        self.assertIsNone(update.gimbal_angle_deg)
        self.assertIsNone(update.motion_gain)
        self.assertEqual(controller.motion.gain, 0.4)
        self.assertEqual(controller.gimbal.angle_deg, 0)

    def test_opposing_servo_buttons_cancel_and_disarm_wins(self):
        for pair in ((15, 16), (17, 18), (19, 20)):
            self.assertEqual(resolve_button_conflicts(button_mask(pair)), 0)
        self.assertEqual(resolve_button_conflicts(button_mask((0, 1, 10, 17))), 1 << 1)
        self.assertEqual(resolve_button_conflicts(button_mask((10, 17))), 1 << 17)

    def test_parameter_files_agree_with_profile(self):
        base = Path(__file__).parent
        qgc = {}
        mp = {}
        for line in (base / "ardusub_button_profile.params").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                _, _, name, value, _ = line.split()
                qgc[name] = float(value)
        for line in (base / "ardusub_button_profile.param").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                name, value = line.split(",")
                mp[name] = float(value)
        self.assertEqual(qgc, self.profile["pixhawk_button_parameters"])
        self.assertEqual(mp, qgc)

    def test_qgc_default_parameter_copy_matches_project_copy(self):
        base = Path(__file__).parent
        default_copy = Path.home() / "Documents" / "QGroundControl" / "Parameters" / "ROV比赛参数_最新版.params"
        self.assertEqual(
            default_copy.read_text(encoding="utf-8"),
            (base / "ardusub_button_profile.params").read_text(encoding="utf-8"),
        )

    def test_motor_endpoints_and_only_one_joystick_gain_layer(self):
        params = self.profile["pixhawk_button_parameters"]
        self.assertEqual(params["MOT_PWM_MIN"], 1000)
        self.assertEqual(params["MOT_PWM_MAX"], 2000)
        self.assertEqual(params["JS_GAIN_DEFAULT"], 1.0)
        self.assertEqual(params["JS_GAIN_STEPS"], 1)
        for i in range(1, 9):
            self.assertEqual([params[f"SERVO{i}_{key}"] for key in ("MIN", "TRIM", "MAX")], [1000,1500,2000])

    def test_startup_reference_is_50_but_stock_parameters_do_not_claim_it_active(self):
        self.assertEqual(self.profile["thruster_startup"]["reference_offset_us"], 50)
        self.assertNotIn("MOT_DZ_PWM", self.profile["pixhawk_button_parameters"])
        self.assertNotIn("MOT_DZ_IN", self.profile["pixhawk_button_parameters"])

    def test_startup_reference_validation_and_legacy_profile_default(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            for value in (0, 50, 100):
                profile = deepcopy(self.profile)
                profile["thruster_startup"]["reference_offset_us"] = value
                path.write_text(json.dumps(profile), encoding="utf-8")
                self.assertEqual(load_profile(path)["thruster_startup"]["reference_offset_us"], value)
            for value in (True, -1, 101, 50.5, "50", None):
                profile["thruster_startup"]["reference_offset_us"] = value
                path.write_text(json.dumps(profile), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_profile(path)
            profile.pop("thruster_startup")
            path.write_text(json.dumps(profile), encoding="utf-8")
            self.assertEqual(load_profile(path)["thruster_startup"]["reference_offset_us"], 50)
            for item in (None, [], 50, "50"):
                profile["thruster_startup"] = item
                path.write_text(json.dumps(profile), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_profile(path)

    def test_stick_gain_cycles_once_per_left_menu_press(self):
        controller = LocalActionController(self.profile)
        cycle = 1 << self.profile["motion_gain"]["cycle_button"]
        self.assertEqual(controller.motion.gain, 0.4)
        self.assertEqual(controller.update(cycle).motion_gain, 1.0)
        self.assertIsNone(controller.update(cycle).motion_gain)
        controller.update(0)
        self.assertEqual(controller.update(cycle).motion_gain, 0.4)

    def test_vertical_gain_is_independent_and_lb_resets_both(self):
        c = LocalActionController(self.profile)
        result = c.update(1 << 6)
        self.assertEqual(result.vertical_gain, 1.0)
        self.assertIsNone(result.motion_gain)
        self.assertEqual(c.motion.gain, 0.4)
        self.assertIsNone(c.update(1 << 6).vertical_gain)
        c.update(0)
        c.update(1 << 4)
        self.assertEqual((c.motion.gain, c.vertical.gain), (1.0, 1.0))
        c.update(0)
        result = c.update((1 << 9) | (1 << 4) | (1 << 6))
        self.assertEqual((result.motion_gain, result.vertical_gain), (0.4, 0.4))
        self.assertEqual((c.motion.gain, c.vertical.gain), (0.4, 0.4))
        self.assertIsNone(c.update(1 << 9).motion_gain)
        c.update(0)
        self.assertEqual(c.update(1 << 6).vertical_gain, 1.0)
        self.assertEqual((c.motion.gain, c.vertical.gain), (0.4, 1.0))

    def test_changing_saved_default_does_not_jump_current_gears(self):
        c = LocalActionController(self.profile)
        c.set_default_gain(1.0)
        self.assertEqual((c.motion.gain, c.vertical.gain), (0.4, 0.4))
        result = c.update(1 << 9)
        self.assertEqual((result.motion_gain, result.vertical_gain), (1.0, 1.0))
        with self.assertRaises(ValueError):
            c.set_default_gain(0.55)

    def test_default_motion_gain_persists_without_touching_servos_or_fc_params(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(self.profile), encoding="utf-8")
            save_default_motion_gain(path, 1.0)
            saved = load_profile(path)
            c = LocalActionController(saved)
            self.assertEqual((c.motion.gain, c.vertical.gain), (1.0, 1.0))
            self.assertEqual(saved["servos"], self.profile["servos"])
            self.assertEqual(saved["pixhawk_button_parameters"], self.profile["pixhawk_button_parameters"])
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                save_default_motion_gain(path, float("nan"))
            self.assertEqual(path.read_bytes(), before)

    def test_disarm_also_suppresses_both_new_gain_cycles(self):
        c = LocalActionController(self.profile)
        result = c.update(button_mask((1, 4, 6, 9)))
        self.assertIsNone(result.motion_gain)
        self.assertIsNone(result.vertical_gain)
        self.assertEqual((c.motion.gain, c.vertical.gain), (0.4, 0.4))

    def test_profile_rejects_invalid_default_or_duplicate_gain_buttons(self):
        for key in ("default", "buttons"):
            profile = deepcopy(self.profile)
            if key == "default":
                profile["vertical_gain"]["initial"] = 0.55
            else:
                profile["vertical_gain"]["cycle_button"] = profile["motion_gain"]["reset_button"]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.json"
                path.write_text(json.dumps(profile), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_profile(path)

    def test_directions_follow_latest_same_command_airflow_confirmation(self):
        from motor_test import PHYSICAL_TO_MOTOR
        params = self.profile["pixhawk_button_parameters"]
        self.assertEqual(params["JS_THR_GAIN"], 1.0)
        for physical, logical in enumerate(PHYSICAL_TO_MOTOR, 1):
            expected = -1 if physical in (1, 4, 5, 8) else 1
            self.assertEqual(params[f"MOT_{logical}_DIRECTION"], expected)
        lines = Path(__file__).with_name("ardusub_button_profile.params").read_text(encoding="utf-8").splitlines()
        types = {line.split()[2]: int(line.split()[4]) for line in lines if line and not line.startswith("#")}
        for logical in range(1, 9):
            self.assertEqual(types[f"MOT_{logical}_DIRECTION"], 2)  # AP_Int8 / MAV_PARAM_TYPE_INT8

    def test_servo_buttons_are_local_and_flight_controller_functions_are_disabled(self):
        params = self.profile["pixhawk_button_parameters"]
        for index in (10, 15, 16, 17, 18, 19, 20):
            self.assertEqual(params[f"BTN{index}_FUNCTION"], 0)
        for channel in (9, 10, 11):
            self.assertEqual(params[f"SERVO{channel}_FUNCTION"], 0)

    def test_forward_mask_excludes_local_and_unused_buttons(self):
        mask = button_mask(self.profile["button_forward_indices"])
        for index in (0, 1):
            self.assertTrue(mask & (1 << index))
        for index in range(2, 21):
            self.assertFalse(mask & (1 << index))

    def test_qgc_actions_match_profile_presets(self):
        actions_path = Path(__file__).with_name("QGC_ROV_FlyView_Actions.json")
        actions = json.loads(actions_path.read_text(encoding="utf-8"))["actions"]
        self.assertEqual(
            [item["param1"] for item in actions],
            self.profile["gimbal"]["presets_deg"],
        )
        self.assertTrue(all(item["mavCmd"] == 205 for item in actions))
        self.assertTrue(all(item["param7"] == 2 for item in actions))


if __name__ == "__main__":
    unittest.main()
