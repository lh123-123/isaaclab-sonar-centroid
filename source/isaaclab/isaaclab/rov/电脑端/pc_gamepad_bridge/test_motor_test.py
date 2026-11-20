import time
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flight_mode import FlightModeController
from motor_test import (EXPECTED_PARAMS, DIRECTION_PARAMS, MOTION_FACTORS, MOTION_GROUPS,
                        MotorTestSession, PHYSICAL_TO_MOTOR, motion_pwm_plan)
from motor_test_panel import MotorTestPanel


def packet(kind, system=1, component=1, **fields):
    return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: system,
                           get_srcComponent=lambda: component, **fields)


class MotorSessionTests(unittest.TestCase):
    def setUp(self):
        self.modes = FlightModeController()
        self.link = Mock()
        self.s = MotorTestSession(self.modes, self.link)
        self.modes.last_heartbeat = 10
        self.modes.mode = 19

    def mapped(self):
        for name, value in EXPECTED_PARAMS.items():
            self.s.observe(packet("PARAM_VALUE", param_id=name, param_value=value), 10)

    def ready(self):
        self.mapped()
        self.assertTrue(self.s.prepare(10))
        self.s.tick(10)
        self.assertEqual(self.s.phase, "arm")
        self.modes.armed = True
        self.s.tick(10.1)
        self.assertEqual(self.s.phase, "ready")
        self.s.observe(packet("COMMAND_ACK", command=209, result=0), 10.2)
        self.link.reset_mock()

    def outputs(self):
        return [c.args for c in self.link.mav.command_long_send.call_args_list if c.args[2] == 209]

    def test_monitor_never_arms_or_sends_motion(self):
        self.s.tick(10)
        self.link.mav.command_long_send.assert_not_called()
        self.link.mav.manual_control_send.assert_not_called()
        self.assertEqual(self.link.mav.param_request_read_send.call_count, 18)

    def test_missing_wrong_stale_or_already_armed_blocks_prepare(self):
        self.assertFalse(self.s.prepare(10))
        self.mapped()
        self.s.params["SERVO1_FUNCTION"] = 33
        self.assertFalse(self.s.prepare(10))
        self.mapped()
        self.assertFalse(self.s.prepare(14))
        self.modes.armed = True
        self.assertFalse(self.s.prepare(10))
        self.link.mav.command_long_send.assert_not_called()

    def test_mode_and_arm_require_real_heartbeat_not_ack(self):
        self.mapped()
        self.modes.mode = 0
        self.s.prepare(10)
        self.s.observe(packet("COMMAND_ACK", command=176, result=0), 10.1)
        self.s.tick(10.1)
        self.assertEqual(self.s.phase, "mode")
        self.modes.mode = 19
        self.s.tick(10.2)
        self.assertEqual(self.s.phase, "arm")
        self.s.observe(packet("COMMAND_ACK", command=400, result=0), 10.3)
        self.s.tick(10.3)
        self.assertEqual(self.s.phase, "arm")
        self.assertEqual(self.outputs(), [])

    def test_each_physical_motor_has_correct_zero_based_command(self):
        self.ready()
        for physical, logical in enumerate(PHYSICAL_TO_MOTOR, 1):
            for direction in (-1, 1):
                self.link.reset_mock()
                self.s.selected = physical
                self.assertTrue(self.s.hold(direction, 10, 10.3))
                self.s.tick(10.3)
                calls = self.outputs()
                self.assertEqual(len(calls), 8)
                expected = (1, 1, 209, 0, logical - 1, 1, 1500+direction*50, 0, 0, 2, 0)
                self.assertIn(expected, calls)
                self.assertEqual(sum(c[6] != 1500 for c in calls), 1)
                self.s.release(10.31)
                self.assertTrue(all(c[6] == 1500 for c in self.outputs()[-8:]))

    def test_hold_runs_beyond_old_limit_until_release_with_fresh_feedback(self):
        self.ready()
        self.assertTrue(self.s.hold(1, 15, 10.3))
        self.assertFalse(self.s.hold(1, 15, 10.4))
        for index in range(1, 301):
            now = 10.3 + index * 0.1
            self.modes.last_heartbeat = now
            self.s.observe(packet("COMMAND_ACK", command=209, result=0), now)
            self.s.tick(now)
            self.assertEqual(self.s.phase, "ready")
            self.assertEqual(self.s.direction, 1)
            self.assertEqual(sum(c[6] != 1500 for c in self.outputs()[-8:]), 1)
        self.s.release(now + 0.01)
        self.assertEqual(self.s.direction, 0)
        self.assertTrue(all(c[6] == 1500 for c in self.outputs()[-8:]))

    def test_no_ack_rejection_link_loss_and_mode_change_stop_and_disarm(self):
        for reason in ("no_ack", "reject", "lost", "mode", "mapping"):
            self.setUp()
            self.ready()
            self.s.hold(1, 10, 10.3)
            if reason == "no_ack":
                self.s.ack_time = None
                self.s.tick(11.2)
            elif reason == "reject":
                self.s.observe(packet("COMMAND_ACK", command=209, result=2), 10.4)
            elif reason == "lost":
                self.s.tick(14)
            elif reason == "mode":
                self.modes.mode = 2
                self.s.tick(10.4)
            else:
                self.s.observe(packet("PARAM_VALUE", param_id="FRAME_CONFIG", param_value=1), 10.4)
            self.assertEqual(self.s.phase, "stopping", reason)
            self.assertEqual(self.s.direction, 0, reason)
            self.assertEqual(self.link.mav.command_long_send.call_args.args[2:5], (400, 0, 0))

    def test_stop_confirmation_cooldown_and_no_force_arm(self):
        self.ready()
        self.s.stop(10.4)
        self.s.tick(13.5)
        self.assertEqual(self.s.phase, "stopping")
        self.assertIn("未确认", self.s.notice)
        self.modes.last_heartbeat = 14
        self.modes.armed = False
        self.s.tick(14)
        self.assertEqual(self.s.phase, "monitor")
        self.assertFalse(self.s.prepare(14))
        for c in self.link.mav.command_long_send.call_args_list:
            if c.args[2] == 400:
                self.assertEqual(c.args[5], 0)

    def test_wrong_vehicle_cannot_prove_mapping_or_ack(self):
        self.s.observe(packet("PARAM_VALUE", system=2, param_id="FRAME_CONFIG", param_value=2), 10)
        self.assertEqual(self.s.params, {})
        self.ready()
        self.s.ack_time = None
        self.s.observe(packet("COMMAND_ACK", command=209, result=0, target_component=191), 10.4)
        self.assertIsNone(self.s.ack_time)

    def directions(self, now=10.2):
        for name, value in zip(DIRECTION_PARAMS, (1, -1, -1, 1, 1, -1, -1, 1)):
            self.s.observe(packet("PARAM_VALUE", param_id=name, param_value=value), now)

    def test_horizontal_group_requires_explicit_fresh_geometry_readback(self):
        self.ready()
        self.directions()
        self.assertFalse(self.s.hold_motion('右转', 1450, 1550, 10.3))
        self.assertEqual(self.outputs(), [])
        for value in (-1, 2, float('nan')):
            self.s.observe(packet('PARAM_VALUE', param_id='MOT_H_REV', param_value=value), 10.2)
            self.assertFalse(self.s.hold_motion('右移', 1450, 1550, 10.3))
        self.s.observe(packet('PARAM_VALUE', param_id='MOT_H_REV', param_value=1), 0)
        self.assertFalse(self.s.hold_motion('右转', 1450, 1550, 10.3))
        self.assertEqual(self.outputs(), [])

    def test_corrected_group_outputs_match_both_horizontal_axes(self):
        self.ready()
        self.directions()
        for reverse in (0, 1):
            self.s.observe(packet('PARAM_VALUE', param_id='MOT_H_REV', param_value=reverse), 10.2)
            for action, expected in (
                    ('右转', (1,1,1,1,0,0,0,0)),
                    ('左转', (-1,-1,-1,-1,0,0,0,0)),
                    ('右移', (1,1,-1,-1,0,0,0,0)),
                    ('左移', (-1,-1,1,1,0,0,0,0))):
                self.link.reset_mock()
                self.assertTrue(self.s.hold_motion(action, 1450, 1550, 10.3))
                self.s.tick(10.3)
                sign = -1 if reverse else 1
                self.assertEqual(tuple(c[6] for c in self.outputs()),
                                 tuple(1500+sign*50*v for v in expected))
                self.s.release(10.31)

    def test_horizontal_geometry_change_stops_active_test(self):
        self.ready()
        self.directions()
        self.s.observe(packet('PARAM_VALUE', param_id='MOT_H_REV', param_value=1), 10.2)
        self.assertTrue(self.s.hold_motion('右转', 1450, 1550, 10.3))
        self.s.observe(packet('PARAM_VALUE', param_id='MOT_H_REV', param_value=0), 10.4)
        self.assertIsNone(self.s.motion_action)
        self.assertEqual(self.s.phase, 'stopping')
        self.assertTrue(all(c[6] == 1500 for c in self.outputs()[-8:]))

    def test_horizontal_geometry_expiry_stops_active_test(self):
        self.ready()
        self.directions()
        self.s.observe(packet('PARAM_VALUE', param_id='MOT_H_REV', param_value=1), 10.2)
        self.assertTrue(self.s.hold_motion('右移', 1450, 1550, 10.3))
        self.directions(21)
        self.modes.last_heartbeat = self.s.ack_time = 21
        self.s.tick(21)
        self.assertEqual(self.s.phase, 'stopping')
        self.assertIn('水平混控方向回传过期', self.s.notice)
        self.assertTrue(all(c[6] == 1500 for c in self.outputs()[-8:]))

    def test_foreign_vehicle_cannot_confirm_horizontal_geometry(self):
        self.s.observe(packet('PARAM_VALUE', system=2, param_id='MOT_H_REV', param_value=1), 10.2)
        self.assertNotIn('MOT_H_REV', self.s.params)

    def test_all_four_motion_plans_follow_mapping_and_readback_direction(self):
        self.ready()
        self.directions()
        expected = {
            "前进": (1517, 1482, 1517, 1482, 1500, 1500, 1500, 1500),
            "后退": (1482, 1517, 1482, 1517, 1500, 1500, 1500, 1500),
            "上升": (1500, 1500, 1500, 1500, 1517, 1482, 1482, 1517),
            "下降": (1500, 1500, 1500, 1500, 1482, 1517, 1517, 1482),
        }
        for action, pwm in expected.items():
            self.link.reset_mock()
            self.assertTrue(self.s.hold_motion(action, 1482, 1517, 10.3))
            self.s.tick(10.3)
            calls = self.outputs()
            self.assertEqual(tuple(c[6] for c in calls), pwm)
            self.assertEqual(tuple(c[4] for c in calls), tuple(i-1 for i in PHYSICAL_TO_MOTOR))
            self.assertFalse(self.s.start_single_pwm(1550, 10.31))
            self.s.release(10.32)
            self.assertTrue(all(c[6] == 1500 for c in self.outputs()[-8:]))

    def test_missing_stale_invalid_direction_blocks_group_without_outputs(self):
        self.ready()
        self.assertFalse(self.s.hold_motion("前进", 1490, 1510, 10.3))
        self.directions(now=0)
        self.assertFalse(self.s.hold_motion("前进", 1490, 1510, 10.3))
        self.directions()
        self.s.params["MOT_1_DIRECTION"] = 0
        self.assertFalse(self.s.hold_motion("前进", 1490, 1510, 10.3))
        self.assertEqual(self.outputs(), [])

    def test_descent_preserves_full_endpoints_and_returns_neutral_on_stop(self):
        self.ready()
        self.directions()
        self.assertTrue(self.s.hold_motion("下降", 1000, 2000, 10.3))
        self.s.tick(10.3)
        self.assertEqual(tuple(c[6] for c in self.outputs()), (1500,1500,1500,1500,1000,2000,2000,1000))
        self.s.release(10.4)
        self.assertTrue(all(c[6] == 1500 for c in self.outputs()[-8:]))

    def test_all_four_motions_preserve_requested_endpoints_and_direction_mapping(self):
        params = dict(zip(DIRECTION_PARAMS, (1,-1,-1,1,1,-1,-1,1)))
        expected = {
            "前进": (2000,1000,2000,1000,1500,1500,1500,1500),
            "后退": (1000,2000,1000,2000,1500,1500,1500,1500),
            "上升": (1500,1500,1500,1500,2000,1000,1000,2000),
            "下降": (1500,1500,1500,1500,1000,2000,2000,1000),
        }
        for action, values in expected.items():
            plan = motion_pwm_plan(action, 1000, 2000, params)
            self.assertEqual(tuple(plan.values()), values)
        self.assertEqual(motion_pwm_plan("下降",1500,1500,params), dict.fromkeys(range(1,9),1500))
        # The plan continues to use actual read-back polarities.
        params["MOT_5_DIRECTION"] = params["MOT_8_DIRECTION"] = -1
        inverted = motion_pwm_plan("下降",1000,2000,params)
        self.assertEqual((inverted[6],inverted[7]),(1000,1000))

    def test_single_exact_pwm_for_each_of_eight_physical_motors(self):
        self.ready()
        for physical in range(1,9):
            for requested in (2000,1600,1500,1400,1000):
                self.link.reset_mock()
                self.s.selected = physical
                self.assertTrue(self.s.start_single_pwm(requested,10.3))
                self.s.tick(10.3)
                outputs = [c[6] for c in self.outputs()]
                self.assertEqual(outputs[physical-1],requested)
                self.assertTrue(all(v == 1500 for i,v in enumerate(outputs,1) if i != physical))
                self.s.release(10.31)

    def test_test_mapping_matches_firmware_endpoint_mapping_for_integer_excursions(self):
        from control_profile import load_profile
        params = load_profile()["pixhawk_button_parameters"]
        self.assertEqual(params["MOT_PWM_MIN"],1000)
        self.assertEqual(params["MOT_PWM_MAX"],2000)
        directions = dict(zip(DIRECTION_PARAMS, (1,-1,-1,1,1,-1,-1,1)))
        for pwm in range(1000,2001):
            # Exact-rational equivalent of pinned AP_Motors6DOF calc_thrust_to_pwm.
            span = params["MOT_PWM_MAX"]-1500 if pwm >= 1500 else 1500-params["MOT_PWM_MIN"]
            expected = 1500 + (pwm-1500)*span//500
            plan = motion_pwm_plan("下降", min(pwm,1500), max(pwm,1500), directions)
            self.assertEqual(plan[6 if pwm >= 1500 else 5], expected)

    def test_group_preserves_every_requested_integer_excursion(self):
        params = dict.fromkeys(DIRECTION_PARAMS,1)
        for offset in range(501):
            plan = motion_pwm_plan("下降",1500-offset,1500+offset,params)
            self.assertTrue(all(plan[i] == 1500 for i in range(1,5)))
            self.assertTrue(all(plan[i] == 1500+offset for i in range(5,9)))

    def test_direction_change_during_group_stops_neutral_and_disarms(self):
        self.ready()
        self.directions()
        self.assertTrue(self.s.hold_motion("上升", 1490, 1510, 10.3))
        self.s.observe(packet("PARAM_VALUE", param_id="MOT_5_DIRECTION", param_value=-1), 10.4)
        self.assertIsNone(self.s.motion_action)
        self.assertEqual(self.s.phase, "stopping")
        self.assertTrue(all(c[6] == 1500 for c in self.outputs()[-8:]))

    def test_exact_single_pwm_validated_and_preserved_until_stop(self):
        self.ready()
        for value in (999, 2001, 1500.5, True, "1510"):
            self.assertFalse(self.s.start_single_pwm(value, 10.3))
        self.assertTrue(self.s.start_single_pwm(1513, 10.3))
        self.s.tick(10.3)
        self.assertEqual([c[6] for c in self.outputs()], [1513]+[1500]*7)
        self.s.release(10.4)
        self.assertIsNone(self.s.single_pwm)

    def test_invalid_motion_pwm_cannot_start(self):
        self.ready()
        self.directions()
        for low, high in ((999, 1510), (1501, 1510), (1490, 1499), (1490, 2001), (1490.5, 1510)):
            self.assertFalse(self.s.hold_motion("前进", low, high, 10.3))
        self.assertEqual(self.outputs(), [])

    def test_group_link_loss_and_ack_rejection_stop(self):
        for reason in ("link", "ack", "mode", "stale_direction"):
            self.setUp()
            self.ready()
            self.directions()
            self.s.hold_motion("前进", 1490, 1510, 10.3)
            if reason == "link":
                self.s.tick(14)
            elif reason == "ack":
                self.s.observe(packet("COMMAND_ACK", command=209, result=2), 10.4)
            elif reason == "mode":
                self.modes.mode = 2
                self.s.tick(10.4)
            else:
                self.modes.last_heartbeat = 21
                self.s.ack_time = 21
                self.s.tick(21)
                self.assertIn("方向参数回传过期", self.s.notice)
            self.assertEqual(self.s.phase, "stopping", reason)
            self.assertIsNone(self.s.motion_action)
            self.assertTrue(all(c[6] == 1500 for c in self.outputs()[-8:]))

    def test_group_runs_until_explicit_stop_with_fresh_feedback(self):
        self.ready()
        self.directions()
        self.assertTrue(self.s.hold_motion("上升", 1491, 1519, 10.3))
        for index in range(1, 151):
            now = 10.3 + index*.1
            self.modes.last_heartbeat = now
            self.directions(now)
            self.s.observe(packet("COMMAND_ACK", command=209, result=0), now)
            self.s.tick(now)
            self.assertEqual(self.s.phase, "ready")
            self.assertEqual(self.s.motion_action, "上升")
        self.s.release(now+.01)
        self.assertTrue(all(c[6] == 1500 for c in self.outputs()[-8:]))


class MotorDialogTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.owner = SimpleNamespace(root=self.root, motor_dialog=None)
        self.modes = FlightModeController()
        self.session = MotorTestSession(self.modes, Mock())
        self.dialog = MotorTestPanel(self.owner, self.session)
        self.owner.motor_dialog = self.dialog

    def tearDown(self):
        self.session.phase = "monitor"
        self.dialog.close()
        self.root.destroy()

    def test_initial_monitor_disables_motion_and_does_not_send(self):
        self.assertEqual(str(self.dialog.forward["state"]), "disabled")
        self.assertEqual(str(self.dialog.prepare["state"]), "disabled")
        self.session.link.mav.command_long_send.assert_not_called()

    def test_motion_categories_show_only_four_rows_without_hidden_controls(self):
        self.dialog.switch_mode('motion')
        for group, actions in MOTION_GROUPS.items():
            self.dialog.motion_group.set(group)
            self.dialog.switch_motion_group(group)
            self.root.update_idletasks()
            visible = {action for action, row in self.dialog.motion_rows.items()
                       if row.winfo_manager() == 'pack'}
            self.assertEqual(visible, set(actions))
            # Each group retains the existing compact, four-row area.
            area = self.dialog.motion_box
            for action in actions:
                button = self.dialog.motion_buttons[action]
                self.assertLessEqual(button.winfo_rooty()+button.winfo_height(),
                                     area.winfo_rooty()+area.winfo_height())
        self.session.link.mav.command_long_send.assert_not_called()

    def test_motion_category_change_stops_running_output_without_starting_another(self):
        now = time.monotonic()
        self.session.phase = 'ready'
        self.session.params = dict(EXPECTED_PARAMS)
        for name in DIRECTION_PARAMS:
            self.session.params[name] = 1
            self.session.direction_times[name] = now
        self.modes.mode, self.modes.armed = 19, True
        self.modes.last_heartbeat = now
        self.dialog.switch_mode('motion')
        self.dialog.press_motion('前进')
        self.assertEqual(self.session.motion_action, '前进')
        self.dialog.switch_motion_group('俯仰 / 横滚')
        self.assertIsNone(self.session.motion_action)
        self.assertTrue(all(c.args[6] == 1500
            for c in self.session.link.mav.command_long_send.call_args_list[-8:]))

    def test_startup_reference_only_fills_values_and_never_starts(self):
        self.dialog.apply_startup_reference()
        for pair in (self.dialog.single_values, *self.dialog.motion_values.values()):
            self.assertEqual(tuple(var.get() for var in pair), ("1450", "1550"))
        self.assertEqual(self.session.phase, "monitor")
        self.assertIsNone(self.session.motion_action)
        self.session.link.mav.command_long_send.assert_not_called()
        self.assertIn("常规控制补偿需自定义飞控固件", self.session.notice)

    def test_reference_prefill_stops_active_test_before_changing_values(self):
        now = time.monotonic()
        self.session.phase = "ready"
        self.session.params = dict(EXPECTED_PARAMS)
        self.modes.mode, self.modes.armed = 19, True
        self.modes.last_heartbeat = now
        self.dialog.press(1)
        self.dialog.apply_startup_reference()
        self.assertEqual(self.session.direction, 0)
        self.assertIsNone(self.session.single_pwm)
        self.assertTrue(all(c.args[6] == 1500 for c in self.session.link.mav.command_long_send.call_args_list[-8:]))

    def test_invalid_startup_reference_rejected_before_window_creation(self):
        for offset in (True, -1, 101, 50.5, "50", None):
            with self.assertRaises(ValueError):
                MotorTestPanel(self.owner, self.session, startup_offset_us=offset)

    def test_mouse_press_release_and_selection_stop(self):
        self.session.phase = "ready"
        self.session.params = dict(EXPECTED_PARAMS)
        self.modes.mode, self.modes.armed = 19, True
        self.modes.last_heartbeat = time.monotonic()
        self.dialog.refresh(time.monotonic())
        self.assertEqual(str(self.dialog.forward["state"]), "normal")
        self.dialog.press(1)
        self.assertEqual(self.session.direction, 1)
        self.dialog.select(3)
        self.assertEqual(self.session.direction, 0)
        self.assertEqual(self.session.selected, 3)
        self.dialog.press(-1)
        self.dialog.release()
        self.assertEqual(self.session.direction, 0)

    def test_close_waits_for_disarm(self):
        self.session.phase = "ready"
        self.modes.armed = True
        self.dialog.close()
        self.assertFalse(self.dialog.closed)
        self.assertEqual(self.session.phase, "stopping")
        self.modes.last_heartbeat = time.monotonic()
        self.modes.armed = False
        self.session.tick(time.monotonic())
        self.dialog.close()
        self.assertTrue(self.dialog.closed)
        self.assertIsNone(self.owner.motor_dialog)

    def test_single_click_starts_second_click_stops_without_mouse_hold(self):
        self.session.phase = "ready"
        self.session.params = dict(EXPECTED_PARAMS)
        self.modes.mode, self.modes.armed = 19, True
        self.modes.last_heartbeat = time.monotonic()
        self.dialog.single_values[1].set("1523")
        self.dialog.refresh(time.monotonic())
        self.dialog.forward.invoke()
        self.assertEqual(self.session.single_pwm, 1523)
        self.assertIn("停止", self.dialog.forward["text"])
        self.assertEqual(self.dialog.window.bind("<ButtonRelease-1>"), "")
        self.dialog.forward.invoke()
        self.assertEqual(self.session.direction, 0)
        self.assertIsNone(self.session.single_pwm)

    def test_motion_click_toggle_tab_change_and_input_edit_stop(self):
        now = time.monotonic()
        self.session.phase = "ready"
        self.session.params = dict(EXPECTED_PARAMS)
        for name in DIRECTION_PARAMS:
            self.session.params[name] = 1
            self.session.direction_times[name] = now
        self.modes.mode, self.modes.armed = 19, True
        self.modes.last_heartbeat = now
        self.dialog.switch_mode("motion")
        self.dialog.motion_buttons["前进"].invoke()
        self.assertEqual(self.session.motion_action, "前进")
        self.dialog.motion_buttons["前进"].invoke()
        self.assertIsNone(self.session.motion_action)
        self.dialog.motion_buttons["上升"].invoke()
        self.dialog.switch_mode("single")
        self.assertIsNone(self.session.motion_action)
        self.dialog.forward.invoke()
        self.dialog.apply_level()
        self.assertEqual(self.session.direction, 0)

    def test_descent_ui_preserves_both_endpoints_and_click_stops(self):
        now = time.monotonic()
        self.session.phase = "ready"
        self.session.params = dict(EXPECTED_PARAMS)
        for name in DIRECTION_PARAMS:
            self.session.params[name] = 1
            self.session.direction_times[name] = now
        self.modes.mode, self.modes.armed = 19, True
        self.modes.last_heartbeat = now
        self.dialog.switch_mode("motion")
        self.dialog.motion_values["下降"][0].set("1000")
        self.dialog.motion_values["下降"][1].set("2000")
        self.dialog.motion_buttons["下降"].invoke()
        self.assertEqual(tuple(self.session.motion_plan.values()), (1500,1500,1500,1500,2000,2000,2000,2000))
        self.assertFalse(hasattr(self.dialog, "experimental_compensation"))
        self.dialog.motion_buttons["下降"].invoke()
        self.assertIsNone(self.session.motion_action)
        self.assertTrue(all(c.args[6] == 1500 for c in self.session.link.mav.command_long_send.call_args_list[-8:]))
        self.dialog.motion_buttons["下降"].invoke()
        self.assertEqual(self.session.motion_plan[6],2000)


if __name__ == "__main__":
    unittest.main()
