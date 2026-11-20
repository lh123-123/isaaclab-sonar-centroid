import tkinter as tk
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flydigi_apex5 import Apex5State
from rov_control_panel import ControlPanel


class PanelTests(unittest.TestCase):
    def test_capture_level_uses_native_targets_not_display_zero(self):
        now = time.monotonic()
        self.feed_attitude(now)
        self.panel.modes.mode = 19
        self.panel.level_target.actual = (0, 0)
        self.panel.receiver = Mock()
        for i in range(11):
            self.panel.level_target.samples.append((now - 1 + .1*i, (5, -3), (0, 0, 0)))
        with patch('rov_control_panel.open_mavlink') as opener, \
             patch('rov_control_panel.time.monotonic', return_value=now):
            self.panel.capture_level_target()
            opener.return_value.mav.command_int_send.assert_called_once()
            opener.return_value.mav.param_set_send.assert_not_called()
        self.panel.refresh_attitude(now)
        self.assertIn('横滚 +5.0°', self.panel.attitude_var.get())
        self.assertEqual(self.panel.level_target.phase, 'capture')
        with patch('rov_control_panel.open_mavlink') as opener:
            self.panel.toggle_control()
            self.panel.open_motor_test()
            opener.assert_not_called()
        self.panel.level_target.fail('test finished')
        self.panel.level_target_link.close()
        self.panel.level_target_link = None

    def feed_attitude(self, now):
        self.panel.modes.last_heartbeat = now
        self.panel.modes.armed = False
        for i in range(8):
            self.panel.modes.attitude_display.observe((5, -3, 175), now - .7 + .1 * i,
                                                       disarmed=True)

    def test_attitude_card_shows_absolute_without_transmitting(self):
        now = time.monotonic()
        self.feed_attitude(now)
        self.panel.refresh_attitude(now)
        self.assertIn('航向 175.0°', self.panel.attitude_var.get())
        self.assertIn('横滚 +5.0°', self.panel.attitude_var.get())
        self.panel.link = Mock()
        with patch('rov_control_panel.time.monotonic', return_value=now):
            self.panel.zero_attitude_display()
        self.panel.link.mav.assert_not_called()
        self.assertEqual(self.panel.link.mock_calls, [])
        self.assertIn('横滚 +5.0°', self.panel.attitude_var.get())
        self.panel.link = None

    def test_actual_angles_expire_and_no_display_zero_button(self):
        now = time.monotonic()
        self.feed_attitude(now)
        reference = self.panel.modes.attitude_display.reference
        self.panel.modes.armed = True
        self.panel.refresh_attitude(now)
        self.assertFalse(hasattr(self.panel, 'zero_display_button'))
        with patch('rov_control_panel.time.monotonic', return_value=now):
            self.panel.zero_attitude_display()
        self.assertEqual(reference, self.panel.modes.attitude_display.reference)
        self.panel.refresh_attitude(now + 4)
        self.assertIn('未知', self.panel.attitude_var.get())

    def test_status_column_has_scrollbar_for_all_controls(self):
        self.root.update_idletasks()
        canvas = self.panel.status_canvas
        self.assertTrue(canvas.cget('yscrollcommand'))
        self.assertTrue(canvas.cget('scrollregion'))
        self.assertGreater(canvas.bbox('all')[3], 650)

    def test_compact_button_guide_matches_profile(self):
        guide = self.panel.button_guide_text
        self.assertIn('左摇杆', guide)
        self.assertIn('RT  上升', guide)
        for index in (0, 1, 2, 3, 4, 6, 9, 10, 11, 12, 13, 14,
                      15, 16, 17, 18, 19, 20):
            meta = self.panel.profile['buttons'][str(index)]
            self.assertIn(meta['name'], guide)
            self.assertIn(meta['action'], guide)

    def test_full_panel_also_contains_read_only_attitude_card(self):
        self.panel.close()
        self.root = tk.Tk()
        self.root.withdraw()
        self.args.compact = False
        self.panel = ControlPanel(self.root, self.args)
        now = time.monotonic()
        self.feed_attitude(now)
        self.panel.refresh_attitude(now)
        self.root.update_idletasks()
        self.assertIn('航向 175.0°', self.panel.attitude_var.get())
        self.assertTrue(self.panel.status_canvas.cget('scrollregion'))

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.args = SimpleNamespace(send=False, allow_arm=False, receive=False, compact=True,
                                    profile=None, pi_ip="127.0.0.1", port=14550,
                                    telemetry_port=14551, target_system=1, rate=25)
        self.panel = ControlPanel(self.root, self.args)

    def tearDown(self):
        self.panel.modes.armed = False
        self.panel.close()

    def test_no_controller_keeps_window_open_without_sending(self):
        with patch("rov_control_panel.open_controller", side_effect=RuntimeError("absent")), \
             patch("rov_control_panel.open_mavlink") as sender:
            self.panel.poll()
            self.assertIsNone(self.panel.controller)
            self.assertFalse(sender.called)
            self.assertFalse(self.panel.closing)
            self.assertIn("未连接", self.panel.status_var.get())

    def test_fly_snapshot_is_preview_only_without_control_link(self):
        self.panel.fly_status_server = Mock()
        self.panel.controller = Mock()
        self.panel.controller.read.return_value = []
        self.panel.next_acquire = float('inf')
        self.panel.modes.mode = 2
        self.panel.last_input = self.panel.modes.last_heartbeat = 10
        with patch('rov_control_panel.time.monotonic', return_value=10), \
             patch('rov_control_panel.send_command') as sender:
            self.panel.poll()
        sender.assert_not_called()
        snapshot = self.panel.fly_status_server.publish.call_args[0][0]
        self.assertFalse(snapshot['sending'])
        self.assertFalse(snapshot['allow_arm'])
        self.assertTrue(snapshot['depth_hold'])
        self.assertTrue(snapshot['attitude_hold'])

    def test_monitor_requests_power_stream_without_sending_motion(self):
        now = time.monotonic()
        self.panel.receiver = Mock()
        self.panel.controller = Mock()
        self.panel.controller.read.return_value = []
        self.panel.next_acquire = float('inf')
        self.panel.modes.last_heartbeat = now
        self.panel.modes.mode = 19
        self.panel.last_input = now
        self.panel.level_target.actual = (0, 0)
        self.panel.attitude_tolerance.actual = 0.5
        with patch('rov_control_panel.time.monotonic', return_value=now), \
             patch('rov_control_panel.poll_telemetry'), \
             patch('rov_control_panel.open_mavlink') as opener, \
             patch('rov_control_panel.request_power_telemetry') as request, \
             patch('rov_control_panel.send_command') as motion:
            self.panel.poll()
            self.panel.poll()
        request.assert_called_once_with(opener.return_value, 1)
        motion.assert_not_called()
        self.assertIsNone(self.panel.link)

    def test_enable_requires_fresh_link_disarmed_neutral(self):
        with patch("rov_control_panel.open_mavlink") as sender:
            self.panel.toggle_control()
            sender.assert_not_called()
            self.panel.last_input = self.panel.modes.last_heartbeat = time.monotonic()
            self.panel.modes.armed = True
            self.panel.toggle_control()
            sender.assert_not_called()
            self.panel.modes.armed = False
            self.panel.state = replace(Apex5State.neutral(), left_y=0.5)
            self.panel.toggle_control()
            sender.assert_not_called()
            self.panel.state = Apex5State.neutral()
            self.panel.toggle_control()
            sender.assert_called_once()
            self.assertFalse(self.panel.args.allow_arm)
            self.assertFalse(self.panel.resume.ready)

    def test_startup_manual_reset_blocks_motion_until_vehicle_confirmation(self):
        self.panel.link = Mock()
        self.panel.controller = Mock()
        self.panel.controller.read.return_value = []
        self.panel.next_acquire = float('inf')
        self.panel.default_confirmed = False
        self.panel.resume.ready = True
        self.panel.modes.mode = 2
        self.panel.last_input = self.panel.modes.last_heartbeat = 10
        with patch('rov_control_panel.time.monotonic', return_value=10):
            self.panel.poll()
        self.assertEqual(self.panel.modes.pending, 19)
        self.panel.link.mav.manual_control_send.assert_not_called()
        self.assertFalse(self.panel.default_confirmed)
        self.panel.modes.mode, self.panel.modes.pending = 19, None
        with patch('rov_control_panel.time.monotonic', return_value=10.1):
            self.panel.poll()
        self.assertTrue(self.panel.default_confirmed)
        self.panel.link.mav.manual_control_send.assert_called_once()
        self.panel.link = None

    def test_disconnect_drops_commands_before_reconnect(self):
        device = Mock()
        device.read.side_effect = OSError("disconnected")
        self.panel.controller = device
        self.panel.next_acquire = float("inf")
        self.panel.link = Mock()
        self.panel.last_input = self.panel.modes.last_heartbeat = time.monotonic()
        self.panel.resume.ready = True
        self.panel.state = replace(Apex5State.neutral(), left_y=1)
        with patch("rov_control_panel.send_command") as send:
            self.panel.poll()
            send.assert_not_called()
        self.assertFalse(self.panel.resume.ready)
        self.assertIsNone(self.panel.controller)

    def test_hid_fifo_uses_latest_center_not_old_forward(self):
        forward = bytearray(32)
        forward[:4] = bytes((4, 0x5A, 0xA5, 0xEF))
        forward[6:8] = bytes((0, 0x80))  # physical forward, full stick
        center = bytearray(forward)
        center[6:8] = bytes((0, 0))
        self.panel.controller = Mock()
        self.panel.controller.read.side_effect = [bytes(forward)] * 10 + [bytes(center), []]
        self.panel.next_acquire = float('inf')
        self.panel.link = Mock()
        self.panel.default_confirmed = True
        self.panel.resume.ready = True
        self.panel.modes.last_heartbeat = 10
        with patch('rov_control_panel.time.monotonic', return_value=10), \
             patch('rov_control_panel.send_command') as send:
            self.panel.poll()
        self.assertEqual(self.panel.state.left_y, 0)
        self.assertEqual(send.call_args.args[2].x, 0)
        self.panel.link = None

    def test_undrained_hid_fifo_blocks_motion_and_requires_neutral_again(self):
        raw = bytearray(32)
        raw[:4] = bytes((4, 0x5A, 0xA5, 0xEF))
        raw[6:8] = bytes((0, 0x80))
        self.panel.controller = Mock()
        self.panel.controller.read.side_effect = [bytes(raw)] * 100 + [[]]
        self.panel.next_acquire = float('inf')
        self.panel.link = Mock()
        self.panel.default_confirmed = True
        self.panel.resume.ready = True
        self.panel.modes.last_heartbeat = 10
        with patch('rov_control_panel.time.monotonic', return_value=10), \
             patch('rov_control_panel.send_command') as send:
            self.panel.poll()
            send.assert_not_called()
        self.assertFalse(self.panel.resume.ready)
        self.assertEqual(self.panel.last_input, float('-inf'))
        with patch('rov_control_panel.time.monotonic', return_value=10.01), \
             patch('rov_control_panel.send_command') as send:
            self.panel.poll()
            send.assert_not_called()  # latest sample still forward, not neutral
        self.panel.link = None

    def test_disarm_during_fifo_overflow_is_sent_neutral_without_arm(self):
        raw = bytearray(32)
        raw[:4] = bytes((4, 0x5A, 0xA5, 0xEF))
        raw[6:8] = bytes((0, 0x80))
        raw[12] = 0x30  # simultaneous A and B; B must win
        self.panel.controller = Mock()
        self.panel.controller.read.side_effect = [bytes(raw)] * 100
        self.panel.next_acquire = float('inf')
        self.panel.link = Mock()
        self.panel.default_confirmed = True
        self.panel.resume.ready = True
        self.panel.modes.last_heartbeat = 10
        self.panel.args.send = self.panel.args.allow_arm = True
        with patch('rov_control_panel.time.monotonic', return_value=10), \
             patch('rov_control_panel.send_command') as send:
            self.panel.poll()
        value = send.call_args.args[2]
        self.assertEqual((value.x,value.y,value.z,value.r,value.s,value.buttons),
                         (0,0,500,0,0,2))
        self.assertEqual(self.panel.last_input, float('-inf'))
        self.panel.link = None

    def test_quick_disarm_does_not_wait_for_normal_send_slot(self):
        raw = bytearray(32)
        raw[:4] = bytes((4, 0x5A, 0xA5, 0xEF))
        raw[12] = 0x20
        center = bytearray(raw)
        center[12] = 0
        self.panel.controller = Mock()
        self.panel.controller.read.side_effect = [bytes(raw), bytes(center), []]
        self.panel.next_acquire = float('inf')
        self.panel.link = Mock()
        self.panel.default_confirmed = True
        self.panel.next_send = 10.04
        self.panel.modes.last_heartbeat = 10
        with patch('rov_control_panel.time.monotonic', return_value=10.01), \
             patch('rov_control_panel.send_command') as send:
            self.panel.poll()
        send.assert_called_once()
        self.assertEqual(send.call_args.args[2].buttons, 2)
        self.panel.link = None

    def test_depth_hold_release_is_immediate_in_actual_panel_poll(self):
        for mode in (2, 22):
            self.panel.motion_limiter.reset()
            self.panel.next_send = 0
            self.panel.controller = Mock()
            self.panel.controller.read.return_value = []
            self.panel.next_acquire = float('inf')
            self.panel.link = Mock()
            self.panel.default_confirmed = True
            self.panel.resume.ready = True
            self.panel.modes.mode = mode
            self.panel.last_input = self.panel.modes.last_heartbeat = 10
            self.panel.state = replace(Apex5State.neutral(), right_trigger=1)
            with patch('rov_control_panel.time.monotonic', return_value=10), \
                 patch('rov_control_panel.send_command'):
                self.panel.poll()
            with patch('rov_control_panel.time.monotonic', return_value=10.05), \
                 patch('rov_control_panel.send_command') as send:
                self.panel.poll()
            self.assertGreater(send.call_args.args[2].z, 500)
            self.panel.state = Apex5State.neutral()
            self.panel.next_send = 10.06
            with patch('rov_control_panel.time.monotonic', return_value=10.06), \
                 patch('rov_control_panel.send_command') as send:
                self.panel.poll()
            self.assertEqual(send.call_args.args[2].z, 500)
            self.panel.link = None

    def test_motor_dialog_blocks_gamepad_transport(self):
        self.panel.motor_dialog = Mock()
        with patch("rov_control_panel.open_mavlink") as sender:
            self.panel.toggle_control()
            sender.assert_not_called()
        self.assertIn("测试", self.panel.modes.notice)
        self.panel.motor_dialog = None

    def test_motor_dialog_observes_shared_receiver_and_ticks(self):
        dialog = Mock()
        self.panel.motor_dialog = dialog
        with patch("rov_control_panel.open_controller", side_effect=RuntimeError("absent")), \
             patch("rov_control_panel.poll_telemetry") as receive:
            self.panel.poll()
        self.assertIn(dialog.session, receive.call_args.args)
        dialog.session.tick.assert_called_once()
        dialog.refresh.assert_called_once()
        self.panel.motor_dialog = None


if __name__ == "__main__":
    unittest.main()
