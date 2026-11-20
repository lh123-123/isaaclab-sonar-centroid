import unittest
import socket
from dataclasses import replace

from flydigi_apex5 import Apex5State
from launch_competition import panel_arguments
from rov_gamepad_bridge import command_neutral_for_arm, open_mavlink, send_servo_pwm
from runtime_support import ResumeGate, SingleInstance, neutral_input


class StartupTests(unittest.TestCase):
    def test_startup_cannot_silently_enable_sending_or_arm(self):
        settings = dict(pi_ip="127.0.0.1", control_port=14550, telemetry_port=14551,
                        target_system=1, rate_hz=25, send=True, allow_arm=True)
        args = panel_arguments(settings)
        self.assertFalse(args.send)
        self.assertFalse(args.allow_arm)
        self.assertTrue(args.receive)

    def test_single_instance_release_and_collision(self):
        first = SingleInstance(0)
        port = first.socket.getsockname()[1]
        try:
            with self.assertRaises(RuntimeError):
                SingleInstance(port)
        finally:
            first.close()
        second = SingleInstance(port)
        second.close()

    def test_returning_link_with_held_stick_cannot_resume(self):
        gate = ResumeGate()
        neutral = Apex5State.neutral()
        moved = replace(neutral, left_y=0.7)
        self.assertFalse(gate.update(neutral, True, 1))
        self.assertTrue(gate.update(neutral, True, 1.6))
        self.assertTrue(gate.update(moved, True, 2))
        self.assertFalse(gate.update(moved, False, 3))
        self.assertFalse(gate.update(moved, True, 4))
        self.assertFalse(gate.update(neutral, True, 5))
        self.assertTrue(gate.update(neutral, True, 5.6))

    def test_buttons_and_cancelled_triggers_still_prevent_neutral_gate(self):
        neutral = Apex5State.neutral()
        self.assertFalse(neutral_input(replace(neutral, buttons=1)))
        self.assertFalse(neutral_input(replace(neutral, left_trigger=1, right_trigger=1)))
        self.assertTrue(command_neutral_for_arm(replace(neutral, buttons=1)))
        self.assertFalse(command_neutral_for_arm(replace(neutral, buttons=1, right_y=0.5)))

    def test_control_sender_can_share_qgc_destination_port(self):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1)
        port = receiver.getsockname()[1]
        sender = None
        try:
            sender = open_mavlink("127.0.0.1", port)
            sender.write(b"probe")
            payload, _source = receiver.recvfrom(32)
            self.assertEqual(payload, b"probe")
            self.assertNotEqual(sender.socket.getsockname()[1], port)
        finally:
            if sender is not None:
                sender.close()
            receiver.close()

    def test_direct_servo_command_uses_aux_channel_and_raw_pwm(self):
        from unittest.mock import Mock

        link = Mock()
        send_servo_pwm(link, 1, 9, 500)
        args = link.mav.command_long_send.call_args.args
        self.assertEqual(args[0], 1)
        self.assertEqual(args[4:6], (9.0, 500.0))


if __name__ == "__main__":
    unittest.main()
