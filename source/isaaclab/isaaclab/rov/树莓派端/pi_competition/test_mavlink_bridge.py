import unittest
from unittest.mock import Mock, patch

import rov_mavlink_bridge as bridge


class BridgeFairnessTests(unittest.TestCase):
    def test_udp_burst_has_budget_without_corrupting_payload(self):
        payload = b'\xfd\x00\x01\xff'  # arbitrary binary remains unchanged
        udp = Mock()
        udp.recvfrom.return_value = (payload, ('192.168.1.4', 12345))
        port = Mock()
        port.write.return_value = len(payload)
        with patch.object(bridge, 'RUNNING', True):
            self.assertEqual(bridge.forward_datagrams(udp, port, '192.168.1.4'), (128, 0))
        self.assertEqual(udp.recvfrom.call_count, 32)
        self.assertTrue(all(call.args == (payload,) for call in port.write.call_args_list))

    def test_unexpected_host_and_empty_packets_are_not_forwarded(self):
        udp = Mock()
        udp.recvfrom.side_effect = [(b'x', ('192.168.1.5', 1)),
                                    (b'', ('192.168.1.4', 1)), BlockingIOError()]
        port = Mock()
        with patch.object(bridge, 'RUNNING', True):
            self.assertEqual(bridge.forward_datagrams(udp, port, '192.168.1.4'), (0, 1))
        port.write.assert_not_called()

    def test_partial_serial_write_is_failure_not_success(self):
        udp = Mock()
        udp.recvfrom.return_value = (b'abcdef', ('192.168.1.4', 1))
        port = Mock()
        port.write.return_value = 3
        with patch.object(bridge, 'RUNNING', True), self.assertRaises(OSError):
            bridge.forward_datagrams(udp, port, '192.168.1.4')

    def test_stop_signal_prevents_further_writes(self):
        udp, port = Mock(), Mock()
        with patch.object(bridge, 'RUNNING', False):
            self.assertEqual(bridge.forward_datagrams(udp, port, '192.168.1.4'), (0, 0))
        udp.recvfrom.assert_not_called()
        port.write.assert_not_called()
