import json
import unittest
import zlib

import secure_protocol
from pc_gui import TelemetryLossWindow


KEY = "telemetry-test-key-0123456789abcdef"
SESSION = "a" * 32


def representative_telemetry():
    return {
        "armed": False,
        "cmd_enabled": False,
        "control_link_ok": True,
        "sensor_health": {
            "imu_valid": True,
            "depth_valid": True,
            "depth_age_ms": 12.5,
            "imu_age_ms": 3.2,
        },
        "imu": {
            "roll": 1.25,
            "pitch": -2.5,
            "yaw": 73.75,
            "heading_vehicle": 114.2,
            "yaw_control": 73.75,
        },
        "motors": {
            f"CH{index}": {
                "force_n": 0.0,
                "pwm_us": 1540,
                "saturated": False,
            }
            for index in range(8)
        },
        "pid": {
            axis: {
                "kp": 0.13 + index / 1000.0,
                "ki": 0.02,
                "kd": 0.08,
                "integral": 0.0,
                "error": index / 10.0,
                "output": 0.0,
                "limit": 60.0,
            }
            for index, axis in enumerate(
                ("depth", "roll", "pitch", "yaw", "speed")
            )
        },
        "diagnostics": [
            {
                "name": f"telemetry-field-{index:03d}",
                "value": f"sample-value-{index:03d}",
                "valid": index % 3 != 0,
            }
            for index in range(80)
        ],
    }


class TelemetryCompressionTests(unittest.TestCase):
    def test_large_telemetry_is_one_non_fragmented_udp_payload(self):
        payload = representative_telemetry()
        packet = secure_protocol.encode_telemetry_packet(
            payload, KEY, SESSION, 42
        )

        self.assertLessEqual(len(packet), 1400)
        self.assertEqual(
            secure_protocol.decode_telemetry_packet(packet, KEY).payload,
            payload,
        )

    def test_small_legacy_json_packet_remains_accepted(self):
        payload = {"armed": False, "depth": 0.52}
        packet = secure_protocol.encode_telemetry_packet(
            payload, KEY, SESSION, 7
        )

        self.assertTrue(packet.startswith(b"{"))
        self.assertEqual(
            secure_protocol.decode_telemetry_packet(packet, KEY).payload,
            payload,
        )

    def test_corrupted_compressed_packet_is_rejected(self):
        packet = bytearray(secure_protocol.encode_telemetry_packet(
            representative_telemetry(), KEY, SESSION, 9
        ))
        self.assertFalse(packet.startswith(b"{"))
        packet[-1] ^= 0x01

        with self.assertRaises(secure_protocol.ProtocolError):
            secure_protocol.decode_telemetry_packet(bytes(packet), KEY)

    def test_compressed_expansion_is_bounded(self):
        oversized_json = json.dumps({"value": "x" * 20000}).encode("utf-8")
        packet = b"RZ1\x00" + zlib.compress(oversized_json)

        with self.assertRaises(secure_protocol.ProtocolError):
            secure_protocol.decode_telemetry_packet(packet, KEY)


class TelemetryLossWindowTests(unittest.TestCase):
    def test_loss_is_recent_window_not_process_lifetime(self):
        window = TelemetryLossWindow(capacity=10)
        window.observe("session-a", 10)
        window.observe("session-a", 11)
        window.observe("session-a", 14)
        self.assertAlmostEqual(window.loss_percent(), 40.0)

        for seq in range(15, 21):
            window.observe("session-a", seq)
        self.assertAlmostEqual(window.loss_percent(), 20.0)

    def test_new_session_resets_loss_and_large_gap_is_bounded(self):
        window = TelemetryLossWindow(capacity=10)
        window.observe("session-a", 1)
        window.observe("session-a", 100)
        self.assertAlmostEqual(window.loss_percent(), 90.0)

        window.observe("session-b", 0)
        self.assertAlmostEqual(window.loss_percent(), 0.0)


if __name__ == "__main__":
    unittest.main()
