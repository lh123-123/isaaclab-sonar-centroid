import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pi_runtime"))

from controller import Controller  # noqa: E402
from pc_udp_client import PCClient  # noqa: E402
from secure_protocol import ProtocolError, validate_control_payload  # noqa: E402


class HeadingProtocolDisableTests(unittest.TestCase):
    def test_protocol_retains_heading_field_but_only_accepts_false(self):
        payload = {
            "mode": {
                "heading_hold": False,
                "depth_hold": True,
                "speed_hold": False,
            }
        }
        self.assertEqual(validate_control_payload(payload), payload)

        payload["mode"]["heading_hold"] = True
        with self.assertRaisesRegex(ProtocolError, "temporarily disabled"):
            validate_control_payload(payload)

    def test_pc_and_pi_protocol_mirrors_are_byte_identical(self):
        self.assertEqual(
            (ROOT / "secure_protocol.py").read_bytes(),
            (ROOT / "pi_runtime" / "secure_protocol.py").read_bytes(),
        )


class HeadingControllerDisableTests(unittest.TestCase):
    def test_controller_rejects_dict_and_legacy_true_modes(self):
        with self.assertRaisesRegex(ValueError, "temporarily disabled"):
            Controller.normalize_mode({"heading_hold": True})
        with self.assertRaisesRegex(ValueError, "temporarily disabled"):
            Controller.normalize_mode([True, False, False])

        self.assertEqual(
            Controller.normalize_mode({"depth_hold": True}),
            {
                "heading_hold": False,
                "depth_hold": True,
                "speed_hold": False,
            },
        )

    def test_disabled_heading_target_never_generates_yaw_moment(self):
        controller = Controller(config_path=ROOT / "config.ini")
        output, inhibited = controller.compute_control(
            current_state={"heading": -90.0, "depth": 0.0},
            mode={
                "heading_hold": False,
                "depth_hold": False,
                "speed_hold": False,
            },
            target_state={"heading_deg": 90.0},
            sensor_valid={"imu": True, "depth": True},
        )
        self.assertEqual(output[5], 0.0)
        self.assertEqual(inhibited, [])


class DebugClientDisableTests(unittest.TestCase):
    def make_client(self):
        client = PCClient.__new__(PCClient)
        client.sent_payloads = []
        client._send_payload = client.sent_payloads.append
        return client

    def test_debug_client_forces_explicit_false_and_has_no_heading_target(self):
        client = self.make_client()
        client.send_command(
            enable=True,
            mode={"depth_hold": True, "speed_hold": False},
            target={"depth_m": 0.5, "speed_mps": 0.0},
        )
        self.assertEqual(
            client.sent_payloads,
            [{
                "enable": True,
                "mode": {
                    "depth_hold": True,
                    "speed_hold": False,
                    "heading_hold": False,
                },
                "target": {"depth_m": 0.5, "speed_mps": 0.0},
            }],
        )

    def test_debug_client_rejects_heading_mode_and_target(self):
        client = self.make_client()
        with self.assertRaisesRegex(ValueError, "temporarily disabled"):
            client.send_command(mode={"heading_hold": True})
        with self.assertRaisesRegex(ValueError, "heading target"):
            client.send_command(target={"heading_deg": 10.0})
        self.assertEqual(client.sent_payloads, [])


if __name__ == "__main__":
    unittest.main()
