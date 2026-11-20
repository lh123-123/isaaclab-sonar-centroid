import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pi_runtime"))

from config_loader import load_config  # noqa: E402
from controller import Controller  # noqa: E402
from secure_protocol import ProtocolError, validate_control_payload  # noqa: E402


class RuntimePidApplyTests(unittest.TestCase):
    def test_runtime_apply_does_not_write_config_but_explicit_save_does(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.ini"
            shutil.copyfile(ROOT / "config.ini", config_path)
            baseline = load_config(config_path)
            depth_kp = baseline.getfloat("pid", "depth_kp")
            yaw_kp = baseline.getfloat("pid", "yaw_kp")
            controller = Controller(config_path=config_path)

            controller.set_pid_runtime(1, (42.0, 0.0, 0.0))
            self.assertEqual(controller.get_pid_snapshot()["depth"]["kp"], 42.0)
            self.assertEqual(load_config(config_path).getfloat("pid", "depth_kp"), depth_kp)

            controller.write_pid_to_config(1, (42.0, 0.0, 0.0))
            self.assertEqual(load_config(config_path).getfloat("pid", "depth_kp"), 42.0)

            controller.set_pid_runtime(6, (0.25, 0.01, 0.09))
            self.assertEqual(controller.get_pid_snapshot()["yaw"]["kp"], 0.25)
            self.assertEqual(load_config(config_path).getfloat("pid", "yaw_kp"), yaw_kp)
            controller.write_pid_to_config(6, (0.25, 0.01, 0.09))
            self.assertEqual(load_config(config_path).getfloat("pid", "yaw_kp"), 0.25)


class PidSaveProtocolTests(unittest.TestCase):
    def test_protocol_accepts_explicit_pid_save_and_rejects_ambiguous_request(self):
        pid = {"type": 1, "kp": 35.0, "ki": 0.0, "kd": 0.0}
        self.assertEqual(validate_control_payload({"pid_save": pid}), {"pid_save": pid})
        with self.assertRaisesRegex(ProtocolError, "cannot be sent together"):
            validate_control_payload({"pid_set": pid, "pid_save": pid})


if __name__ == "__main__":
    unittest.main()
