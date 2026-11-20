import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_allocator_class():
    module_name = "test_isolated_thrust_allocator"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "pi_runtime" / "thrust_allocator.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.ThrustAllocator


class ThrusterStartCompensationTests(unittest.TestCase):
    def test_all_thrusters_jump_past_measured_start_deadbands(self):
        ThrustAllocator = load_allocator_class()
        allocator = ThrustAllocator(
            enable_hardware=False,
            reverse_start_us=1532,
            forward_start_us=1578,
        )

        self.assertEqual(allocator.thrust_to_pwm(0.0, "lateral_fl"), 1540)
        self.assertEqual(allocator.thrust_to_pwm(-0.001, "lateral_fl"), 1532)
        self.assertEqual(allocator.thrust_to_pwm(0.001, "lateral_fl"), 1578)
        self.assertEqual(allocator.thrust_to_pwm(-0.001, "vertical_fl"), 1532)
        self.assertEqual(allocator.thrust_to_pwm(0.001, "vertical_fl"), 1578)


if __name__ == "__main__":
    unittest.main()
