import configparser
import unittest
from pathlib import Path

from pi_runtime.thrust_allocator import THRUSTER_NAMES, ThrustAllocator


ROOT = Path(__file__).resolve().parent


def configured_allocator():
    config = configparser.ConfigParser()
    config.read(ROOT / "config.ini", encoding="utf-8")
    thrusters = config["thrusters"]
    signs = dict(zip(
        THRUSTER_NAMES,
        (int(value.strip()) for value in thrusters["motor_signs"].split(",")),
    ))
    channels = dict(zip(
        THRUSTER_NAMES,
        (int(value.strip()) for value in thrusters["channels"].split(",")),
    ))
    positions = {
        name: tuple(float(value) for value in thrusters[f"{name}_position_m"].split(","))
        for name in THRUSTER_NAMES[4:]
    }
    directions = {
        name: tuple(float(value) for value in thrusters[f"{name}_direction_body"].split(","))
        for name in THRUSTER_NAMES[4:]
    }
    return ThrustAllocator(
        enable_hardware=False,
        channels_dict=channels,
        motor_signs=signs,
        vertical_positions_m=positions,
        vertical_directions_body=directions,
    )


class VerticalForceSignTests(unittest.TestCase):
    def test_confirmed_negative_motor_command_is_negative_body_fz(self):
        allocator = configured_allocator()
        result = allocator.allocate(tau_z=-4.0, write_output=False)

        self.assertTrue(all(value < 0.0 for value in result.motor_thrusts[4:8]))
        self.assertAlmostEqual(result.achieved_tau[2], -4.0, places=6)

    def test_positive_roll_moment_uses_left_up_and_right_down(self):
        allocator = configured_allocator()
        result = allocator.allocate(M_x=0.4, write_output=False)
        fl, fr, rl, rr = result.motor_thrusts[4:8]

        self.assertLess(fl, 0.0)
        self.assertGreater(fr, 0.0)
        self.assertLess(rl, 0.0)
        self.assertGreater(rr, 0.0)
        self.assertAlmostEqual(result.achieved_tau[3], 0.4, places=6)

    def test_positive_pitch_moment_uses_front_up_and_rear_down(self):
        allocator = configured_allocator()
        result = allocator.allocate(M_y=0.4, write_output=False)
        fl, fr, rl, rr = result.motor_thrusts[4:8]

        self.assertLess(fl, 0.0)
        self.assertLess(fr, 0.0)
        self.assertGreater(rl, 0.0)
        self.assertGreater(rr, 0.0)
        self.assertAlmostEqual(result.achieved_tau[4], 0.4, places=6)


if __name__ == "__main__":
    unittest.main()
