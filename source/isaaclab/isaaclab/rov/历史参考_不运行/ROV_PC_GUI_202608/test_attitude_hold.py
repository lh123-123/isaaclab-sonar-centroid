import math
import importlib.util
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.modules.setdefault("serial", mock.Mock())
joystick_handler = types.ModuleType("joystick_handler")
joystick_handler.JoystickHandler = object
joystick_handler.NEUTRAL_JOYSTICK = {
    "tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0,
    "M_x": 0.0, "M_y": 0.0, "M_z": 0.0,
}
sys.modules.setdefault("joystick_handler", joystick_handler)
save = types.ModuleType("save")
save.is_saving = lambda: False
save.close_session = lambda: None
sys.modules.setdefault("save", save)

sys.path.insert(0, str(ROOT / "pi_runtime"))
from controller import Controller, PIDController
from attitude_math import (
    IDENTITY_QUATERNION,
    from_euler_deg,
    inverse as quaternion_inverse,
    multiply as quaternion_multiply,
    rotate_vector,
    to_euler_deg,
    vehicle_body_rate_from_raw,
)
from main import ROVController, tilt_compensated_hover_per_thruster
from imu_depth_combined import IMUReader
from config_loader import load_config

# Some older test modules install a minimal ``thrust_allocator`` stub before
# importing ``main``.  Load the real calculation-only allocator under a unique
# name so unittest discovery order cannot replace this mechanics seam.
_ALLOCATOR_SPEC = importlib.util.spec_from_file_location(
    "attitude_test_real_thrust_allocator",
    ROOT / "pi_runtime" / "thrust_allocator.py",
)
_ALLOCATOR_MODULE = importlib.util.module_from_spec(_ALLOCATOR_SPEC)
sys.modules[_ALLOCATOR_SPEC.name] = _ALLOCATOR_MODULE
_ALLOCATOR_SPEC.loader.exec_module(_ALLOCATOR_MODULE)
THRUSTER_NAMES = _ALLOCATOR_MODULE.THRUSTER_NAMES
ThrustAllocator = _ALLOCATOR_MODULE.ThrustAllocator


def clean_mode():
    return {"heading_hold": False, "depth_hold": False, "speed_hold": False}


def clean_target():
    return {"heading_deg": 0.0, "depth_m": 0.0, "speed_mps": 0.0}


class AttitudePidTests(unittest.TestCase):
    def make_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.ini"
            shutil.copyfile(ROOT / "config.ini", path)
            return Controller(config_path=path)

    def test_enable_edge_is_zero_output_without_derivative_kick(self):
        controller = self.make_controller()
        controller.roll_pid_params = (0.0, 0.0, 10.0)
        controller.roll_controller.set_pid(controller.roll_pid_params)
        controller.roll_controller.set_output_limit(20.0)
        output, _ = controller.compute_control(
            {"roll": 25.0, "pitch": 0.0}, clean_mode(), clean_target(),
            {"imu": True, "depth": False},
            attitude_hold={"roll": True, "pitch": False},
        )
        self.assertEqual(output[3], 0.0)
        controller.roll_controller.last_time -= 0.11
        output, _ = controller.compute_control(
            {"roll": 25.0, "pitch": 0.0}, clean_mode(), clean_target(),
            {"imu": True, "depth": False},
            attitude_hold={"roll": True, "pitch": False},
        )
        self.assertEqual(output[3], 0.0)

    def test_angle_derivative_crosses_wrap_by_two_degrees_not_358(self):
        pid = PIDController(0.0, 0.0, 1.0, sample_time=0.0, output_limit=20.0)
        pid.prime_angle(0.0, 179.0)
        pid.last_time -= 1.0
        output = pid.update_angle(0.0, -179.0)
        self.assertAlmostEqual(output, -2.0, delta=0.05)

    def test_attitude_integral_guard_tracks_moment_limit(self):
        pid = PIDController(0.0, 1.0, 0.0, sample_time=0.0, output_limit=20.0)
        pid.integral = 15.0
        pid.set_output_limit(2.0)
        self.assertEqual(pid.windup_guard, 2.0)
        self.assertEqual(pid.integral, 2.0)

    def test_runtime_tuning_cannot_reverse_feedback_with_a_negative_gain(self):
        controller = self.make_controller()
        for pid_type in (4, 5):
            with self.subTest(pid_type=pid_type):
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    controller.set_pid_runtime(pid_type, (-0.01, 0.0, 0.0))

    def test_switches_are_independent_and_target_calibrated_zero(self):
        controller = self.make_controller()
        controller.roll_controller.sample_time = 0.0
        controller.pitch_controller.sample_time = 0.0
        output, _ = controller.compute_control(
            {"roll": 8.0, "pitch": -6.0}, clean_mode(), clean_target(),
            {"imu": True, "depth": False},
            attitude_hold={"roll": False, "pitch": True},
        )
        self.assertEqual(output[3], 0.0)
        self.assertGreater(output[4], 0.0)

    def test_invalid_imu_clears_loop_and_zeroes_moments(self):
        controller = self.make_controller()
        controller.roll_controller.sample_time = 0.0
        controller.compute_control(
            {"roll": 8.0, "pitch": 0.0}, clean_mode(), clean_target(),
            {"imu": True, "depth": False},
            attitude_hold={"roll": True, "pitch": False},
        )
        output, inhibited = controller.compute_control(
            {"roll": 8.0, "pitch": 0.0}, clean_mode(), clean_target(),
            {"imu": False, "depth": False},
            attitude_hold={"roll": True, "pitch": False},
        )
        self.assertEqual(output[3:5], (0.0, 0.0))
        self.assertIn("roll_stale", inhibited)
        self.assertFalse(controller._roll_hold_active)

    def test_large_tilt_keeps_requested_attitude_holds_active(self):
        controller = self.make_controller()
        output, inhibited = controller.compute_control(
            {"roll": 55.0, "pitch": 65.0, "gyro_x": 0.0, "gyro_y": 0.0},
            clean_mode(), clean_target(),
            {"imu": True, "depth": False},
            attitude_hold={"roll": True, "pitch": True},
        )
        self.assertNotIn("attitude_tilt_limit", inhibited)
        self.assertEqual(output[3:5], (0.0, 0.0))
        self.assertTrue(controller._roll_hold_active)
        self.assertTrue(controller._pitch_hold_active)

    def test_attitude_moment_grows_with_error_instead_of_staying_fixed(self):
        cases = {
            "roll": (3, {"roll": True, "pitch": False}),
            "pitch": (4, {"roll": False, "pitch": True}),
        }
        for axis, (output_index, hold) in cases.items():
            with self.subTest(axis=axis):
                moments = []
                for angle_deg in (5.0, 20.0):
                    controller = self.make_controller()
                    controller.roll_controller.sample_time = 0.0
                    controller.pitch_controller.sample_time = 0.0
                    state = {"roll": 0.0, "pitch": 0.0}
                    state[axis] = angle_deg
                    output, _ = controller.compute_control(
                        state, clean_mode(), clean_target(),
                        {"imu": True, "depth": False}, attitude_hold=hold,
                    )
                    moments.append(output[output_index])

                self.assertLess(moments[0], 0.0)
                self.assertLess(moments[1], moments[0])
                self.assertGreater(abs(moments[1]), abs(moments[0]))

    def test_stationary_body_rate_does_not_turn_euler_sample_jitter_into_damping(self):
        """The D term must damp measured motion, not differentiate angle noise."""
        clock = [1000.0]
        with mock.patch("controller.time.monotonic", side_effect=lambda: clock[0]):
            controller = self.make_controller()
            outputs = []
            for roll in (0.0, 0.5, -0.5, 0.5, -0.5):
                output, _ = controller.compute_control(
                    {
                        "roll": roll,
                        "pitch": 0.0,
                        "gyro_x": 0.0,
                        "gyro_y": 0.0,
                    },
                    clean_mode(),
                    clean_target(),
                    {"imu": True, "depth": False},
                    attitude_hold={"roll": True, "pitch": False},
                )
                outputs.append(output[3])
                clock[0] += 0.101

        # Angle quantization may create only the configured proportional plus
        # tiny integral response; it must never create a derivative kick.
        proportional_bound = controller.roll_controller.Kp * (
            0.5 - controller.roll_controller.error_deadband_deg
        )
        self.assertLess(
            max(abs(value) for value in outputs), proportional_bound + 0.01
        )

    def test_integral_rejects_a_constant_unknown_bias_moment(self):
        """A cable/CG/CB bias must not become permanent attitude error."""
        clock = [1000.0]
        with mock.patch("controller.time.monotonic", side_effect=lambda: clock[0]):
            controller = self.make_controller()
            allocator = AttitudeAllocationTests.make_allocator(vertical_fraction=0.3)
            angles = [0.0, 0.0]
            rates = [0.0, 0.0]
            disturbance = [0.08, -0.06]

            for _ in range(1000):
                output, _ = controller.compute_control(
                    {
                        "roll": math.degrees(angles[0]),
                        "pitch": math.degrees(angles[1]),
                        "gyro_x": math.degrees(rates[0]),
                        "gyro_y": math.degrees(rates[1]),
                    },
                    clean_mode(),
                    clean_target(),
                    {"imu": True, "depth": False},
                    attitude_hold={"roll": True, "pitch": True},
                )
                allocation = allocator.allocate(
                    tau_z=4.0,
                    M_x=output[3],
                    M_y=output[4],
                    write_output=False,
                )
                dt = 0.1
                for axis, moment_index, inertia in ((0, 3, 0.8), (1, 4, 1.2)):
                    acceleration = (
                        allocation.achieved_tau[moment_index]
                        + disturbance[axis]
                        - 0.8 * rates[axis]
                    ) / inertia
                    rates[axis] += acceleration * dt
                    angles[axis] += rates[axis] * dt
                clock[0] += 0.101

        self.assertLess(abs(math.degrees(angles[0])), 0.25)
        self.assertLess(abs(math.degrees(angles[1])), 0.25)


class AttitudeAllocationTests(unittest.TestCase):
    @staticmethod
    def make_allocator(vertical_fraction=0.1):
        return ThrustAllocator(
            enable_hardware=False,
            motor_signs={name: 1 for name in THRUSTER_NAMES},
            vertical_x_m=0.0625,
            vertical_y_m=0.1775,
            horizontal_limit_fraction=0.1,
            vertical_limit_fraction=vertical_fraction,
        )

    def test_dual_axis_desaturation_preserves_vertical_baseline_and_moment_ratio(self):
        allocator = self.make_allocator()
        requested_fz = 16.0
        requested_roll = 1.0
        requested_pitch = 0.3

        result = allocator.allocate(
            tau_z=requested_fz,
            M_x=requested_roll,
            M_y=requested_pitch,
            write_output=False,
        )

        self.assertAlmostEqual(result.achieved_tau[2], requested_fz, places=6)
        roll_scale = result.achieved_tau[3] / requested_roll
        pitch_scale = result.achieved_tau[4] / requested_pitch
        self.assertGreater(roll_scale, 0.0)
        self.assertLess(roll_scale, 1.0)
        self.assertAlmostEqual(roll_scale, pitch_scale, places=6)

    def test_roll_pitch_and_dual_hold_reduce_error_in_a_simple_plant(self):
        cases = {
            "roll_only": ({"roll": True, "pitch": False}, 15.0, 0.0),
            "pitch_only": ({"roll": False, "pitch": True}, 0.0, -12.0),
            "dual": ({"roll": True, "pitch": True}, 15.0, -12.0),
        }
        mode = clean_mode()
        target = clean_target()

        for name, (hold, initial_roll, initial_pitch) in cases.items():
            with self.subTest(name=name):
                clock = [1000.0]
                with mock.patch(
                    "controller.time.monotonic", side_effect=lambda: clock[0]
                ):
                    controller = AttitudePidTests().make_controller()
                    allocator = self.make_allocator(vertical_fraction=0.3)
                    angles = {
                        "roll": math.radians(initial_roll),
                        "pitch": math.radians(initial_pitch),
                    }
                    rates = {"roll": 0.0, "pitch": 0.0}
                    inertias = {"roll": 0.8, "pitch": 1.2}
                    damping = {"roll": 0.8, "pitch": 0.8}
                    initial = {axis: abs(value) for axis, value in angles.items()}
                    peak = dict(initial)
                    first_active_moment = None

                    for _ in range(200):
                        output, _ = controller.compute_control(
                            {
                                "roll": math.degrees(angles["roll"]),
                                "pitch": math.degrees(angles["pitch"]),
                            },
                            mode,
                            target,
                            {"imu": True, "depth": False},
                            attitude_hold=hold,
                        )
                        allocation = allocator.allocate(
                            tau_z=4.0,
                            M_x=output[3],
                            M_y=output[4],
                            write_output=False,
                        )
                        if first_active_moment is None and (
                            abs(output[3]) > 1e-9 or abs(output[4]) > 1e-9
                        ):
                            first_active_moment = (output[3], output[4])

                        dt = 0.1
                        for axis, moment_index in (("roll", 3), ("pitch", 4)):
                            acceleration = (
                                allocation.achieved_tau[moment_index]
                                - damping[axis] * rates[axis]
                            ) / inertias[axis]
                            rates[axis] += acceleration * dt
                            angles[axis] += rates[axis] * dt
                            peak[axis] = max(peak[axis], abs(angles[axis]))
                        clock[0] += 0.101

                self.assertIsNotNone(first_active_moment)
                for axis, moment_index in (("roll", 0), ("pitch", 1)):
                    if hold[axis]:
                        initial_angle = initial_roll if axis == "roll" else initial_pitch
                        self.assertLess(
                            initial_angle * first_active_moment[moment_index], 0.0
                        )
                        self.assertLess(abs(angles[axis]), 0.1 * initial[axis])
                        self.assertLessEqual(peak[axis], initial[axis] * 1.001)

    def test_each_vertical_thruster_uses_its_configured_r_cross_force_column(self):
        positions = {
            "vertical_fl": (0.0625, -0.1775, 0.0),
            "vertical_fr": (0.0625, 0.1775, 0.0),
            "vertical_rl": (-0.0625, -0.1775, 0.0),
            "vertical_rr": (-0.0625, 0.1775, 0.0),
        }
        tilt = math.radians(8.0)
        directions = {
            name: (math.sin(tilt), 0.0, math.cos(tilt))
            for name in positions
        }
        allocator = ThrustAllocator(
            enable_hardware=False,
            motor_signs={name: 1 for name in THRUSTER_NAMES},
            vertical_positions_m=positions,
            vertical_directions_body=directions,
            horizontal_limit_fraction=0.1,
            vertical_limit_fraction=0.3,
        )
        result = allocator.allocate(
            tau_z=8.0, M_x=0.7, M_y=-0.25, write_output=False,
        )

        self.assertAlmostEqual(result.achieved_tau[2], 8.0, places=6)
        self.assertAlmostEqual(result.achieved_tau[3], 0.7, places=6)
        self.assertAlmostEqual(result.achieved_tau[4], -0.25, places=6)
        self.assertGreater(len({round(value, 6) for value in result.nominal_thrusts[4:8]}), 2)

    def test_world_vertical_component_accounts_for_vehicle_tilt(self):
        body_force = (0.0, 0.0, 12.0)
        level = _ALLOCATOR_MODULE.world_down_force_component(
            body_force, roll_deg=0.0, pitch_deg=0.0,
        )
        tilted = _ALLOCATOR_MODULE.world_down_force_component(
            body_force, roll_deg=30.0, pitch_deg=20.0,
        )
        self.assertAlmostEqual(level, 12.0, places=6)
        self.assertAlmostEqual(
            tilted,
            12.0 * math.cos(math.radians(30.0)) * math.cos(math.radians(20.0)),
            places=6,
        )

    def test_coupled_tilted_body_reduces_geometric_energy_without_yaw_lock(self):
        """Exercise controller + exact allocator + coupled 3-D rigid-body plant."""
        def rotation_matrix(roll_deg, pitch_deg, yaw_deg):
            roll, pitch, yaw = map(
                math.radians, (roll_deg, pitch_deg, yaw_deg)
            )
            cr, sr = math.cos(roll), math.sin(roll)
            cp, sp = math.cos(pitch), math.sin(pitch)
            cy, sy = math.cos(yaw), math.sin(yaw)
            return np.asarray((
                (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                (-sp, cp * sr, cp * cr),
            ))

        def euler_from_matrix(matrix):
            pitch = math.asin(max(-1.0, min(1.0, -matrix[2, 0])))
            roll = math.atan2(matrix[2, 1], matrix[2, 2])
            return math.degrees(roll), math.degrees(pitch)

        def exp_rotation(vector):
            angle = float(np.linalg.norm(vector))
            if angle < 1e-12:
                return np.eye(3)
            axis = vector / angle
            x, y, z = axis
            skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
            return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)

        clock = [1000.0]
        with mock.patch("controller.time.monotonic", side_effect=lambda: clock[0]):
            controller = AttitudePidTests().make_controller()
            controller.set_pid_runtime(4, (0.05, 0.0, 0.08))
            controller.set_pid_runtime(5, (0.05, 0.0, 0.08))
            controller.set_attitude_moment_limits(3.0, 3.0)
            allocator = self.make_allocator(vertical_fraction=0.3)

            orientation = rotation_matrix(25.0, -20.0, 47.0)
            body_rate = np.asarray((0.0, 0.0, 0.0))
            inertia = np.diag((0.8, 1.2, 1.4))
            inverse_inertia = np.linalg.inv(inertia)
            damping = np.diag((0.9, 0.9, 0.25))
            energies = []
            tilts = []

            for _ in range(500):
                roll, pitch = euler_from_matrix(orientation)
                output, _ = controller.compute_control(
                    {
                        "roll": roll,
                        "pitch": pitch,
                        "gyro_x": math.degrees(body_rate[0]),
                        "gyro_y": math.degrees(body_rate[1]),
                    },
                    clean_mode(), clean_target(),
                    {"imu": True, "depth": False},
                    attitude_hold={"roll": True, "pitch": True},
                )
                self.assertEqual(output[5], 0.0)
                allocation = allocator.allocate(
                    tau_z=4.0,
                    M_x=output[3],
                    M_y=output[4],
                    M_z=output[5],
                    write_output=False,
                )
                moment = allocation.achieved_tau[3:6]
                acceleration = inverse_inertia @ (
                    moment
                    - np.cross(body_rate, inertia @ body_rate)
                    - damping @ body_rate
                )
                dt = 0.02
                body_rate += acceleration * dt
                orientation = orientation @ exp_rotation(body_rate * dt)
                # Suppress integration drift without altering the physical
                # orientation represented by the nearest orthogonal matrix.
                left, _, right = np.linalg.svd(orientation)
                orientation = left @ right
                down_body = orientation.T @ np.asarray((0.0, 0.0, 1.0))
                tilt = math.acos(max(-1.0, min(1.0, down_body[2])))
                kinetic = 0.5 * float(body_rate @ inertia @ body_rate)
                stiffness_rad = 0.05 * 180.0 / math.pi
                energies.append(kinetic + 0.5 * stiffness_rad * tilt * tilt)
                tilts.append(tilt)
                clock[0] += dt

        self.assertLess(max(energies), energies[0] * 1.01)
        self.assertLess(energies[-1], energies[0] * 0.02)
        self.assertLess(tilts[-1], tilts[0] * 0.05)


class PiAttitudeStateTests(unittest.TestCase):
    def make_rov(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "config.ini"
        shutil.copyfile(ROOT / "config.ini", path)
        return ROVController(config_path=path)

    def test_all_vehicle_euler_angles_are_finite_and_wrapped(self):
        rov = self.make_rov()
        attitude = rov._referenced_attitude({
            "roll": 541.0, "pitch": 541.0, "yaw": -541.0,
        })
        self.assertTrue(all(-180.0 <= value < 180.0 for value in attitude.values()))

    def test_attitude_guard_accepts_normal_tilt_and_holds_large_single_frame_jump(self):
        rov = self.make_rov()
        rates = (0.0, 0.0, 0.0)
        first, accepted = rov._filter_attitude_sample(
            {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, rates, 1000.0
        )
        self.assertTrue(accepted)
        normal, accepted = rov._filter_attitude_sample(
            {"roll": 34.0, "pitch": -28.0, "yaw": 15.0}, rates, 1000.05
        )
        self.assertTrue(accepted)
        rejected, accepted = rov._filter_attitude_sample(
            {"roll": 154.0, "pitch": -28.0, "yaw": 15.0}, rates, 1000.10
        )
        self.assertFalse(accepted)
        self.assertEqual(rejected, normal)
        self.assertEqual(rov._attitude_fault_reason, "attitude_step_implausible")

    def test_runtime_stores_calibrated_vehicle_body_rates_not_raw_imu_axes(self):
        rov = self.make_rov()
        raw_rate = (3.0, -4.0, 5.0)
        expected = vehicle_body_rate_from_raw(
            raw_rate,
            rov.imu_mount_quaternion,
            rov._imu_axis_signs(),
            mount_calibrated=rov.imu_3d_mount_calibrated,
        )
        rov._update_imu_state(
            {
                "euler_deg": {"roll": 1.0, "pitch": 2.0, "yaw": 3.0},
                "gyro_dps": {"x": raw_rate[0], "y": raw_rate[1], "z": raw_rate[2]},
                "acc_mps2": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            time.monotonic(),
        )
        actual = tuple(rov.sensor_state[key] for key in ("gyro_x", "gyro_y", "gyro_z"))
        for value, wanted in zip(actual, expected):
            self.assertAlmostEqual(value, wanted, places=9)

    def test_enable_requires_armed_link_and_fresh_imu(self):
        rov = self.make_rov()
        with self.assertRaisesRegex(ValueError, "thrusters_locked"):
            rov._set_attitude_hold({"roll": True, "pitch": False})
        rov.thrusters_armed = True
        rov.link_timed_out = False
        rov._last_imu_update = time.monotonic()
        rov.attitude_control_mapping_validated = True
        self.assertEqual(
            rov._set_attitude_hold({"roll": True, "pitch": False}),
            {"roll": True, "pitch": False, "yaw": False},
        )

    def test_yaw_enable_captures_referenced_euler_yaw_not_magnetic_heading(self):
        rov = self.make_rov()
        rov.thrusters_armed = True
        rov.link_timed_out = False
        rov._last_imu_update = time.monotonic()
        rov._last_heading_update = rov._last_imu_update
        rov.attitude_control_mapping_validated = True
        rov.sensor_state["yaw"] = -44.0
        rov.sensor_state["yaw_control"] = 287.5

        rov._set_attitude_hold({"roll": False, "pitch": False, "yaw": True})

        self.assertEqual(rov.attitude_hold_target_deg["yaw"], -44.0)
        self.assertEqual(
            rov.build_telemetry_packet()["attitude_hold"]["yaw_target_deg"],
            -44.0,
        )

    def test_pi_converts_raw_magnetic_heading_into_the_control_heading(self):
        rov = self.make_rov()
        rov.imu_roll_inverted = False
        rov.imu_pitch_inverted = False
        rov.imu_yaw_inverted = False
        rov.imu_mount_quaternion = IDENTITY_QUATERNION
        rov.imu_reference_quaternion = IDENTITY_QUATERNION
        magnetic_angle = math.radians(72.5)
        rov._update_imu_state(
            {
                "euler_deg": {"roll": 0.0, "pitch": 0.0, "yaw": -44.0},
                "gyro_dps": {"x": 0.0, "y": 0.0, "z": 0.0},
                "acc_mps2": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mag_cal": {
                    "x": math.cos(magnetic_angle),
                    "y": math.sin(magnetic_angle),
                    "z": 0.4,
                },
            },
            time.monotonic(),
        )

        expected = (
            (-72.5 if rov.imu_heading_inverted else 72.5)
            + rov.imu_heading_offset_deg
        ) % 360.0
        self.assertAlmostEqual(rov.sensor_state["heading_vehicle"], expected, places=9)
        self.assertNotEqual(rov.sensor_state["heading_vehicle"], rov.sensor_state["yaw"])

    def test_vehicle_magnetic_yaw_is_invariant_to_tilt_and_diagonal_mount(self):
        rov = self.make_rov()
        rov.imu_heading_inverted = True
        rov.imu_heading_offset_deg = 0.0
        rov.imu_3d_mount_calibrated = True
        rov.imu_mount_quaternion = from_euler_deg(22.0, -17.0, 31.0)
        rov.imu_reference_quaternion = IDENTITY_QUATERNION
        world_magnetic_field = (1.0, 0.0, 0.4)

        def sample(roll, pitch, yaw, now):
            vehicle = from_euler_deg(roll, pitch, yaw)
            raw_orientation = quaternion_multiply(
                vehicle, quaternion_inverse(rov.imu_mount_quaternion)
            )
            corrected_raw_euler = to_euler_deg(raw_orientation)
            axis_signs = rov._imu_axis_signs()
            raw_euler = {
                axis: corrected_raw_euler[axis] / sign
                for axis, sign in zip(("roll", "pitch", "yaw"), axis_signs)
            }
            body_mag = rotate_vector(
                quaternion_inverse(vehicle), world_magnetic_field
            )
            imu_mag = rotate_vector(rov.imu_mount_quaternion, body_mag)
            raw_mag = tuple(
                component / sign
                for component, sign in zip(imu_mag, axis_signs)
            )
            rov._update_imu_state(
                {
                    "euler_deg": raw_euler,
                    "gyro_dps": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "acc_mps2": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "mag_cal": {
                        "x": raw_mag[0], "y": raw_mag[1], "z": raw_mag[2]
                    },
                },
                now,
            )
            return rov.sensor_state["heading_vehicle"]

        level = sample(0.0, 0.0, 73.0, 1000.0)
        tilted = sample(28.0, -34.0, 73.0, 1000.05)

        self.assertAlmostEqual(level, 73.0, places=6)
        self.assertAlmostEqual(tilted, 73.0, places=6)

    def test_imu_tilt_compensation_handles_combined_roll_and_pitch(self):
        reader = IMUReader.__new__(IMUReader)
        vehicle = from_euler_deg(28.0, -34.0, 73.0)
        body_magnetic = rotate_vector(
            quaternion_inverse(vehicle), (1.0, 0.0, 0.4)
        )

        heading_180, _heading_360 = reader.calc_mag_heading_tilt_compensated(
            *body_magnetic, 28.0, -34.0
        )

        self.assertAlmostEqual(heading_180, -73.0, places=6)

    def test_missing_heading_does_not_refresh_yaw_control_health(self):
        rov = self.make_rov()
        rov._update_imu_state(
            {
                "euler_deg": {"roll": 1.0, "pitch": 2.0, "yaw": 3.0},
                "gyro_dps": {"x": 0.0, "y": 0.0, "z": 0.0},
                "acc_mps2": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            time.monotonic(),
        )

        self.assertTrue(rov._sensor_health()["imu_valid"])
        self.assertFalse(rov._sensor_health()["heading_valid"])

    def test_control_loop_tracks_heading_only_while_manual_override_is_active(self):
        rov = self.make_rov()
        rov.attitude_hold = {"roll": False, "pitch": False, "yaw": True}
        rov.attitude_hold_target_deg["yaw"] = 10.0

        rov._track_yaw_target_during_manual_override(
            -170.0,
            yaw_hold=True,
            imu_valid=True,
            manual_override=True,
        )
        target = rov._track_yaw_target_during_manual_override(
            -167.25,
            yaw_hold=True,
            imu_valid=True,
            manual_override=False,
        )

        self.assertEqual(target, -170.0)
        self.assertEqual(rov.attitude_hold_target_deg["yaw"], -170.0)

    def test_first_neutral_joystick_packet_starts_inertia_wait(self):
        rov = self.make_rov()
        rov.attitude_hold = {"roll": False, "pitch": False, "yaw": True}
        rov.attitude_hold_target_deg["yaw"] = 10.0
        rov._last_imu_update = time.monotonic()
        rov._last_heading_update = rov._last_imu_update
        rov.sensor_state["yaw"] = -56.6

        rov._on_joystick({"M_z": 5.0})
        active_target = rov.attitude_hold_target_deg["yaw"]
        rov.sensor_state["yaw"] = -52.75
        rov.sensor_state["gyro_z"] = 8.0
        rov._on_joystick({"M_z": 0.0})

        self.assertEqual(rov.attitude_hold_target_deg["yaw"], active_target)
        self.assertTrue(rov._yaw_release_pending)

    def test_yaw_target_coasts_until_rate_is_below_threshold_for_dwell(self):
        rov = self.make_rov()
        rov.attitude_hold = {"roll": False, "pitch": False, "yaw": True}
        rov.attitude_hold_target_deg["yaw"] = -50.0
        rov._yaw_release_pending = True
        rov._yaw_release_started_at = 100.0

        target = rov._track_yaw_target_during_manual_override(
            -45.0, yaw_rate=8.0, yaw_hold=True, imu_valid=True,
            manual_override=False, now=100.0,
        )
        self.assertEqual(target, -45.0)
        self.assertTrue(rov._yaw_release_pending)

        rov._track_yaw_target_during_manual_override(
            -42.0, yaw_rate=2.0, yaw_hold=True, imu_valid=True,
            manual_override=False, now=100.10,
        )
        target = rov._track_yaw_target_during_manual_override(
            -41.5, yaw_rate=1.5, yaw_hold=True, imu_valid=True,
            manual_override=False, now=100.31,
        )
        self.assertEqual(target, -41.5)
        self.assertFalse(rov._yaw_release_pending)

    def test_yaw_pid_is_paused_during_release_settling(self):
        actual = ROVController._attitude_hold_for_control(
            {"roll": True, "pitch": True, "yaw": True},
            {"roll": False, "pitch": False, "yaw": False},
            yaw_release_settling=True,
        )

        self.assertEqual(actual, {"roll": True, "pitch": True, "yaw": False})

    def test_yaw_release_wait_has_a_bounded_safety_timeout(self):
        rov = self.make_rov()
        rov.attitude_hold = {"roll": False, "pitch": False, "yaw": True}
        rov._yaw_release_pending = True
        rov._yaw_release_started_at = 100.0

        target = rov._track_yaw_target_during_manual_override(
            25.0, yaw_rate=12.0, yaw_hold=True, imu_valid=True,
            manual_override=False, now=102.01,
        )

        self.assertEqual(target, 25.0)
        self.assertFalse(rov._yaw_release_pending)

    def test_uncommissioned_attitude_mapping_fails_closed(self):
        rov = self.make_rov()
        rov.thrusters_armed = True
        rov.link_timed_out = False
        rov._last_imu_update = time.monotonic()
        rov.attitude_control_mapping_validated = False
        with self.assertRaisesRegex(ValueError, "mapping_unvalidated"):
            rov._set_attitude_hold({"roll": True, "pitch": True})
        self.assertEqual(rov.attitude_hold_reason, "mapping_unvalidated")

    def test_mapping_gate_disable_stops_holds_and_persists(self):
        rov = self.make_rov()
        rov.attitude_control_mapping_validated = True
        rov.attitude_hold = {"roll": True, "pitch": True}

        actual = rov._set_attitude_mapping_validated(False)

        self.assertFalse(actual)
        self.assertEqual(
            rov.attitude_hold,
            {"roll": False, "pitch": False, "yaw": False},
        )
        self.assertEqual(rov.attitude_hold_reason, "mapping_unvalidated")
        self.assertFalse(load_config(rov.config_path).getboolean(
            "control", "attitude_control_mapping_validated"
        ))

    def test_mapping_gate_enable_requires_safe_calibrated_stationary_state(self):
        cases = (
            ("thrusters_armed", True, "thrusters_must_be_locked"),
            ("control_enabled", True, "autonomous_control_must_be_off"),
            ("attitude_hold", {"roll": True, "pitch": False}, "holds_must_be_off"),
            ("imu_3d_mount_calibrated", False, "mount_uncalibrated"),
            ("_last_imu_update", 0.0, "imu_not_fresh"),
        )
        for attribute, value, error in cases:
            with self.subTest(attribute=attribute):
                rov = self.make_rov()
                rov.thrusters_armed = False
                rov.control_enabled = False
                rov.attitude_hold = {"roll": False, "pitch": False}
                rov.imu_3d_mount_calibrated = True
                rov._last_imu_update = time.monotonic()
                setattr(rov, attribute, value)
                with self.assertRaisesRegex(ValueError, error):
                    rov._set_attitude_mapping_validated(True)

    def test_mapping_gate_enable_persists_when_all_checks_pass(self):
        rov = self.make_rov()
        rov.thrusters_armed = False
        rov.control_enabled = False
        rov.attitude_hold = {"roll": False, "pitch": False}
        rov.imu_3d_mount_calibrated = True
        rov._last_imu_update = time.monotonic()

        self.assertTrue(rov._set_attitude_mapping_validated(True))
        self.assertTrue(load_config(rov.config_path).getboolean(
            "control", "attitude_control_mapping_validated"
        ))

    def test_mapping_rejection_ack_keeps_exact_actionable_error(self):
        rov = self.make_rov()
        rov.thrusters_armed = True
        rov._last_imu_update = time.monotonic()
        rov.attitude_control_mapping_validated = False

        rov._on_command(
            {"attitude_hold": {"roll": True, "pitch": False}},
            {"authenticated": True, "session": "test", "session_changed": False},
        )

        self.assertFalse(rov.last_command_ack["ok"])
        self.assertEqual(rov.last_command_ack["message"], "command_rejected")
        self.assertEqual(
            rov.last_command_ack["error"],
            "attitude_hold_rejected_mapping_unvalidated",
        )
        self.assertEqual(rov.attitude_hold_reason, "mapping_unvalidated")

    def test_servo_keepalive_does_not_replace_transaction_ack(self):
        rov = self.make_rov()
        expected_ack = {
            "ok": True,
            "message": "motion_tuning_saved",
            "timestamp": time.time(),
            "motion_tuning": {"horizontal_limit_fraction": 0.2},
        }
        rov.last_command_ack = dict(expected_ack)

        for _ in range(3):
            rov._on_command(
                {"servo_position": 0.2},
                {"authenticated": True, "session": "test", "session_changed": False},
            )

        self.assertEqual(rov.last_command_ack, expected_ack)

    def test_legacy_config_without_commissioning_key_starts_but_stays_inhibited(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.ini"
            text = (ROOT / "config.ini").read_text(encoding="utf-8")
            new_geometry_prefixes = tuple(
                f"vertical_{corner}_{suffix}"
                for corner in ("fl", "fr", "rl", "rr")
                for suffix in ("position_m", "direction_body")
            )
            new_control_keys = (
                "attitude_control_mapping_validated",
                "attitude_hover_tilt_max_scale",
            )
            text = "\n".join(
                line for line in text.splitlines()
                if not line.startswith(new_geometry_prefixes + new_control_keys)
            ) + "\n"
            path.write_text(text, encoding="utf-8")
            rov = ROVController(config_path=path)
        self.assertFalse(rov.attitude_control_mapping_validated)

    def test_hover_tilt_compensation_is_level_neutral_and_strictly_bounded(self):
        self.assertAlmostEqual(
            tilt_compensated_hover_per_thruster(1.0, 0.0, 0.0, 1.15),
            1.0,
        )
        self.assertAlmostEqual(
            tilt_compensated_hover_per_thruster(1.0, 30.0, 20.0, 1.15),
            1.15,
        )
        self.assertLessEqual(
            tilt_compensated_hover_per_thruster(1.0, 49.0, 49.0, 1.15),
            1.15,
        )

    def test_disable_clears_runtime_switches_outputs_and_pid_state(self):
        rov = self.make_rov()
        rov.attitude_hold = {"roll": True, "pitch": True}
        rov.attitude_hold_output = {"roll_moment_nm": 2.0, "pitch_moment_nm": -2.0}
        rov.controller._roll_hold_active = True
        rov.controller._pitch_hold_active = True
        rov._disable_attitude_hold("control_link_timeout")
        self.assertEqual(
            rov.attitude_hold,
            {"roll": False, "pitch": False, "yaw": False},
        )
        self.assertEqual(rov.attitude_hold_output, {
            "roll_moment_nm": 0.0,
            "pitch_moment_nm": 0.0,
            "yaw_moment_nm": 0.0,
            "roll_manual_override": False,
            "pitch_manual_override": False,
            "yaw_manual_override": False,
        })
        self.assertFalse(rov.controller._roll_hold_active)
        self.assertFalse(rov.controller._pitch_hold_active)
        self.assertFalse(rov.controller._yaw_hold_active)

    def test_telemetry_is_pi_authoritative_for_switch_target_and_output(self):
        rov = self.make_rov()
        rov.attitude_hold = {"roll": True, "pitch": False}
        rov.attitude_hold_output = {"roll_moment_nm": 1.25, "pitch_moment_nm": 0.0}
        tele = rov.build_telemetry_packet()
        self.assertEqual(tele["attitude_hold"]["roll_target_deg"], 0.0)
        self.assertEqual(tele["attitude_hold"]["pitch_target_deg"], 0.0)
        self.assertEqual(tele["attitude_hold"]["yaw_target_deg"], 0.0)
        self.assertEqual(
            tele["attitude_hold"]["yaw_control_source"],
            "referenced_euler_yaw",
        )
        self.assertEqual(
            tele["attitude_hold"]["yaw_control_deg"],
            rov.sensor_state["yaw"],
        )
        self.assertEqual(tele["attitude_hold"]["roll_moment_nm"], 1.25)
        self.assertTrue(tele["attitude_hold"]["roll"])
        self.assertEqual(
            tele["attitude_hold"]["mapping_validated"],
            rov.attitude_control_mapping_validated,
        )
        self.assertNotIn("max_tilt_deg", tele["attitude_hold"])
        self.assertNotIn("tilt_cutoff_enabled", tele["attitude_hold"])
        calibration = tele["attitude_calibration_config"]
        self.assertEqual(
            calibration["mapping_validated"],
            rov.attitude_control_mapping_validated,
        )
        self.assertEqual(
            set(calibration),
            {
                "imu_3d_mount_calibrated", "mapping_validated",
                "imu_mount_quat_w", "imu_mount_quat_x",
                "imu_mount_quat_y", "imu_mount_quat_z",
                "imu_reference_quat_w", "imu_reference_quat_x",
                "imu_reference_quat_y", "imu_reference_quat_z",
            },
        )

    def test_manual_attitude_override_clears_pid_and_release_primes_without_windup(self):
        cases = {
            "roll": ("M_x", 3, {"roll": 10.0, "pitch": 0.0}),
            "pitch": ("M_y", 4, {"roll": 0.0, "pitch": -10.0}),
            "yaw": (
                "M_z", 5,
                {"roll": 0.0, "pitch": 0.0, "yaw": 10.0, "gyro_z": 0.0},
            ),
        }
        for axis, (manual_axis, output_index, current) in cases.items():
            with self.subTest(axis=axis):
                rov = self.make_rov()
                pid = getattr(rov.controller, f"{axis}_controller")
                pid.set_pid((0.0, 1.0, 0.0))
                pid.sample_time = 0.0
                hold = {
                    "roll": axis == "roll",
                    "pitch": axis == "pitch",
                    "yaw": axis == "yaw",
                }

                # Accumulate an integral while the hold owns this moment axis.
                rov.controller.compute_control(
                    current, clean_mode(), clean_target(),
                    {"imu": True, "depth": False}, attitude_hold=hold,
                )
                pid.last_time -= 1.0
                rov.controller.compute_control(
                    current, clean_mode(), clean_target(),
                    {"imu": True, "depth": False}, attitude_hold=hold,
                )
                self.assertGreater(abs(pid.integral), 0.0)

                joy = dict(joystick_handler.NEUTRAL_JOYSTICK)
                joy[manual_axis] = 5.0
                override = rov._attitude_manual_overrides(hold, joy, True, 0.01)
                paused_hold = rov._attitude_hold_for_control(hold, override)
                self.assertTrue(override[axis])
                rov.controller.compute_control(
                    current, clean_mode(), clean_target(),
                    {"imu": True, "depth": False}, attitude_hold=paused_hold,
                )
                self.assertEqual(pid.integral, 0.0)
                self.assertFalse(getattr(rov.controller, f"_{axis}_hold_active"))

                joy[manual_axis] = 0.0
                override = rov._attitude_manual_overrides(hold, joy, True, 0.01)
                resumed_hold = rov._attitude_hold_for_control(hold, override)
                pid.sample_time = 0.1
                output, _ = rov.controller.compute_control(
                    current, clean_mode(), clean_target(),
                    {"imu": True, "depth": False}, attitude_hold=resumed_hold,
                )
                self.assertEqual(output[output_index], 0.0)
                self.assertEqual(pid.integral, 0.0)
                self.assertTrue(getattr(rov.controller, f"_{axis}_hold_active"))

    def test_manual_moment_is_not_reported_as_pid_output(self):
        rov = self.make_rov()
        rov.thrusters_armed = True
        rov.link_timed_out = False
        rov.motion_output_enabled = False

        class Allocator:
            last_result = None

            def set_wrench(self, **kwargs):
                values = [
                    kwargs["tau_x"], kwargs["tau_y"], kwargs["tau_z"],
                    kwargs["M_x"], kwargs["M_y"], kwargs["M_z"],
                ]
                self.last_result = types.SimpleNamespace(achieved_tau=values)
                return ({"vertical_fl": kwargs["M_y"]}, {"vertical_fl": 1600})

        rov.allocator = Allocator()
        result = rov._write_control_wrench(
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            {"roll": False, "pitch": True},
            (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            {"roll": False, "pitch": True},
        )
        self.assertIsNotNone(result)
        self.assertEqual(rov.force_state["M_y"], 5.0)
        self.assertEqual(rov.attitude_hold_output["pitch_moment_nm"], 0.0)
        self.assertTrue(rov.attitude_hold_output["pitch_manual_override"])

    def test_control_write_forwards_attitude_and_trim_priority_moments(self):
        rov = self.make_rov()
        rov.thrusters_armed = True
        rov.link_timed_out = False
        rov.motion_output_enabled = False

        class Allocator:
            last_result = None

            def __init__(self):
                self.kwargs = None

            def set_wrench(self, **kwargs):
                self.kwargs = kwargs
                values = [
                    kwargs["tau_x"], kwargs["tau_y"], kwargs["tau_z"],
                    kwargs["M_x"], kwargs["M_y"], kwargs["M_z"],
                ]
                self.last_result = types.SimpleNamespace(achieved_tau=values)
                return ({"vertical_fl": 0.0}, {"vertical_fl": 1540})

        rov.allocator = Allocator()
        rov._write_control_wrench(
            [0.0, 0.0, -66.0, 2.0, 3.0, 0.0],
            {"roll": True, "pitch": True},
            (0.0, 0.0, 0.0, 2.0, 2.5, 0.0),
            {"roll": False, "pitch": False},
            priority_moments=(2.0, 3.0),
        )

        self.assertEqual(rov.allocator.kwargs["priority_M_x"], 2.0)
        self.assertEqual(rov.allocator.kwargs["priority_M_y"], 3.0)

    def test_safety_transitions_are_ordered_after_inflight_control_write(self):
        transitions = {
            "emergency_stop": lambda rov: rov.trigger_emergency_stop("test"),
            "link_timeout": lambda rov: rov._lock_for_rearm(
                "control_link_timeout", link_timed_out=True
            ),
            "manual_lock": lambda rov: rov._lock_for_rearm("manual_lock"),
        }
        for name, transition in transitions.items():
            with self.subTest(name=name):
                rov = self.make_rov()
                rov.thrusters_armed = True
                rov.link_timed_out = False
                rov.motion_output_enabled = True
                rov.servo_output_enabled = False
                entered_write = threading.Event()
                release_write = threading.Event()

                class RecordingAllocator:
                    last_result = None

                    def __init__(self):
                        self.records = []

                    def set_wrench(self, **kwargs):
                        entered_write.set()
                        if not release_write.wait(2.0):
                            raise TimeoutError("test did not release control write")
                        values = [
                            kwargs["tau_x"], kwargs["tau_y"], kwargs["tau_z"],
                            kwargs["M_x"], kwargs["M_y"], kwargs["M_z"],
                        ]
                        self.last_result = types.SimpleNamespace(achieved_tau=values)
                        self.records.append("nonneutral")
                        return ({"t": 1.0}, {"t": 1600})

                    def stop_all(self):
                        self.records.append("neutral")

                allocator = RecordingAllocator()
                rov.allocator = allocator
                writer_result = []
                writer = threading.Thread(target=lambda: writer_result.append(
                    rov._write_control_wrench(
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        {"roll": False, "pitch": False},
                        (0.0,) * 6,
                        {"roll": False, "pitch": False},
                    )
                ))
                writer.start()
                self.assertTrue(entered_write.wait(1.0))

                safety = threading.Thread(target=lambda: transition(rov))
                safety.start()
                release_write.set()
                writer.join(2.0)
                safety.join(2.0)
                self.assertFalse(writer.is_alive())
                self.assertFalse(safety.is_alive())
                self.assertIsNotNone(writer_result[0])
                self.assertEqual(allocator.records, ["nonneutral", "neutral"])

                # Once neutralization has completed, the stale cycle is rejected
                # before touching the allocator, so neutral remains the final write.
                rejected = rov._write_control_wrench(
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    {"roll": False, "pitch": False},
                    (0.0,) * 6,
                    {"roll": False, "pitch": False},
                )
                self.assertIsNone(rejected)
                self.assertEqual(allocator.records[-1], "neutral")

    def test_servo_safety_write_is_ordered_after_inflight_worker_write(self):
        rov = self.make_rov()
        rov.servo_output_enabled = True
        rov.servo_target_position = 0.5
        rov.servo_position = 0.0
        rov.servo_angle_deg = 0.0
        rov._servo_last_update = time.monotonic() - 1.0
        rov.link_timed_out = False
        rov.emergency_stop_latched = False
        rov.is_running = True

        writes = []

        class RecordingAllocator:
            def set_aux_pwm(self, channel, pulse):
                writes.append((channel, pulse))

            def stop_all(self):
                pass

        rov.allocator = RecordingAllocator()
        original_write = rov._write_servo_position
        worker_ready = threading.Event()
        release_worker = threading.Event()

        def paused_worker_write(position):
            if threading.current_thread().name == "ServoRaceWorker":
                worker_ready.set()
                if not release_worker.wait(2.0):
                    raise TimeoutError("test did not release servo write")
            original_write(position)

        rov._write_servo_position = paused_worker_write
        worker = threading.Thread(
            target=rov._servo_worker,
            name="ServoRaceWorker",
        )
        worker.start()
        self.assertTrue(worker_ready.wait(1.0))

        safety = threading.Thread(
            target=lambda: rov.trigger_emergency_stop("test")
        )
        safety.start()
        time.sleep(0.01)
        release_worker.set()
        safety.join(2.0)
        rov.is_running = False
        worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertFalse(safety.is_alive())
        self.assertGreaterEqual(len(writes), 2)
        self.assertEqual(
            writes[-1],
            (rov.servo_channel, rov._servo_position_to_pulse(rov.servo_failsafe_position)),
        )


if __name__ == "__main__":
    unittest.main()
