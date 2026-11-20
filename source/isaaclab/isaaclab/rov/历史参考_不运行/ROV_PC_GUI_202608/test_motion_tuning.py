import importlib.util
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


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

from config_loader import load_config
from pc_gui import ROVClient, validate_attitude_tuning, validate_motion_tuning
from secure_protocol import ProtocolError, validate_control_payload

sys.path.insert(0, str(ROOT / "pi_runtime"))
from main import (
    ROVController,
    fuse_pitch_heave_feedforward,
    select_vertical_front_rear_trim,
    vertical_command_with_hover,
    vertical_front_rear_trim_moment,
)

_allocator_spec = importlib.util.spec_from_file_location(
    "test_motion_tuning_allocator", ROOT / "pi_runtime" / "thrust_allocator.py"
)
_allocator_module = importlib.util.module_from_spec(_allocator_spec)
sys.modules[_allocator_spec.name] = _allocator_module
_allocator_spec.loader.exec_module(_allocator_module)
ThrustAllocator = _allocator_module.ThrustAllocator


def make_allocator(
    horizontal=0.2, vertical=0.2, vertical_x_m=0.18,
    forward_trim=0.0, backward_trim=0.0, correction=0.5,
    left_lateral_trim=0.0, right_lateral_trim=0.0,
):
    return ThrustAllocator(
        enable_hardware=False,
        pwm_min_us=1100,
        pwm_neutral_us=1540,
        pwm_max_us=1900,
        pwm_output_min_us=1200,
        pwm_output_max_us=1850,
        reverse_start_us=1532,
        forward_start_us=1578,
        horizontal_limit_fraction=horizontal,
        vertical_limit_fraction=vertical,
        vertical_correction_limit_fraction=correction,
        vertical_x_m=vertical_x_m,
        forward_front_rear_trim_percent=forward_trim,
        backward_front_rear_trim_percent=backward_trim,
        left_lateral_surge_trim_percent=left_lateral_trim,
        right_lateral_surge_trim_percent=right_lateral_trim,
    )


class MotionTuningValidationTests(unittest.TestCase):
    def test_fixed_vertical_compensation_is_validated_with_thrust_limits(self):
        self.assertEqual(
            validate_motion_tuning(
                "0.2", "0.3", "0.5", "10", "-25", "12", "15"
            ),
            {
                "horizontal_limit_fraction": 0.2,
                "vertical_limit_fraction": 0.3,
                "hover_force_per_thruster_n": 0.5,
                "vertical_ascent_front_rear_trim_percent": 10.0,
                "vertical_descent_front_rear_trim_percent": -25.0,
                "forward_front_rear_trim_percent": 12.0,
                "backward_front_rear_trim_percent": 15.0,
            },
        )
        with self.assertRaises(ValueError):
            validate_motion_tuning(
                "0.2", "0.01", "0.6", "0", "0", "0", "0"
            )
        for invalid_trim in ("-100.1", "100.1", "nan"):
            with self.assertRaises(ValueError):
                validate_motion_tuning(
                    "0.2", "0.3", "0.5",
                    invalid_trim, "0", "0", "0",
                )
            with self.assertRaises(ValueError):
                validate_motion_tuning(
                    "0.2", "0.3", "0.5",
                    "0", "0", "0", invalid_trim,
                )
        self.assertEqual(
            validate_motion_tuning("0.2", "0.3", "10", "0", "0", "0", "0")
            ["hover_force_per_thruster_n"],
            10.0,
        )
        with self.assertRaises(ValueError):
            validate_motion_tuning(
                "0.2", "0.3", "10.01", "0", "0", "0", "0"
            )

    def test_protocol_accepts_bounded_fixed_vertical_compensation(self):
        tuning = {
            "horizontal_limit_fraction": 0.2,
            "vertical_limit_fraction": 0.3,
            "hover_force_per_thruster_n": 0.5,
            "vertical_ascent_front_rear_trim_percent": 10.0,
            "vertical_descent_front_rear_trim_percent": -25.0,
            "forward_front_rear_trim_percent": 12.0,
            "backward_front_rear_trim_percent": 15.0,
        }
        self.assertEqual(
            validate_control_payload({"motion_tuning": tuning}),
            {"motion_tuning": tuning},
        )
        invalid = dict(tuning, hover_force_per_thruster_n=20.0)
        with self.assertRaises(ProtocolError):
            validate_control_payload({"motion_tuning": invalid})
        with self.assertRaises(ProtocolError):
            validate_control_payload({
                "motion_tuning": dict(
                    tuning, vertical_descent_front_rear_trim_percent=100.1
                )
            })

    def test_protocol_strictly_accepts_attitude_switches_and_limits(self):
        self.assertEqual(
            validate_control_payload({
                "attitude_hold": {"roll": True, "pitch": False, "yaw": False}
            }),
            {"attitude_hold": {"roll": True, "pitch": False, "yaw": False}},
        )
        tuning = validate_attitude_tuning("4", "3.5", "4", "3.0")
        self.assertEqual(
            validate_control_payload({"attitude_tuning": tuning}),
            {"attitude_tuning": tuning},
        )
        for invalid_threshold in (0.49, 30.01, float("nan")):
            with self.assertRaises((ValueError, ProtocolError)):
                invalid = validate_attitude_tuning(
                    "4", "3.5", "4", invalid_threshold
                )
                validate_control_payload({"attitude_tuning": invalid})
        for invalid in (
            {"roll": True},
            {"roll": True, "pitch": 1},
            {"roll": True, "pitch": False},
        ):
            with self.assertRaises(ProtocolError):
                validate_control_payload({"attitude_hold": invalid})

    def test_gui_client_transmits_new_commands(self):
        client = ROVClient.__new__(ROVClient)
        sent = []
        client._send_payload = sent.append
        client.send_cmd(attitude_hold={"roll": True, "pitch": False, "yaw": False})
        client.send_cmd(attitude_tuning={
            "roll_max_moment_nm": 2.0,
            "pitch_max_moment_nm": 1.0,
            "yaw_max_moment_nm": 3.0,
            "yaw_release_rate_threshold_dps": 3.0,
        })
        self.assertEqual(sent, [
            {"attitude_hold": {"roll": True, "pitch": False, "yaw": False}},
            {"attitude_tuning": {
                "roll_max_moment_nm": 2.0,
                "pitch_max_moment_nm": 1.0,
                "yaw_max_moment_nm": 3.0,
                "yaw_release_rate_threshold_dps": 3.0,
            }},
        ])


class ThrusterIsolationTests(unittest.TestCase):
    def test_fixed_vertical_compensation_restores_two_newtons_total(self):
        # The configured positive magnitude is upward compensation, while
        # body +Z is down, so the injected wrench is negative Fz.
        self.assertEqual(vertical_command_with_hover(0.0, 0.5), -2.0)
        self.assertEqual(vertical_command_with_hover(-3.0, 0.5), -5.0)

    def test_fixed_hover_is_independent_of_depth_hold_state(self):
        self.assertEqual(
            ROVController._effective_vertical_hover(1.5, True, False), 1.5
        )
        self.assertEqual(
            ROVController._effective_vertical_hover(1.5, True, True), 1.5
        )
        self.assertEqual(
            ROVController._effective_vertical_hover(1.5, False, False), 1.5
        )

    def test_zero_fixed_hover_keeps_manual_and_depth_corrections_unchanged(self):
        self.assertEqual(ROVController._effective_vertical_hover(0.0, True, False), 0.0)
        self.assertEqual(vertical_command_with_hover(-12.5, 0.0), -12.5)
        self.assertEqual(vertical_command_with_hover(7.25, 0.0), 7.25)

    def test_pi_tracks_depth_target_during_manual_heave_override(self):
        rov = ROVController(config_path=ROOT / "config.ini")
        rov.mode = {
            "heading_hold": False,
            "depth_hold": True,
            "speed_hold": False,
        }
        rov.target["depth_m"] = 0.5

        target = rov._track_depth_target_during_manual_override(
            dict(rov.target), 0.83, depth_valid=True, manual_override=True
        )

        self.assertEqual(target["depth_m"], 0.83)
        self.assertEqual(rov.target["depth_m"], 0.83)

    def test_front_rear_trim_redistributes_without_changing_total_vertical_force(self):
        arm = 0.0625
        allocator = make_allocator(vertical=0.5, vertical_x_m=arm)
        for trim, expected_front, expected_rear in (
            (-50.0, 12.5, 7.5),
            (50.0, 7.5, 12.5),
        ):
            pitch_moment = vertical_front_rear_trim_moment(40.0, trim, arm)
            result = allocator.allocate(
                tau_z=40.0, M_y=pitch_moment, write_output=False
            )
            for thrust in result.nominal_thrusts[4:6]:
                self.assertAlmostEqual(thrust, expected_front, places=6)
            for thrust in result.nominal_thrusts[6:8]:
                self.assertAlmostEqual(thrust, expected_rear, places=6)
            self.assertAlmostEqual(sum(result.nominal_thrusts[4:]), 40.0, places=6)
            self.assertAlmostEqual(result.achieved_tau[2], 40.0, places=6)

    def test_zero_lateral_surge_trim_preserves_the_original_allocation(self):
        baseline = make_allocator(horizontal=0.5)
        explicit_zero = make_allocator(
            horizontal=0.5, left_lateral_trim=0.0, right_lateral_trim=0.0
        )

        expected = baseline.allocate(
            tau_x=3.0, tau_y=-20.0, M_z=0.5, write_output=False
        )
        actual = explicit_zero.allocate(
            tau_x=3.0, tau_y=-20.0, M_z=0.5, write_output=False
        )

        self.assertEqual(actual.achieved_tau.tolist(), expected.achieved_tau.tolist())
        self.assertEqual(
            actual.nominal_thrusts.tolist(), expected.nominal_thrusts.tolist()
        )

    def test_left_and_right_lateral_trims_add_independent_surge_only(self):
        allocator = make_allocator(
            horizontal=0.5,
            left_lateral_trim=-5.0,
            right_lateral_trim=-8.0,
        )

        left = allocator.allocate(tau_y=-20.0, write_output=False)
        right = allocator.allocate(tau_y=20.0, write_output=False)
        forward_only = allocator.allocate(tau_x=4.0, write_output=False)

        self.assertAlmostEqual(left.achieved_tau[0], -1.0, places=6)
        self.assertAlmostEqual(left.achieved_tau[1], -20.0, places=6)
        self.assertAlmostEqual(right.achieved_tau[0], -1.6, places=6)
        self.assertAlmostEqual(right.achieved_tau[1], 20.0, places=6)
        self.assertAlmostEqual(forward_only.achieved_tau[0], 4.0, places=6)
        self.assertTrue(all(abs(value) < 1e-9 for value in left.achieved_tau[2:5]))

    def test_lateral_surge_trim_rejects_nonfinite_and_out_of_range_values(self):
        allocator = make_allocator()
        for invalid in (-100.1, 100.1, float("inf"), float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    allocator.set_left_lateral_surge_trim_percent(invalid)
                with self.assertRaises(ValueError):
                    allocator.set_right_lateral_surge_trim_percent(invalid)

    def test_front_rear_trim_scales_with_full_signed_vertical_command(self):
        self.assertEqual(vertical_front_rear_trim_moment(0.0, 80.0, 0.1), 0.0)
        self.assertAlmostEqual(
            vertical_front_rear_trim_moment(-20.0, -50.0, 0.1), 0.5
        )
        with self.assertRaises(ValueError):
            vertical_front_rear_trim_moment(20.0, 101.0, 0.1)

    def test_full_heave_keeps_calibrated_trim_in_reserved_vertical_headroom(self):
        arm = 0.0625
        allocator = make_allocator(vertical=0.3, vertical_x_m=arm)
        full_base_fz = -4.0 * 55.0 * 0.3
        trim_moment = vertical_front_rear_trim_moment(
            full_base_fz, -19.0, arm
        )

        result = allocator.allocate(
            tau_z=full_base_fz,
            M_y=trim_moment,
            priority_M_y=trim_moment,
            write_output=False,
        )

        self.assertAlmostEqual(result.achieved_tau[2], full_base_fz, places=6)
        self.assertAlmostEqual(result.achieved_tau[4], trim_moment, places=6)
        vertical = result.motor_thrusts[4:8]
        self.assertTrue(any(abs(value) > 16.5 for value in vertical))
        self.assertTrue(all(abs(value) <= 27.5 + 1e-9 for value in vertical))

    def test_attitude_priority_can_exceed_motion_limit_but_not_hard_ceiling(self):
        allocator = make_allocator(vertical=0.3, vertical_x_m=0.0625)
        result = allocator.allocate(
            tau_z=-66.0,
            M_y=3.0,
            priority_M_y=3.0,
            write_output=False,
        )

        vertical = result.motor_thrusts[4:8]
        self.assertGreater(abs(result.achieved_tau[4]), 2.5)
        self.assertTrue(any(abs(value) > 16.5 for value in vertical))
        self.assertTrue(all(abs(value) <= 27.5 + 1e-9 for value in vertical))

    def test_vertical_trim_selection_separates_ascent_descent_and_equal_hover(self):
        self.assertEqual(select_vertical_front_rear_trim(-2.0, 15, -38, 0.1), 15)
        self.assertEqual(select_vertical_front_rear_trim(2.0, 15, -38, 0.1), -38)
        self.assertEqual(select_vertical_front_rear_trim(0.0, 15, -38, 0.1), 0)
        self.assertEqual(select_vertical_front_rear_trim(0.05, 15, -38, 0.1), 0)

    def test_pitch_hold_suppresses_open_loop_vertical_trim(self):
        for correction in (-40.0, 40.0):
            self.assertEqual(
                select_vertical_front_rear_trim(
                    correction, -28.0, -38.0, 0.1, pitch_hold=True
                ),
                0.0,
            )

    def test_pitch_hold_fuses_half_scale_signed_heave_after_slew(self):
        ascent_total, ascent_feedforward = fuse_pitch_heave_feedforward(
            0.0, -66.0, 0.0, -28.0, -38.0, 0.001, 0.0625, 0.5, 3.0
        )
        descent_total, descent_feedforward = fuse_pitch_heave_feedforward(
            0.0, 66.0, 0.0, -28.0, -38.0, 0.001, 0.0625, 0.5, 3.0
        )
        self.assertAlmostEqual(ascent_total, 0.28875)
        self.assertAlmostEqual(ascent_feedforward, 0.28875)
        self.assertAlmostEqual(descent_total, -0.391875)
        self.assertAlmostEqual(descent_feedforward, -0.391875)

    def test_pitch_feedforward_uses_slewed_force_and_total_moment_limit(self):
        ramp_total, ramp_feedforward = fuse_pitch_heave_feedforward(
            0.0, -10.0, 0.0, -28.0, -38.0, 0.001, 0.0625, 0.5, 3.0
        )
        saturated_total, applied_feedforward = fuse_pitch_heave_feedforward(
            2.9, -66.0, 0.0, -28.0, -38.0, 0.001, 0.0625, 0.5, 3.0
        )
        self.assertAlmostEqual(ramp_total, 0.04375)
        self.assertAlmostEqual(ramp_feedforward, 0.04375)
        self.assertAlmostEqual(saturated_total, 3.0)
        self.assertAlmostEqual(applied_feedforward, 0.1)

    def test_fixed_equal_hover_does_not_select_pitch_feedforward(self):
        total, feedforward = fuse_pitch_heave_feedforward(
            0.25, -4.0, 1.0, -28.0, -38.0, 0.001, 0.0625, 0.5, 3.0
        )
        self.assertAlmostEqual(total, 0.25)
        self.assertAlmostEqual(feedforward, 0.0)

    def test_zero_hover_and_zero_direction_trims_keep_pitch_pid_unchanged(self):
        for vertical_force in (-60.0, 60.0):
            total, feedforward = fuse_pitch_heave_feedforward(
                0.75,
                vertical_force,
                0.0,
                0.0,
                0.0,
                0.001,
                0.0625,
                0.25,
                3.0,
            )
            self.assertAlmostEqual(total, 0.75)
            self.assertAlmostEqual(feedforward, 0.0)

    def test_each_zero_trim_contributes_nothing_without_disabling_control(self):
        for vertical_force, ascent_trim, descent_trim in (
            (-40.0, 0.0, -35.0),
            (40.0, 20.0, 0.0),
        ):
            total, feedforward = fuse_pitch_heave_feedforward(
                0.75,
                vertical_force,
                0.0,
                ascent_trim,
                descent_trim,
                0.001,
                0.0625,
                0.25,
                3.0,
            )
            self.assertAlmostEqual(total, 0.75)
            self.assertAlmostEqual(feedforward, 0.0)

        for tau_x, forward_trim, backward_trim in (
            (20.0, 0.0, 60.0),
            (-20.0, -60.0, 0.0),
        ):
            baseline = make_allocator(
                horizontal=0.5, vertical=0.5
            ).allocate(
                tau_x=tau_x, tau_z=4.0, M_x=0.5, M_y=0.75,
                write_output=False,
            )
            actual = make_allocator(
                horizontal=0.5,
                vertical=0.5,
                forward_trim=forward_trim,
                backward_trim=backward_trim,
            ).allocate(
                tau_x=tau_x, tau_z=4.0, M_x=0.5, M_y=0.75,
                write_output=False,
            )
            for before, after in zip(
                baseline.nominal_thrusts, actual.nominal_thrusts
            ):
                self.assertAlmostEqual(before, after, places=6)
            self.assertAlmostEqual(actual.achieved_tau[0], tau_x, places=6)
            self.assertAlmostEqual(actual.achieved_tau[2], 4.0, places=6)
            self.assertNotEqual(actual.achieved_tau[3], 0.0)
            self.assertNotEqual(actual.achieved_tau[4], 0.0)

    def test_pitch_feedforward_scale_is_bounded(self):
        with self.assertRaises(ValueError):
            fuse_pitch_heave_feedforward(
                0.0, -10.0, 0.0, -28.0, -38.0, 0.001, 0.0625, 1.01, 3.0
            )

    def test_depth_heave_and_dual_attitude_moments_share_vertical_thrusters(self):
        allocator = make_allocator(
            vertical=0.3, vertical_x_m=0.0625, correction=0.5
        )
        result = allocator.allocate(
            tau_z=40.0,
            M_x=-1.5,
            M_y=1.5,
            priority_M_x=-1.5,
            priority_M_y=1.5,
            write_output=False,
        )

        self.assertAlmostEqual(result.achieved_tau[2], 40.0, places=6)
        self.assertLess(result.achieved_tau[3], 0.0)
        self.assertGreater(result.achieved_tau[4], 0.0)
        self.assertTrue(all(abs(value) <= 27.5 + 1e-9 for value in result.motor_thrusts[4:8]))

    def test_forward_trim_uses_vertical_pair_without_changing_lateral_allocation(self):
        baseline = make_allocator(
            horizontal=0.5, vertical=0.5, forward_trim=0.0
        ).allocate(tau_x=20.0, tau_z=4.0, write_output=False)
        trimmed = make_allocator(
            horizontal=0.5, vertical=0.5, forward_trim=-60.0
        ).allocate(tau_x=20.0, tau_z=4.0, write_output=False)

        # Forward propulsion remains equal on all four diagonal thrusters.
        for before, after in zip(
            baseline.nominal_thrusts[:4], trimmed.nominal_thrusts[:4]
        ):
            self.assertAlmostEqual(before, after, places=6)

        # Negative trim strengthens the front vertical pair and weakens the
        # rear pair around the unchanged fixed-hover total.
        self.assertGreater(
            trimmed.nominal_thrusts[4], trimmed.nominal_thrusts[6]
        )
        self.assertAlmostEqual(
            sum(trimmed.nominal_thrusts[4:]),
            sum(baseline.nominal_thrusts[4:]),
            places=6,
        )
        self.assertAlmostEqual(trimmed.achieved_tau[0], 20.0, places=6)
        self.assertLess(trimmed.achieved_tau[4], 0.0)

    def test_forward_vertical_trim_is_inactive_on_release(self):
        allocator = make_allocator(
            horizontal=0.5, vertical=0.5, forward_trim=60.0
        )
        result = allocator.allocate(tau_x=0.0, tau_z=4.0, write_output=False)
        self.assertTrue(all(
            abs(value - 1.0) < 1e-9
            for value in result.nominal_thrusts[4:]
        ))
        self.assertAlmostEqual(result.achieved_tau[4], 0.0, places=6)

    def test_backward_trim_uses_vertical_pair_and_lateral_motion_does_not(self):
        allocator = make_allocator(
            horizontal=0.5, vertical=0.5,
            forward_trim=0.0, backward_trim=50.0,
        )
        backward = allocator.allocate(
            tau_x=-20.0, tau_z=4.0, write_output=False
        )
        self.assertLess(
            backward.nominal_thrusts[4], backward.nominal_thrusts[6]
        )
        self.assertAlmostEqual(sum(backward.nominal_thrusts[4:]), 4.0)
        self.assertGreater(backward.achieved_tau[4], 0.0)

        lateral = allocator.allocate(
            tau_y=20.0, tau_z=4.0, write_output=False
        )
        self.assertTrue(all(
            abs(value - 1.0) < 1e-9
            for value in lateral.nominal_thrusts[4:]
        ))
        self.assertAlmostEqual(lateral.achieved_tau[4], 0.0, places=6)

    def test_horizontal_saturation_keeps_fixed_hover_and_forward_vertical_trim(self):
        baseline = make_allocator(
            horizontal=0.2, vertical=0.3, forward_trim=0.0
        ).allocate(tau_x=150.0, tau_z=4.0, write_output=False)
        trimmed = make_allocator(
            horizontal=0.2, vertical=0.3, forward_trim=60.0
        ).allocate(tau_x=150.0, tau_z=4.0, write_output=False)

        self.assertAlmostEqual(baseline.achieved_tau[2], 4.0, places=6)
        self.assertAlmostEqual(trimmed.achieved_tau[2], 4.0, places=6)
        self.assertAlmostEqual(
            baseline.achieved_tau[0], trimmed.achieved_tau[0], places=6
        )
        self.assertLess(trimmed.nominal_thrusts[4], 1.0)
        self.assertGreater(trimmed.nominal_thrusts[6], 1.0)

    def test_measured_directional_start_thresholds_are_preserved(self):
        allocator = make_allocator()
        self.assertEqual(allocator.thrust_to_pwm(-0.001, "vertical_fl"), 1532)
        self.assertEqual(allocator.thrust_to_pwm(0.001, "vertical_fl"), 1578)

    def test_pure_roll_and_pitch_moments_use_only_four_vertical_thrusters(self):
        allocator = make_allocator()
        for kwargs in ({"M_x": 2.0}, {"M_y": -2.0}, {"M_x": 1.5, "M_y": 1.0}):
            result = allocator.allocate(write_output=False, **kwargs)
            self.assertTrue(all(abs(value) < 1e-9 for value in result.nominal_thrusts[:4]))
            self.assertTrue(all(pulse == 1540 for pulse in result.pwm_us[:4]))
            self.assertTrue(any(abs(value) > 1e-6 for value in result.nominal_thrusts[4:]))


class PersistentLimitTests(unittest.TestCase):
    def test_runtime_apply_persists_fixed_vertical_compensation(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            shutil.copyfile(ROOT / "config.ini", config_path)
            rov = ROVController(config_path=config_path)
            rov.allocator = make_allocator()
            applied = rov._apply_motion_tuning({
                "horizontal_limit_fraction": 0.3,
                "vertical_limit_fraction": 0.25,
                "hover_force_per_thruster_n": 0.5,
                "vertical_ascent_front_rear_trim_percent": 5.0,
                "vertical_descent_front_rear_trim_percent": -35.0,
                "forward_front_rear_trim_percent": 20.0,
                "backward_front_rear_trim_percent": -10.0,
            })
            self.assertEqual(set(applied), {
                "horizontal_limit_fraction", "vertical_limit_fraction",
                "hover_force_per_thruster_n",
                "vertical_ascent_front_rear_trim_percent",
                "vertical_descent_front_rear_trim_percent",
                "forward_front_rear_trim_percent",
                "backward_front_rear_trim_percent",
            })
            saved = load_config(config_path)
            self.assertEqual(saved.getfloat("thrusters", "horizontal_limit_fraction"), 0.3)
            self.assertEqual(
                saved.getfloat("control", "vertical_hover_force_per_thruster_n"),
                0.5,
            )
            self.assertEqual(
                saved.getfloat(
                    "control", "vertical_descent_front_rear_trim_percent"
                ),
                -35.0,
            )
            self.assertEqual(
                saved.getfloat("control", "forward_front_rear_trim_percent"),
                20.0,
            )
            self.assertEqual(
                saved.getfloat("control", "backward_front_rear_trim_percent"),
                -10.0,
            )

    def test_attitude_limits_persist_without_enabling_a_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            shutil.copyfile(ROOT / "config.ini", config_path)
            rov = ROVController(config_path=config_path)
            original_rage_limits = {
                field: rov.performance_profiles["rage"][field]
                for field in (
                    "roll_max_moment_nm",
                    "pitch_max_moment_nm",
                    "yaw_max_moment_nm",
                )
            }
            actual = rov._apply_attitude_tuning({
                "roll_max_moment_nm": 3.0,
                "pitch_max_moment_nm": 2.5,
                "yaw_max_moment_nm": 3.5,
                "yaw_release_rate_threshold_dps": 2.5,
            })
            saved = load_config(config_path)
            self.assertEqual(actual["roll_max_moment_nm"], 3.0)
            self.assertEqual(saved.getfloat("control", "pitch_max_moment_nm"), 2.5)
            self.assertEqual(saved.getfloat("control", "yaw_max_moment_nm"), 3.5)
            self.assertEqual(
                saved.getfloat("mode_balance", "roll_max_moment_nm"), 3.0
            )
            self.assertEqual(
                saved.getfloat("mode_balance", "pitch_max_moment_nm"), 2.5
            )
            self.assertEqual(
                saved.getfloat("mode_balance", "yaw_max_moment_nm"), 3.5
            )
            self.assertEqual(
                rov.performance_profiles["balance"]["pitch_max_moment_nm"],
                2.5,
            )
            self.assertEqual(
                {
                    field: rov.performance_profiles["rage"][field]
                    for field in original_rage_limits
                },
                original_rage_limits,
            )
            self.assertEqual(
                saved.getfloat("control", "yaw_release_rate_threshold_dps"),
                2.5,
            )
            self.assertEqual(
                rov.attitude_hold,
                {"roll": False, "pitch": False, "yaw": False},
            )


if __name__ == "__main__":
    unittest.main()
