import math
import random
import unittest
from collections import deque
from dataclasses import replace

from control_profile import load_profile
from flydigi_apex5 import Apex5State, build_manual_control
from motion_input import MotionSlewLimiter, read_latest_input


def command(**changes):
    return replace(build_manual_control(Apex5State.neutral()), **changes)


def report(y=0, buttons=0):
    raw = bytearray(32)
    raw[:4] = bytes((4, 0x5A, 0xA5, 0xEF))
    raw[6:8] = (-y).to_bytes(2, 'little', signed=True)
    raw[12] = buttons
    return bytes(raw)


class QueueDevice:
    def __init__(self, reports):
        self.queue = deque(reports)

    def read(self, _size):
        return self.queue.popleft() if self.queue else []


class HidBatchTests(unittest.TestCase):
    def test_latest_center_replaces_history_and_does_not_replay_arm(self):
        device = QueueDevice([report(20000, 0x10)] * 10 + [report()])
        batch = read_latest_input(device)
        self.assertFalse(batch.backlogged)
        self.assertEqual(batch.state, Apex5State.neutral())

    def test_overflow_never_marks_old_motion_fresh(self):
        device = QueueDevice([report(20000)] * 100 + [report()])
        batch = read_latest_input(device)
        self.assertTrue(batch.backlogged)
        self.assertIsNone(batch.state)
        recovered = read_latest_input(device)
        self.assertFalse(recovered.backlogged)
        self.assertEqual(recovered.state, Apex5State.neutral())

    def test_quick_disarm_is_latched_and_neutralizes_other_commands(self):
        batch = read_latest_input(QueueDevice([report(20000, 0x20), report(20000)]))
        self.assertTrue(batch.disarm_requested)
        self.assertEqual(batch.state, replace(Apex5State.neutral(), buttons=2))

    def test_disarm_survives_queue_overflow_without_faking_freshness(self):
        batch = read_latest_input(QueueDevice([report(20000, 0x20)] * 100))
        self.assertTrue(batch.disarm_requested)
        self.assertTrue(batch.backlogged)
        self.assertIsNone(batch.state)

    def test_invalid_packets_do_not_refresh_state(self):
        batch = read_latest_input(QueueDevice([b'bad', bytes(32)]))
        self.assertIsNone(batch.state)
        self.assertFalse(batch.backlogged)


class MotionSlewTests(unittest.TestCase):
    def test_depth_hold_receives_neutral_on_first_release_not_smoothing_tail(self):
        for end in (0, 1000):
            limiter = MotionSlewLimiter()
            for i in range(31):
                limiter.update(command(x=1000, z=end), i*.01, active=True, depth_hold=True)
            released = limiter.update(command(), .31, active=True, depth_hold=True)
            self.assertEqual(released.z, 500)
            self.assertGreater(released.x, 0)  # retain horizontal smoothing
            self.assertEqual(limiter.vertical, 0)
            again = limiter.update(command(z=end), .32, active=True, depth_hold=True)
            self.assertLess(abs(again.z-500), 25)  # restart from neutral

    def test_full_authority_after_time_based_ramp_at_both_poll_rates(self):
        for dt in (.01, .04):
            limiter = MotionSlewLimiter()
            target = command(x=1000, z=0)
            limiter.update(target, 0, active=True)
            previous = command()
            for i in range(1, round(.4/dt) + 1):
                output = limiter.update(target, i*dt, active=True)
                self.assertLessEqual(output.x-previous.x, math.ceil(4000*dt)+1)
                self.assertLessEqual(previous.z-output.z, math.ceil(2000*dt)+1)
                previous = output
            self.assertEqual((output.x, output.z), (1000, 0))
            for i in range(1, round(.2/dt) + 1):
                output = limiter.update(command(), .4+i*dt, active=True)
            self.assertEqual((output.x, output.z), (0, 500))

    def test_diagonal_ratio_is_preserved_on_acceleration(self):
        limiter = MotionSlewLimiter()
        target = command(x=800, y=600)
        limiter.update(target, 0, active=True)
        for i in range(1, 31):
            output = limiter.update(target, i*.01, active=True)
            self.assertAlmostEqual(output.x*.6, output.y*.8, delta=1)
        self.assertEqual((output.x, output.y), (800, 600))

    def test_reversal_passes_neutral_before_opposite_thrust(self):
        limiter = MotionSlewLimiter()
        for i in range(31):
            output = limiter.update(command(x=1000, z=1000), i*.01, active=True)
        self.assertEqual(output.x, 1000)
        values = []
        for i in range(1, 71):
            output = limiter.update(command(x=-1000, z=0), .3+i*.01, active=True)
            values.append(output.x)
        self.assertIn(0, values)
        self.assertEqual(values[-1], -1000)
        self.assertEqual(output.z, 0)

    def test_yaw_and_pitch_release_are_not_delayed(self):
        limiter = MotionSlewLimiter()
        moving = command(x=1000, r=1000, s=-400)
        first = limiter.update(moving, 0, active=True)
        self.assertEqual((first.r, first.s), (1000, -400))
        limiter.update(moving, .05, active=True)
        released = limiter.update(command(), .06, active=True)
        self.assertGreater(released.x, 0)  # translation deceleration only
        self.assertEqual((released.r, released.s, released.t), (0, 0, 0))

    def test_disarm_loss_and_scheduler_gap_remove_smoothing_tail(self):
        for stop in ('disarm', 'inactive', 'gap', 'invalid_time'):
            limiter = MotionSlewLimiter()
            target = command(x=1000, y=0, z=0, r=1000, s=400)
            for i in range(31):
                limiter.update(target, i*.01, active=True)
            value = replace(target, buttons=2) if stop == 'disarm' else target
            now = 1.0 if stop == 'gap' else float('nan') if stop == 'invalid_time' else .31
            result = limiter.update(value, now, active=stop != 'inactive')
            self.assertEqual((result.x, result.y, result.z), (0, 0, 500))
            if stop != 'gap':
                self.assertEqual((result.r, result.s), (0, 0))
            if stop == 'disarm':
                self.assertEqual(result.buttons, 2)
            resumed = limiter.update(target, 1.1, active=True)
            self.assertEqual(resumed.x, 0)

    def test_lower_gear_caps_immediately_and_can_reach_its_endpoint(self):
        limiter = MotionSlewLimiter()
        for i in range(31):
            limiter.update(command(x=1000, z=1000), i*.01, active=True)
        output = limiter.update(command(x=400, z=700), .31, active=True,
                                motion_gain=.4, vertical_gain=.4)
        self.assertEqual((output.x, output.z), (400, 700))
        for i in range(1, 31):
            output = limiter.update(command(x=1000, z=1000), .31+i*.01, active=True)
        self.assertEqual((output.x, output.z), (1000, 1000))

    def test_random_commands_remain_in_current_gear_and_protocol_bounds(self):
        limiter = MotionSlewLimiter()
        rng = random.Random(104)
        for i in range(3000):
            gear = rng.choice((.4, .6, .8, 1.0))
            angle, radius = rng.uniform(-math.pi, math.pi), rng.random() * gear
            target = command(x=round(1000*radius*math.cos(angle)),
                             y=round(1000*radius*math.sin(angle)), z=rng.randrange(1001))
            out = limiter.update(target, i*.01, active=True,
                                 motion_gain=gear, vertical_gain=gear)
            self.assertLessEqual(math.hypot(out.x, out.y), gear*1000+1)
            self.assertLessEqual(abs(out.z-500), gear*500+1)

    def test_config_validation_and_disable_preserve_existing_mapping(self):
        self.assertTrue(MotionSlewLimiter.from_profile(load_profile()).enabled)
        for settings in ({'enabled': 1}, {'rise_per_s': 0}, {'fall_per_s': float('nan')}):
            with self.assertRaises(ValueError):
                MotionSlewLimiter.from_profile({'motion_smoothing': settings})
        limiter = MotionSlewLimiter(enabled=False)
        target = command(x=1000, z=0, r=-1000, s=400)
        self.assertEqual(limiter.update(target, 0, active=True), target)
        stopped = limiter.update(replace(target, buttons=2), .01, active=True)
        self.assertEqual((stopped.x, stopped.z, stopped.r, stopped.s), (0,500,0,0))


if __name__ == '__main__':
    unittest.main()
