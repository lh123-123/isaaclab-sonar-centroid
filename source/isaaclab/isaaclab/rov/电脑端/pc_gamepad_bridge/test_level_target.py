import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from level_target import LevelTarget, PARAMS, COMMAND, MAGIC


def message(kind, **kw):
    return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: 1,
                           get_srcComponent=lambda: 1, **kw)


class LevelTargetTests(unittest.TestCase):
    def setUp(self):
        self.target = LevelTarget()
        self.link = Mock()
        self.modes = SimpleNamespace(fresh=lambda now: True, armed=False, mode=19, pending=None)

    def feed_params(self, pair, now):
        for name, value in zip(PARAMS, pair):
            self.target.observe(message('PARAM_VALUE', param_id=name, param_value=value, param_type=9), now)

    def stationary(self):
        for i in range(11):
            self.target.observe(message('ATTITUDE', roll=math.radians(2), pitch=math.radians(-3),
                rollspeed=0, pitchspeed=0, yawspeed=0), 10 + i * .1)

    def test_capture_ack_then_independent_readback_no_sensor_writes(self):
        self.target.read(self.link, 9)
        self.feed_params((0, 0), 9.2)
        self.stationary()
        self.target.capture(self.link, self.modes, 11.1, stopped=True)
        self.link.mav.command_int_send.assert_called_once_with(1, 1, 2, COMMAND, 0, 0, MAGIC, 1, 0, 0, 0, 0, 0)
        self.feed_params((2, -3), 11.2)  # unsolicited pre-ACK values cannot confirm capture
        self.assertEqual(self.target.phase, 'capture')
        self.target.observe(message('COMMAND_ACK', command=COMMAND, result=0), 11.3)
        self.assertEqual(self.target.phase, 'verify')
        self.feed_params((2.01, -2.99), 11.4)
        self.assertEqual(self.target.actual, (2.01, -2.99))
        self.assertFalse(self.target.busy)
        self.link.mav.param_set_send.assert_not_called()
        self.link.mav.command_long_send.assert_not_called()

    def test_gates(self):
        self.stationary()
        with self.assertRaises(ValueError):
            self.target.capture(self.link, self.modes, 11, stopped=True)
        self.target.actual = (0, 0)
        for name, value in [('armed', True), ('mode', 0), ('pending', 19)]:
            original = getattr(self.modes, name)
            setattr(self.modes, name, value)
            with self.assertRaises(ValueError):
                self.target.capture(self.link, self.modes, 11, stopped=True)
            setattr(self.modes, name, original)
        with self.assertRaises(ValueError):
            self.target.capture(self.link, self.modes, 11, stopped=False)
        with self.assertRaises(ValueError):
            self.target.capture(self.link, self.modes, 15, stopped=True)
        self.link.mav.command_int_send.assert_not_called()

    def test_timeout_reject_and_mismatch_never_success(self):
        for outcome in ('timeout', 'reject', 'mismatch'):
            self.setUp()
            self.target.actual = (0, 0)
            self.stationary()
            self.target.capture(self.link, self.modes, 11, stopped=True)
            if outcome == 'timeout':
                self.target.tick(self.modes, 16)
            else:
                self.target.observe(message('COMMAND_ACK', command=COMMAND, result=0 if outcome == 'mismatch' else 2), 11.1)
                if outcome == 'mismatch':
                    self.feed_params((0, 0), 11.2)
            self.assertIsNone(self.target.actual)
            self.assertFalse(self.target.busy)

    def test_moving_or_excess_tilt_is_not_captured(self):
        self.target.actual = (0, 0)
        for rate, roll in ((.1, 2), (0, 11)):
            self.target.samples.clear()
            for i in range(11):
                self.target.observe(message('ATTITUDE', roll=math.radians(roll), pitch=0,
                    rollspeed=rate, pitchspeed=0, yawspeed=0), 10+i*.1)
            self.assertFalse(self.target.stable(11))


class FirmwareLevelTests(unittest.TestCase):
    def test_real_target_kernel_returns_to_captured_angles(self):
        sys.path.insert(0, str(Path(__file__).parent / 'firmware'))
        from prepare_level_target import sources, KERNEL
        original, updated = sources()
        scenario = r'''
#include "ROV_PilotKernel.h"
#include <cassert>
using namespace ROV_PilotKernel;
int main() {
    Config c; c.level_roll=2; c.level_pitch=-3;
    Targets t; t.enter(false,2,-3,123);
    Output o;
    for(int i=0;i<1000;++i) o=t.update(.01f,2,-3,123,0,0,0,true,c);
    assert(fabs(o.roll-2)<.001 && fabs(o.pitch+3)<.001 && fabs(o.yaw-123)<.001);
    for(int i=0;i<500;++i) o=t.update(.01f,2,-3,123,.8f,-.8f,0,true,c);
    assert(hypot(o.roll,o.pitch)<=30.001);
    for(int i=0;i<1500;++i) o=t.update(.01f,9,8,145,0,0,0,true,c);
    assert(fabs(o.roll-2)<.001 && fabs(o.pitch+3)<.001 && fabs(o.yaw-123)<.001);
    // No moving reference: disturbances never recapture R/P or yaw.
    t.update(.01f,2,-3,140,0,0,1,true,c);
    o=t.update(.01f,2,-3,142,0,0,0,true,c);
    assert(fabs(o.yaw-142)<.001);
    c.level_roll=11; assert(!t.update(.01f,0,0,0,0,0,0,true,c).valid);
}
'''
        with tempfile.TemporaryDirectory(prefix='rov_level_') as folder:
            path = Path(folder)
            (path/'ROV_PilotKernel.h').write_text(updated[KERNEL], encoding='utf8')
            (path/'test.cpp').write_text(scenario, encoding='utf8')
            subprocess.run([shutil.which('g++') or 'C:/MinGW/bin/g++.exe', '-std=c++11', str(path/'test.cpp'), '-o', str(path/'test.exe')], check=True, capture_output=True)
            subprocess.run([str(path/'test.exe')], check=True, capture_output=True)
        changed = {k for k in original if original[k] != updated[k]}
        self.assertEqual(changed, {KERNEL, 'ArduSub/ROV_PilotHold.h', 'ArduSub/ROV_PilotHold.cpp', 'ArduSub/GCS_MAVLink_Sub.cpp'})
        self.assertNotIn('AHRS_TRIM', updated['ArduSub/ROV_PilotHold.cpp'])


if __name__ == '__main__':
    unittest.main()
