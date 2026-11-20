"""Offline mixer sign checks, NOT measured thrust or physical acceptance."""
from pathlib import Path
import re
import unittest

from control_profile import load_profile
from motor_test import MOTION_FACTORS, MOTION_GROUPS, motion_pwm_plan


class MotionGeometryTests(unittest.TestCase):
    def setUp(self):
        self.params = dict(load_profile()['pixhawk_button_parameters'], MOT_H_REV=0)

    def test_all_motion_factors_match_actual_pinned_firmware_source(self):
        path = Path(__file__).parent / 'firmware' / 'prepared_independent_20261005' / 'source' / 'libraries' / 'AP_Motors' / 'AP_Motors6DOF.cpp'
        source = path.read_text(encoding='utf-8')
        block = source.split('case SUB_FRAME_VECTORED_6DOF:')[1].split('break;')[0]
        rows = re.findall(r'add_motor_raw_6dof\(AP_MOTORS_MOT_(\d),\s*(.*?)\);', block)
        factors = {int(index): tuple(float(value.strip().removesuffix('f'))
                                    for value in values.split(','))[:6]
                   for index, values in rows}
        self.assertEqual(set(factors), set(range(1,9)))
        axes = {'前进': (4,1), '后退': (4,-1), '上升': (3,1), '下降': (3,-1),
                '右移': (5,1), '左移': (5,-1), '右转': (2,1), '左转': (2,-1),
                '抬头': (1,1), '低头': (1,-1), '右倾': (0,1), '左倾': (0,-1)}
        self.assertEqual(set(MOTION_FACTORS), set(axes))
        for action, (column, sign) in axes.items():
            self.assertEqual(MOTION_FACTORS[action],
                             tuple(sign*factors[i][column] for i in range(1,9)), action)

    def test_physical_pwm_patterns_and_reverse_symmetry(self):
        expected = {
            '前进': (1,-1,1,-1,0,0,0,0),
            '右移': (1,1,-1,-1,0,0,0,0),
            '右转': (1,1,1,1,0,0,0,0),
            '下降': (0,0,0,0,-1,1,1,-1),
            '抬头': (0,0,0,0,1,-1,1,-1),
            '右倾': (0,0,0,0,1,1,-1,-1),
        }
        for forward, reverse in (('前进','后退'), ('右移','左移'), ('右转','左转'),
                                 ('下降','上升'), ('抬头','低头'), ('右倾','左倾')):
            first = tuple((value-1500)//100 for value in motion_pwm_plan(forward,1400,1600,self.params).values())
            second = tuple((value-1500)//100 for value in motion_pwm_plan(reverse,1400,1600,self.params).values())
            self.assertEqual(first, expected[forward], forward)
            self.assertEqual(first, tuple(-value for value in second), reverse)

    def test_descent_baseline_implies_correct_vertical_moments_under_axial_geometry(self):
        # Conditional physical model: vertical motors at the stated four corners,
        # positive PWM offsets create the inverse of the verified descent wrench.
        # Body axes are forward/right/down; positive pitch is nose up.
        points = ((1,-1), (1,1), (-1,-1), (-1,1))
        descent = tuple(value-1500 for value in motion_pwm_plan('下降',1400,1600,self.params).values())[4:]
        for action, expected in (('下降',(4,0,0)), ('上升',(-4,0,0)),
                                 ('抬头',(0,0,4)), ('低头',(0,0,-4)),
                                 ('右倾',(0,4,0)), ('左倾',(0,-4,0))):
            delta = tuple(value-1500 for value in motion_pwm_plan(action,1400,1600,self.params).values())[4:]
            force = tuple(value/base for value,base in zip(delta,descent))
            wrench = (sum(force), sum(y*f for (_,y),f in zip(points,force)),
                      sum(-x*f for (x,_),f in zip(points,force)))
            self.assertEqual(wrench, expected, action)

    def test_test_categories_are_disjoint_and_complete(self):
        actions = [action for group in MOTION_GROUPS.values() for action in group]
        self.assertEqual(len(actions), len(set(actions)))
        self.assertEqual(set(actions), set(MOTION_FACTORS))
        self.assertTrue(all(len(group)==4 for group in MOTION_GROUPS.values()))


if __name__ == '__main__':
    unittest.main()
