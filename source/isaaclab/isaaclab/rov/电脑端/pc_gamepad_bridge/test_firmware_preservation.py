import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'firmware'))
from verify_parameter_preservation import main


class PreservationTests(unittest.TestCase):
    def snapshot(self):
        fields = {'STAT_BOOTCNT': (95, 6), 'STAT_RUNTIME': (61800, 6),
                  'MAV1_EXTRA1': (20, 4), 'PSC_D_ACC_IMAX': (.10000000149, 9),
                  'MOT_THST_HOVER': (.5, 9), 'SERVO12_REVERSED': (0, 2), 'FRAME_CONFIG': (2, 2),
                  'MOT_BAT_CURR_MAX': (0, 9), 'MOT_BAT_CURR_TC': (5, 9),
                  'ATC_ANG_RLL_P': (6, 9), 'ATC_ANG_PIT_P': (6, 9)}
        return {'parameters': {n: {'value': v, 'type': t, 'index': i} for i, (n, (v, t)) in enumerate(fields.items())},
                'parameter_count': len(fields), 'autopilot_version': {'uid2': [1, 2, 3]}}

    def run_compare(self, before, after, flags=(), mode=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in (('before', before), ('after', after), ('modes', mode)):
                (root / (name + '.json')).write_text(json.dumps(data), encoding='utf-8')
            argv = ['verify', '--before', str(root / 'before.json'), '--after', str(root / 'after.json'),
                    '--output', str(root / 'report.json'), *flags]
            if mode:
                argv.extend(('--mode-report', str(root / 'modes.json')))
            with patch.object(sys, 'argv', argv):
                main()
            return json.loads((root / 'report.json').read_text())

    def test_no_general_configuration_waiver(self):
        before = self.snapshot()
        after = copy.deepcopy(before)
        after['parameters']['FRAME_CONFIG']['value'] = 1
        with self.assertRaises(RuntimeError):
            self.run_compare(before, after)

    def test_exact_transient_rate_only_across_reboot(self):
        before = self.snapshot()
        after = copy.deepcopy(before)
        after['parameters']['MAV1_EXTRA1']['value'] = 10
        with self.assertRaises(RuntimeError):
            self.run_compare(before, after, ('--allow-boot-stream-reset',))
        after['parameters']['STAT_BOOTCNT']['value'] += 1
        report = self.run_compare(before, after, ('--allow-boot-stream-reset',))
        self.assertTrue(report['verified'])
        after['parameters']['MAV1_EXTRA1']['value'] = 30
        with self.assertRaises(RuntimeError):
            self.run_compare(before, after, ('--allow-boot-stream-reset',))

    def test_exact_native_imax_with_mode_evidence_not_arbitrary_pid(self):
        before = self.snapshot()
        after = copy.deepcopy(before)
        after['parameters']['PSC_D_ACC_IMAX']['value'] = .5
        with self.assertRaises(RuntimeError):
            self.run_compare(before, after)
        mode = {'ROV_DONLY': 1, 'armed': False, 'final_mode': 19,
                'modes': [{'mode': n, 'confirmed_disarmed': True} for n in (19, 0, 2, 22, 2, 0, 19)]}
        report = self.run_compare(before, after, mode=mode)
        self.assertIn('PSC_D_ACC_IMAX', report['volatile_depth_controller'])
        after['parameters']['PSC_D_ACC_IMAX']['value'] = .6
        with self.assertRaises(RuntimeError):
            self.run_compare(before, after, mode=mode)

    def test_only_new_tolerance_parameter_may_be_added(self):
        before = self.snapshot()
        after = copy.deepcopy(before)
        after['parameters']['ROV_T_ERR'] = {'value': .5, 'type': 9,
                                            'index': before['parameter_count']}
        after['parameter_count'] += 1
        self.assertTrue(self.run_compare(before, after,
                        ('--expect-tilt-tolerance', '0.5'))['verified'])
        with self.assertRaises(RuntimeError):
            self.run_compare(before, after)
        after['parameters']['ROV_T_ERR']['value'] = .7
        with self.assertRaises(RuntimeError):
            self.run_compare(before, after, ('--expect-tilt-tolerance', '0.5'))

    def test_current_guard_allows_only_exact_additions_and_settings(self):
        before = self.snapshot()
        after = copy.deepcopy(before)
        for name, value, kind in (('MOT_H_REV', 1, 2), ('ROV_LVL_R', 0, 9), ('ROV_LVL_P', 0, 9)):
            after['parameters'][name] = {'value': value, 'type': kind,
                                         'index': after['parameter_count']}
            after['parameter_count'] += 1
        after['parameters']['MOT_BAT_CURR_MAX']['value'] = 15
        after['parameters']['MOT_BAT_CURR_TC']['value'] = .5
        report = self.run_compare(before, after, ('--current-guard',))
        self.assertEqual(set(report['permitted_configuration_changes']),
                         {'MOT_BAT_CURR_MAX', 'MOT_BAT_CURR_TC'})
        for name, bad in (('MOT_H_REV', 0), ('ROV_LVL_R', 11), ('MOT_BAT_CURR_MAX', 14)):
            broken = copy.deepcopy(after)
            broken['parameters'][name]['value'] = bad
            with self.assertRaises(RuntimeError):
                self.run_compare(before, broken, ('--current-guard',))

    def test_bounded_boot_gyro_and_depth_ram_reset_require_reboot(self):
        before = self.snapshot()
        gyro_names = ('INS_GYROFFS_X', 'INS_GYROFFS_Y', 'INS_GYROFFS_Z',
                      'INS_GYR2OFFS_X', 'INS_GYR2OFFS_Y', 'INS_GYR2OFFS_Z',
                      'INS_GYR1_CALTEMP', 'INS_GYR2_CALTEMP')
        for index, name in enumerate(gyro_names, start=7):
            before['parameters'][name] = {'value': 57 if 'CALTEMP' in name else .05,
                                          'type': 9, 'index': index}
        before['parameters']['PSC_D_ACC_IMAX']['value'] = .5
        before['parameter_count'] += len(gyro_names)
        after = copy.deepcopy(before)
        after['parameters']['INS_GYROFFS_X']['value'] = .052
        after['parameters']['INS_GYR1_CALTEMP']['value'] = 55
        after['parameters']['PSC_D_ACC_IMAX']['value'] = .1
        flags = ('--allow-boot-gyro-recalibration', '--allow-depth-imax-reset')
        with self.assertRaises(RuntimeError):
            self.run_compare(before, after, flags)
        after['parameters']['STAT_BOOTCNT']['value'] += 1
        report = self.run_compare(before, after, flags)
        self.assertIn('INS_GYROFFS_X', report['volatile_gyro_recalibration'])
        self.assertIn('PSC_D_ACC_IMAX', report['volatile_depth_controller'])
        after['parameters']['INS_GYROFFS_X']['value'] = .08
        with self.assertRaises(RuntimeError):
            self.run_compare(before, after, flags)

    def test_gentle_attitude_allows_only_two_gains_and_new_cap(self):
        before = self.snapshot()
        after = copy.deepcopy(before)
        after['parameters']['MOT_RP_MAX'] = {
            'value': 25, 'type': 9, 'index': after['parameter_count']}
        after['parameter_count'] += 1
        after['parameters']['ATC_ANG_RLL_P']['value'] = 2
        after['parameters']['ATC_ANG_PIT_P']['value'] = 2
        report = self.run_compare(before, after, ('--gentle-attitude',))
        self.assertEqual(set(report['permitted_configuration_changes']),
                         {'ATC_ANG_RLL_P', 'ATC_ANG_PIT_P'})
        for name, bad in (('MOT_RP_MAX', 30), ('ATC_ANG_RLL_P', 3),
                          ('ATC_ANG_PIT_P', 3)):
            broken = copy.deepcopy(after)
            broken['parameters'][name]['value'] = bad
            with self.assertRaises(RuntimeError):
                self.run_compare(before, broken, ('--gentle-attitude',))


if __name__ == '__main__':
    unittest.main()
