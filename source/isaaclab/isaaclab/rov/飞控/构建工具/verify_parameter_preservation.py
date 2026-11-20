"""Offline comparison of complete real snapshots; never writes to the vehicle."""
import argparse
import copy
import json
import math
from pathlib import Path
from activate_deadzone import same_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gimbal-reversed', action='store_true')
    parser.add_argument('--allow-boot-stream-reset', action='store_true')
    parser.add_argument('--allow-boot-gyro-recalibration', action='store_true')
    parser.add_argument('--allow-depth-imax-reset', action='store_true')
    parser.add_argument('--allow-flight-statistics', action='store_true')
    parser.add_argument('--allow-live-ground-pressure', action='store_true')
    parser.add_argument('--mode-report', type=Path)
    parser.add_argument('--expect-tilt-tolerance', type=float)
    parser.add_argument('--current-guard', action='store_true')
    parser.add_argument('--expect-current-max', type=float, default=15.0)
    parser.add_argument('--expect-current-tc', type=float, default=0.5)
    parser.add_argument('--gentle-attitude', action='store_true')
    parser.add_argument('--expect-angle-p', type=float, default=2.0)
    parser.add_argument('--expect-rp-max', type=float, default=25.0)
    parser.add_argument('--gentle-depth', action='store_true')
    parser.add_argument('--expect-depth-max', type=float, default=35.0)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Evidence already exists')
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    expected = copy.deepcopy(before)
    additions = set()
    permitted = {}
    if args.current_guard:
        for name, value, kind in (('MOT_H_REV', 1.0, 2), ('ROV_LVL_R', 0.0, 9),
                                  ('ROV_LVL_P', 0.0, 9)):
            item = after['parameters'].get(name)
            if item is None or item['type'] != kind or not math.isclose(item['value'], value, abs_tol=1e-7):
                raise RuntimeError('Current-guard firmware addition missing/mismatched: '+name)
            if name not in before['parameters']:
                additions.add(name)
        for name, value in (('MOT_BAT_CURR_MAX', args.expect_current_max),
                            ('MOT_BAT_CURR_TC', args.expect_current_tc)):
            if not math.isfinite(value):
                raise RuntimeError('Invalid expected current setting')
            item = after['parameters'][name]
            if item['type'] != 9 or not math.isclose(item['value'], value, abs_tol=1e-7):
                raise RuntimeError('Current guard readback mismatch: '+name)
            expected['parameters'][name]['value'] = value
            permitted[name] = {'before': before['parameters'][name]['value'], 'after': value}
    if args.gentle_attitude:
        cap = after['parameters'].get('MOT_RP_MAX')
        if (cap is None or cap['type'] != 9 or not math.isfinite(args.expect_rp_max)
                or not 0 <= args.expect_rp_max <= 100
                or not math.isclose(cap['value'], args.expect_rp_max, abs_tol=1e-7)):
            raise RuntimeError('Gentle attitude cap readback mismatch')
        if 'MOT_RP_MAX' not in before['parameters']:
            additions.add('MOT_RP_MAX')
        if not math.isfinite(args.expect_angle_p) or not 0.5 <= args.expect_angle_p <= 6.0:
            raise RuntimeError('Invalid expected angle P')
        for name in ('ATC_ANG_RLL_P', 'ATC_ANG_PIT_P'):
            item = after['parameters'].get(name)
            if item is None or item['type'] != 9 or not math.isclose(
                    item['value'], args.expect_angle_p, abs_tol=1e-7):
                raise RuntimeError('Gentle attitude gain readback mismatch: ' + name)
            expected['parameters'][name]['value'] = args.expect_angle_p
            permitted[name] = {'before': before['parameters'][name]['value'],
                               'after': args.expect_angle_p}
    if args.gentle_depth:
        cap = after['parameters'].get('MOT_DH_MAX')
        if (cap is None or cap['type'] != 9 or not math.isfinite(args.expect_depth_max)
                or not 0 <= args.expect_depth_max <= 100
                or not math.isclose(cap['value'], args.expect_depth_max, abs_tol=1e-7)):
            raise RuntimeError('Gentle depth cap readback mismatch')
        if 'MOT_DH_MAX' not in before['parameters']:
            additions.add('MOT_DH_MAX')
    if args.expect_tilt_tolerance is not None:
        if not 0.0 <= args.expect_tilt_tolerance <= 5.0:
            raise RuntimeError('Invalid expected tilt tolerance')
        param = after['parameters'].get('ROV_T_ERR')
        if param is None or param['type'] != 9 or not math.isclose(
                param['value'], args.expect_tilt_tolerance, abs_tol=1e-5):
            raise RuntimeError('ROV_T_ERR readback missing or mismatched')
        if 'ROV_T_ERR' not in before['parameters']:
            additions.add('ROV_T_ERR')
    volatile_stream = {}
    volatile_controller = {}
    volatile_gyro = {}
    volatile_statistics = {}
    volatile_pressure = {}
    boot_delta = after['parameters']['STAT_BOOTCNT']['value'] - before['parameters']['STAT_BOOTCNT']['value']
    reboot_observed = 0 < boot_delta <= 10
    if args.allow_boot_gyro_recalibration:
        if not reboot_observed:
            raise RuntimeError('Gyro recalibration waiver requires proven reboot')
        for name in ('INS_GYROFFS_X', 'INS_GYROFFS_Y', 'INS_GYROFFS_Z',
                     'INS_GYR2OFFS_X', 'INS_GYR2OFFS_Y', 'INS_GYR2OFFS_Z',
                     'INS_GYR1_CALTEMP', 'INS_GYR2_CALTEMP'):
            old, new = before['parameters'][name], after['parameters'][name]
            if old == new:
                continue
            limit = 5.0 if 'CALTEMP' in name else 0.01
            if (old['type'] != new['type'] or old['type'] != 9
                    or not math.isfinite(old['value']) or not math.isfinite(new['value'])
                    or abs(new['value'] - old['value']) > limit):
                raise RuntimeError('Unreviewed gyro calibration change: ' + name)
            expected['parameters'][name]['value'] = new['value']
            volatile_gyro[name] = {'before': old['value'], 'after': new['value'],
                                   'reason': 'bounded gyro recalibration after confirmed reboot'}
    if args.allow_depth_imax_reset:
        if not reboot_observed:
            raise RuntimeError('Depth-controller RAM reset waiver requires proven reboot')
        name = 'PSC_D_ACC_IMAX'
        old, new = before['parameters'][name], after['parameters'][name]
        if old != new:
            if not (old['type'] == new['type'] == 9
                    and math.isclose(old['value'], .5, abs_tol=1e-7)
                    and math.isclose(new['value'], .1, abs_tol=1e-7)):
                raise RuntimeError('Unreviewed depth-controller RAM reset')
            expected['parameters'][name]['value'] = new['value']
            volatile_controller[name] = {'before': old['value'], 'after': new['value'],
                'reason': 'native RAM IMAX reverted to saved 0.1 on reboot; no PARAM_SET sent'}
    if args.mode_report:
        modes = json.loads(args.mode_report.read_text())
        if (modes['ROV_DONLY'] != 1 or modes['armed'] is not False or modes['final_mode'] != 19
                or [r['mode'] for r in modes['modes']] != [19, 0, 2, 22, 2, 0, 19]
                or not all(r['confirmed_disarmed'] is True for r in modes['modes'])):
            raise RuntimeError('Independent mode check evidence missing')
        name = 'PSC_D_ACC_IMAX'
        old, new = before['parameters'][name], after['parameters'][name]
        if old != new:
            # Pinned AC_PosControl.cpp:1119-1122 raises only this RAM limit to
            # throttle_hover during D_update_controller(). AC_PID::set_imax()
            # uses AP_Float.set(), not save(). Never waive unrelated PID tuning.
            if not (old['type'] == new['type'] == 9 and math.isclose(old['value'], .1, abs_tol=1e-7)
                    and new['value'] == .5 == after['parameters']['MOT_THST_HOVER']['value']):
                raise RuntimeError('Unreviewed depth-controller change')
            expected['parameters'][name]['value'] = new['value']
            volatile_controller[name] = {'before': old['value'], 'after': new['value'],
                'reason': 'native D_update_controller raises RAM IMAX to hover throttle; no PARAM_SET sent'}
    if args.allow_boot_stream_reset:
        # Pinned GCS_Param.cpp handle_request_data_stream() calls rate->set(),
        # not save(), because ArduSub inherits persist_streamrates()==false.
        # This observed QGC 20Hz RAM override resets to its stored 10Hz at boot.
        # Only this exact pair/type, across a proven reboot, is permitted here.
        name = 'MAV1_EXTRA1'
        old, new = before['parameters'][name], after['parameters'][name]
        if old != new:
            if not (old['value'] == 20 and new['value'] == 10 and old['type'] == new['type'] == 4 and 0 < boot_delta <= 10):
                raise RuntimeError('Unreviewed telemetry-rate change')
            expected['parameters'][name]['value'] = new['value']
            volatile_stream[name] = {'before': 20, 'after': 10,
                'reason': 'nonpersistent REQUEST_DATA_STREAM override reset on reboot; no PARAM_SET sent'}
    if args.allow_flight_statistics:
        # Read-only persistent statistics may advance between an older backup
        # and this maintenance window.  Keep the waiver exact, monotonic and
        # bounded; it never covers a control/configuration parameter.
        for name, bound in (('STAT_FLTTIME', 86400), ('STAT_FLTCNT', 100)):
            old, new = before['parameters'][name], after['parameters'][name]
            if old == new:
                continue
            old_value, new_value = float(old['value']), float(new['value'])
            if (old['type'] != new['type'] or old['type'] != 6
                    or not math.isfinite(old_value) or not math.isfinite(new_value)
                    or not old_value.is_integer() or not new_value.is_integer()
                    or not 0 <= new_value - old_value <= bound):
                raise RuntimeError('Unreviewed flight-statistic change: ' + name)
            expected['parameters'][name]['value'] = new['value']
            volatile_statistics[name] = {'before': old_value, 'after': new_value,
                'reason': 'monotonic read-only vehicle usage statistic'}
    if args.allow_live_ground_pressure:
        # AP_Baro ground pressure is a live calibration value.  It can move a
        # few pascals between two complete parameter reads even without a boot.
        for name in ('BARO1_GND_PRESS', 'BARO2_GND_PRESS', 'BARO3_GND_PRESS'):
            old = before['parameters'].get(name)
            new = after['parameters'].get(name)
            if old is None or new is None or old == new:
                continue
            old_value, new_value = float(old['value']), float(new['value'])
            if (old['type'] != new['type'] or old['type'] != 9
                    or not math.isfinite(old_value) or not math.isfinite(new_value)
                    or not 80000 <= old_value <= 120000
                    or not 80000 <= new_value <= 120000
                    or abs(new_value - old_value) > 2000):
                raise RuntimeError('Unreviewed live ground-pressure change: ' + name)
            expected['parameters'][name]['value'] = new['value']
            volatile_pressure[name] = {'before': old_value, 'after': new_value,
                'reason': 'bounded live barometer ground-pressure calibration'}
    if args.gimbal_reversed:
        old = before['parameters']['SERVO12_REVERSED']
        new = after['parameters']['SERVO12_REVERSED']
        if old['type'] != 2 or old['value'] not in (0, 1) or new['type'] != 2 or new['value'] != 1:
            raise RuntimeError('Unexpected gimbal direction configuration')
        expected['parameters']['SERVO12_REVERSED']['value'] = 1.0
        permitted['SERVO12_REVERSED'] = {'before': old['value'], 'after': new['value']}
    counters = same_parameters(expected, after, expected_additions=additions)
    report = {'verified': True, 'before_count': before['parameter_count'], 'after_count': after['parameter_count'],
              'uid_unchanged': True, 'permitted_configuration_changes': permitted, 'volatile_changes': counters,
              'volatile_stream_reset': volatile_stream, 'volatile_depth_controller': volatile_controller,
              'volatile_gyro_recalibration': volatile_gyro,
              'volatile_flight_statistics': volatile_statistics,
              'volatile_ground_pressure': volatile_pressure,
              'real_water_acceptance': False}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
