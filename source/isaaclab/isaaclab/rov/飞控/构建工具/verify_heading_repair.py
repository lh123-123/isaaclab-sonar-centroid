"""Offline strict comparison after native gyro calibration; never sends commands."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    folder = args.directory
    before = json.loads((folder/'postflash_final.json').read_text())['parameters']
    final = json.loads((folder/'heading_native_final.json').read_text())
    after = final['complete_parameters']
    calibration = json.loads((folder/'gyro_calibration.json').read_text())
    setting = json.loads((folder/'gyro_startup_verified.json').read_text())
    assert len(before) == len(after) == final['parameter_count'] == 1004
    assert set(before) == set(after)
    assert calibration['ack']['result'] == 0 and calibration['armed'] is False
    assert setting['independent_readback'] is True and setting['after'] == 1
    assert final['armed'] is False and final['mode'] == 19
    assert final['summary']['max_native_display_error_deg'] == 0
    permitted, differences = {}, {}
    for name, old in before.items():
        new = after[name]
        assert old['type'] == new['type'], name
        a, b = old['value'], new['value']
        if a == b:
            continue
        if name in calibration['after'] and name.startswith(('INS_GYROFFS_', 'INS_GYR2OFFS_')):
            assert b == calibration['after'][name] and a == calibration['before'][name], name
            reason = 'native gyro calibration ACK and exact readback'
        elif name == 'INS_GYR_CAL':
            assert a == 0 and b == 1 and new['type'] == 2
            reason = 'one explicitly verified startup-calibration parameter'
        elif name in ('INS_GYR1_CALTEMP', 'INS_GYR2_CALTEMP'):
            assert new['type'] == 9 and 0 < a < 70 and 0 < b < 70
            reason = 'native init_gyro saves actual calibration temperature'
        elif name in ('BARO1_GND_PRESS', 'BARO2_GND_PRESS'):
            assert 90000 < a < 110000 and 90000 < b < 110000 and abs(b-a) <= 2000
            reason = 'native ground pressure update'
        elif name == 'STAT_RUNTIME':
            assert 0 <= b-a < 7200
            reason = 'monotonic runtime counter'
        elif name == 'MAV1_EXTRA1':
            assert a == 10 and b == 20 and new['type'] == 4
            reason = 'QGC nonpersistent REQUEST_DATA_STREAM RAM rate; no PARAM_SET to this setting'
        else:
            raise RuntimeError('Unreviewed configuration difference: '+name)
        differences[name] = {'before': a, 'after': b, 'reason': reason}
        if name.startswith(('INS_GYR', 'INS_GYRO')):
            permitted[name] = b
    report = {'verified': True, 'complete_parameter_count': 1004,
              'changes': differences, 'firmware_reflashed': False,
              'pid_compass_frame_servo_motor_mapping_unchanged': True,
              'stationary_result': final['summary'], 'physical_rotation_retest_pending': True}
    (folder/'heading_repair_preservation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
