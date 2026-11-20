import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from control_profile import LocalActionController, load_profile
from flight_mode import FlightModeController, MODE_HOLDS
from install_qgc_fly_panel import competition_settings, install, set_instrument
from qgc_fly_status import CompetitionTelemetry, StatusServer, make_snapshot
from servo_control import ServoController
from system_monitor import SystemMonitor


class Message(SimpleNamespace):
    def get_srcSystem(self):
        return getattr(self, 'system', 1)
    def get_srcComponent(self):
        return getattr(self, 'component', 1)
    def get_type(self):
        return self.kind


class FlyStatusTests(unittest.TestCase):
    def setUp(self):
        self.modes = FlightModeController()
        self.modes.mode, self.modes.last_heartbeat = 19, 10
        self.telemetry = CompetitionTelemetry()
        self.actions = LocalActionController(load_profile())
        self.servos = ServoController.from_profile(load_profile())

    def snapshot(self, now=10, **kwargs):
        return make_snapshot(self.modes, self.telemetry, self.actions, self.servos, now,
                             hid_fresh=True, sending=False, ready=False, allow_arm=False, **kwargs)

    def test_four_real_modes_and_pending_do_not_fake_switches(self):
        self.modes.rov_values['ROV_DONLY'] = (1, 10)
        for mode, holds in MODE_HOLDS.items():
            self.modes.mode = mode
            snapshot = self.snapshot()
            self.assertEqual((snapshot['depth_hold'], snapshot['attitude_hold']), holds)
        self.modes.mode, self.modes.pending = 19, 2
        snapshot = self.snapshot()
        self.assertFalse(snapshot['depth_hold'])
        self.assertFalse(snapshot['attitude_hold'])
        self.assertIn('等待', snapshot['warning'])
        self.modes.mode = 22
        self.modes.rov_values.clear()
        self.assertIsNone(self.snapshot()['depth_hold'])

    def test_depth_sign_units_and_source(self):
        message = Message(kind='GLOBAL_POSITION_INT', relative_alt=-2345, vz=-12)
        self.telemetry.observe(message, 10)
        snapshot = self.snapshot()
        self.assertEqual(snapshot['depth_m'], 2.345)
        self.assertEqual(snapshot['descent_mps'], -.12)
        message.system, message.relative_alt = 2, -9999
        self.telemetry.observe(message, 10.1)
        self.assertEqual(self.snapshot()['depth_m'], 2.345)
        message.system, message.component = 1, 190
        self.telemetry.observe(message, 10.1)
        self.assertEqual(self.snapshot()['depth_m'], 2.345)
        message.component, message.relative_alt = 1, float('nan')
        self.telemetry.observe(message, 10.1)
        self.assertEqual(self.snapshot()['depth_m'], 2.345)

    def test_timeout_clears_mode_angles_depth_pwm_power(self):
        self.modes.attitude, self.modes.last_attitude = (1, 2, -1), 10
        self.modes.battery['primary'] = (3, 24, 10)
        self.telemetry.observe(Message(kind='GLOBAL_POSITION_INT', relative_alt=-2000, vz=0), 10)
        self.telemetry.observe(Message(kind='SERVO_OUTPUT_RAW', servo9_raw=700), 10)
        self.assertEqual(self.snapshot()['heading_deg'], 359)
        self.assertEqual(self.snapshot()['gripper_output_pwm'], 700)
        snapshot = self.snapshot(14)
        for key in ('armed', 'depth_hold', 'attitude_hold', 'roll_deg', 'depth_m',
                    'current_a', 'voltage_v', 'gripper_output_pwm', 'gripper_next'):
            self.assertIsNone(snapshot[key], key)
        self.assertFalse(snapshot['fc_online'])
        self.modes.last_heartbeat = 14
        self.assertIsNone(self.snapshot(14)['depth_m'])

    def test_fly_card_uses_same_freshest_valid_power_sample_as_status_window(self):
        self.modes.battery['primary'] = (2.0, 24.0, 9.5)
        self.modes.battery['system'] = (7.75, 23.8, 10.0)
        snapshot = self.snapshot(10.1)
        self.assertEqual(snapshot['current_a'], 7.75)
        self.assertEqual(snapshot['voltage_v'], 23.8)

    def test_next_rb_matches_slow_control_and_is_not_mechanical_position(self):
        for button, expected in ((17, '快速闭合'), (18, '快速张开')):
            self.servos.update(1 << button, 10, enabled=True)
            self.assertEqual(self.snapshot()['gripper_next'], expected)
        # Last FC PWM is independent from latest outgoing target.
        self.telemetry.observe(Message(kind='SERVO_OUTPUT_RAW', servo9_raw=601), 10)
        self.assertEqual(self.snapshot()['gripper_output_pwm'], 601)
        self.assertNotEqual(self.snapshot()['gripper_target_pwm'], 601)

    def test_critical_alerts_and_leak_are_reported_without_mutating_controls(self):
        self.telemetry.observe(Message(kind='SYS_STATUS', onboard_control_sensors_present_extended=2,
            onboard_control_sensors_enabled_extended=2, onboard_control_sensors_health_extended=0), 10)
        self.telemetry.observe(Message(kind='STATUSTEXT', severity=2, text=b'Deadlock\x00ignored'), 10)
        self.assertIn('Deadlock', self.snapshot()['warning'])
        self.assertIn('漏水', self.snapshot()['warning'])
        self.assertEqual(self.modes.mode, 19)
        self.assertFalse(self.modes.armed)
        self.modes.last_heartbeat = 30
        self.assertNotIn('Deadlock', self.snapshot(30)['warning'])
        self.assertNotIn('漏水', self.snapshot(30)['warning'])

    def test_hold_flags_are_truthful_but_health_faults_are_visible(self):
        self.modes.mode = 2
        self.modes.observe(Message(kind='SYS_STATUS', onboard_control_sensors_present=4,
            onboard_control_sensors_enabled=4, onboard_control_sensors_health=0,
            load=950), 10)
        snapshot = self.snapshot()
        self.assertTrue(snapshot['attitude_hold'])
        self.assertTrue(snapshot['depth_hold'])
        self.assertIn('设备异常', snapshot['warning'])
        self.assertIn('负载', snapshot['warning'])
        self.modes.last_heartbeat = 14
        self.assertNotIn('设备异常', self.snapshot(14)['warning'])

    def test_sensor_fault_one_is_labeled_as_gyro_not_compass(self):
        self.modes.observe(Message(kind='SYS_STATUS', onboard_control_sensors_present=1,
            onboard_control_sensors_enabled=1, onboard_control_sensors_health=0,
            load=100), 10)
        warning = self.snapshot()['warning']
        self.assertIn('0x00000001（陀螺仪）', warning)
        self.assertNotIn('罗盘', warning)

    def test_warning_messages_are_reported_and_critical_is_not_overwritten(self):
        self.telemetry.observe(Message(kind='STATUSTEXT', severity=4, text='sensor warning'), 10)
        self.assertIn('sensor warning', self.snapshot()['warning'])
        self.telemetry.observe(Message(kind='STATUSTEXT', severity=2, text='critical fault'), 10.1)
        self.telemetry.observe(Message(kind='STATUSTEXT', severity=4, text='minor warning'), 10.2)
        self.assertIn('critical fault', self.snapshot(10.2)['warning'])

    def test_resource_cache_expiry_keeps_only_pc_memory(self):
        monitor = SystemMonitor('127.0.0.1')
        monitor.pc_memory, monitor.next_pc = (42, 16), 100
        monitor.report, monitor.received = {'temperature_c': 85, 'memory_percent': 20}, 10
        self.assertEqual(monitor.snapshot(10)['temperature_c'], 85)
        self.assertEqual(monitor.snapshot(14), {'pc_memory_percent': 42})
        self.assertIn('散热', self.snapshot(resources=monitor.snapshot(10))['warning'])


class ServerTests(unittest.TestCase):
    def test_read_only_loopback_and_publish_expiry(self):
        server = StatusServer(port=0)
        self.addCleanup(server.close)
        address = f'http://127.0.0.1:{server.port}/status'
        self.assertEqual(server.server.server_address[0], '127.0.0.1')
        self.assertTrue(json.load(urlopen(address, timeout=1))['bridge_stale'])
        server.publish({'schema': 1, 'target_system': 1}, time.monotonic())
        with urlopen(address, timeout=1) as response:
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
            self.assertEqual(json.load(response)['target_system'], 1)
        server.updated -= 2
        self.assertTrue(json.load(urlopen(address, timeout=1))['bridge_stale'])
        for request, code in ((Request(address, method='POST'), 501),
                              (Request(address, headers={'Host': 'example.com'}), 403),
                              (Request(address.replace('/status', '/command')), 404)):
            with self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=1)
            self.assertEqual(error.exception.code, code)
            error.exception.close()


class InstallerTests(unittest.TestCase):
    def test_compass_is_unchanged_installed_qgc_resource_and_separate_from_bridge(self):
        instrument = (Path(__file__).with_name('qgc') / 'ROVCompetitionInstrument.qml').read_text(encoding='utf-8')
        self.assertIn('qrc:/qml/QGroundControl/FlightMap/Widgets/IntegratedCompassAttitude.qml', instrument)
        self.assertIn('ROVCompetitionCard {', instrument)
        self.assertIn('onLoaded: item.vehicle = Qt.binding', instrument)
        self.assertIn('QGroundControl.multiVehicleManager.activeVehicle', instrument)
        self.assertNotIn('snapshot.heading', instrument)
        self.assertNotIn('Canvas {', instrument)
        self.assertNotIn('Behavior on', instrument)

    def test_precise_section_replacement_idempotent_and_reversible(self):
        original = ('[General]\npassword=@ByteArray(unchanged)\n\n'
                    '[FlyView]\nkeepMapCenteredOnVehicle=true\ninstrumentQmlFile2=qrc:/old.qml\n\n'
                    '[TelemetryBarUserSettings-12]\nrowCount=2\ncolumns\\size=4\n\n'
                    '[Video]\nvideoSource=UDP h.264\n\n')
        qml = Path(__file__).with_name('qgc') / 'ROVCompetitionInstrument.qml'
        updated = competition_settings(original, qml)
        self.assertEqual(competition_settings(updated, qml), updated)
        self.assertIn('password=@ByteArray(unchanged)', updated)
        self.assertIn('keepMapCenteredOnVehicle=true', updated)
        self.assertIn('[Video]\nvideoSource=UDP h.264', updated)
        self.assertIn('factName=FlightTime', updated)
        self.assertNotIn('GroundSpeed', updated)
        with tempfile.TemporaryDirectory() as folder:
            ini = Path(folder) / 'QGroundControl.ini'
            ini.write_text(original, encoding='utf-8')
            with patch('install_qgc_fly_panel.qgc_running', return_value=False):
                install(ini)
                self.assertEqual(ini.read_text(encoding='utf-8'), updated)
                install(ini)
                install(ini, restore=True)
            self.assertEqual(ini.read_text(encoding='utf-8'), original)

    def test_running_qgc_is_not_modified(self):
        with tempfile.TemporaryDirectory() as folder:
            ini = Path(folder) / 'QGroundControl.ini'
            ini.write_text('[General]\nkeep=1\n', encoding='utf-8')
            with patch('install_qgc_fly_panel.qgc_running', return_value=True), self.assertRaises(RuntimeError):
                install(ini)
            self.assertEqual(ini.read_text(encoding='utf-8'), '[General]\nkeep=1\n')
            self.assertEqual(len(list(Path(folder).iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
