import unittest
import math
from types import SimpleNamespace
from unittest.mock import Mock

from flight_mode import FlightModeController, health_fault_description
from rov_gamepad_bridge import poll_telemetry, request_mode, request_power_telemetry


def heartbeat(mode=0, system=1, component=1, vehicle_type=12):
    return SimpleNamespace(
        get_srcSystem=lambda: system, get_srcComponent=lambda: component,
        get_type=lambda: "HEARTBEAT", custom_mode=mode, base_mode=0,
        type=vehicle_type, autopilot=3,
    )


class ModeTests(unittest.TestCase):
    def test_health_bit_one_identifies_gyro_not_compass(self):
        self.assertIn('陀螺仪', health_fault_description(0x00000001))
        self.assertNotIn('罗盘', health_fault_description(0x00000001))
        self.assertIn('罗盘', health_fault_description(0x00000004))

    def test_attitude_display_ignores_old_packets_and_never_creates_datum(self):
        state = FlightModeController()
        msg = SimpleNamespace(get_srcSystem=lambda: 1, get_srcComponent=lambda: 1,
                              get_type=lambda: 'ATTITUDE', roll=0, pitch=0, yaw=0,
                              time_boot_ms=10000)
        for i in range(8):
            msg.time_boot_ms = 10000 + 100 * i
            state.observe(msg, 10 + i * .1)
        self.assertIsNone(state.attitude_display.reference)  # no heartbeat, no zero
        state.observe(heartbeat(), 11)
        for i in range(8):
            msg.time_boot_ms = 11000 + 100 * i
            state.observe(msg, 11 + i * .1)
        self.assertIsNone(state.attitude_display.reference)
        last = state.last_attitude
        msg.time_boot_ms -= 100
        msg.yaw = 1
        state.observe(msg, 11.9)
        self.assertEqual(state.last_attitude, last)
        self.assertEqual(state.attitude[2], 0)

    def test_health_mask_ignores_disabled_devices_and_expires(self):
        state = FlightModeController()
        state.observe(heartbeat(), 10)
        msg = SimpleNamespace(get_srcSystem=lambda: 1, get_srcComponent=lambda: 1,
                              get_type=lambda: 'SYS_STATUS', onboard_control_sensors_present=15,
                              onboard_control_sensors_enabled=3, onboard_control_sensors_health=1, load=925)
        state.observe(msg, 10)
        self.assertIn('异常位 0x00000002', state.health_status(10))
        self.assertIn('92.5%（负载偏高）', state.health_status(10))
        msg.onboard_control_sensors_enabled = 1
        msg.load = 65535
        state.observe(msg, 11)
        self.assertIn('不代表校准或验收完成', state.health_status(11))
        self.assertIn('负载未知', state.health_status(11))
        msg.get_srcSystem = lambda: 2
        state.observe(msg, 12)
        self.assertEqual(state.system_health[2], 11)
        state.observe(heartbeat(), 15)
        self.assertIn('未知', state.health_status(15))

    @staticmethod
    def named(name, value):
        return SimpleNamespace(get_srcSystem=lambda: 1, get_srcComponent=lambda: 1,
                               get_type=lambda: "NAMED_VALUE_FLOAT", name=name, value=value)

    def test_actual_attitude_finite_fresh_and_correct_vehicle_only(self):
        state = FlightModeController()
        state.observe(heartbeat(), 10)
        attitude = SimpleNamespace(get_srcSystem=lambda: 1, get_srcComponent=lambda: 1,
                                   get_type=lambda: "ATTITUDE", roll=0.1, pitch=-0.2, yaw=-1)
        state.observe(attitude, 10)
        self.assertIn("俯仰 -11.5°", state.attitude_status(10.1))
        self.assertIn("未知", state.attitude_status(11.1))
        attitude.pitch = math.nan
        state.observe(attitude, 10.5)
        self.assertEqual(state.last_attitude, 10)
        attitude.pitch = 0
        attitude.get_srcComponent = lambda: 190
        state.observe(attitude, 10.7)
        self.assertEqual(state.last_attitude, 10)

    def test_enhancement_requires_actual_fresh_execution_confirmation(self):
        state = FlightModeController()
        packet = heartbeat()
        packet.base_mode = 128
        state.observe(packet, 10)
        self.assertIn("未确认", state.pilot_control_status(10))
        state.observe(self.named(b"ROV_EN\0", 1), 10)
        self.assertIn("状态未知", state.pilot_control_status(10))
        state.observe(self.named("ROV_ACT", 1), 10)
        self.assertIn("目标回传超时", state.pilot_control_status(10))
        state.observe(self.named("ROV_PTGT", -12), 10)
        state.observe(self.named("ROV_YTGT", -90), 10)
        self.assertIn("目标俯仰 -12.0° / 航向 270.0°", state.pilot_control_status(10.1))
        state.observe(self.named("ROV_PTGT", math.inf), 10.5)
        self.assertEqual(state.rov_values["ROV_PTGT"], (-12.0, 10))
        state.observe(heartbeat(19), 10.8)
        self.assertIn("不提供", state.pilot_control_status(10.8))
        state.observe(packet, 14)
        self.assertIn("未确认", state.pilot_control_status(14))
        state.observe(self.named("ROV_EN", 0), 14)
        self.assertIn("标准回平", state.pilot_control_status(14))

    def test_missing_stale_or_wrong_vehicle_cannot_enable_depth(self):
        state = FlightModeController()
        self.assertIsNone(state.request("depth_toggle", 10))
        for packet in (heartbeat(system=2), heartbeat(component=190), heartbeat(vehicle_type=6)):
            state.observe(packet, 10)
            self.assertFalse(state.fresh(10))
        state.observe(heartbeat(), 10)
        self.assertIsNone(state.request("depth_toggle", 14))

    def test_toggle_uses_actual_mode_and_requires_heartbeat_confirmation(self):
        state = FlightModeController()
        state.observe(heartbeat(), 10)
        self.assertEqual(state.request("depth_toggle", 10), 2)
        self.assertEqual(state.mode, 0)
        self.assertIsNone(state.request("depth_toggle", 10.5))
        state.observe(heartbeat(2), 11)
        self.assertIsNone(state.pending)
        self.assertEqual(state.request("depth_toggle", 11), 0)

    def test_timeout_does_not_fake_success_or_retry(self):
        state = FlightModeController()
        state.observe(heartbeat(), 10)
        state.request("depth_toggle", 10)
        state.observe(heartbeat(), 12)
        state.tick(13.1)
        self.assertEqual(state.mode, 0)
        self.assertIsNone(state.pending)
        self.assertIn("未确认", state.notice)

    def test_independent_toggles_all_four_combinations(self):
        state = FlightModeController()
        for initial, action, expected in ((19, 'depth_toggle', 22), (19, 'attitude_toggle', 0),
                (0, 'depth_toggle', 2), (0, 'attitude_toggle', 19), (2, 'depth_toggle', 0),
                (2, 'attitude_toggle', 22), (22, 'depth_toggle', 19), (22, 'attitude_toggle', 2)):
            with self.subTest(initial=initial, action=action):
                state.pending = None
                state.observe(heartbeat(initial), 10)
                state.observe(self.named('ROV_DONLY', 1), 10)
                self.assertEqual(state.request(action, 10), expected)
                self.assertEqual(state.mode, initial)
                self.assertIsNone(state.request(action, 10.1))
                state.observe(heartbeat(expected), 10.2)
                self.assertIsNone(state.pending)
                self.assertEqual(state.mode, expected)

    def test_old_firmware_never_fakes_depth_only(self):
        state = FlightModeController()
        state.observe(heartbeat(19), 10)
        self.assertIsNone(state.request('depth_toggle', 10))
        self.assertIn('当前固件', state.notice)
        state.observe(self.named('ROV_DONLY', 1), 5)
        self.assertIsNone(state.request('depth_toggle', 10))
        self.assertEqual(state.request('attitude_toggle', 10), 0)

    def test_default_off_needs_actual_manual_confirmation_and_disarmed(self):
        state = FlightModeController()
        packet = heartbeat(2)
        packet.base_mode = 128
        state.observe(packet, 10)
        self.assertIsNone(state.request('reset_manual', 10))
        packet.base_mode = 0
        state.observe(packet, 11)
        self.assertEqual(state.request('reset_manual', 11), 19)
        self.assertEqual(state.mode, 2)
        state.observe(heartbeat(19), 11.5)
        self.assertIn('两项关闭', state.status(11.5))

    def test_current_units_invalid_source_and_expiry(self):
        state = FlightModeController()
        state.observe(heartbeat(19), 10)
        msg = SimpleNamespace(get_srcSystem=lambda: 1, get_srcComponent=lambda: 1,
            get_type=lambda: 'BATTERY_STATUS', id=0, current_battery=1234, voltages=[24000, 65535])
        state.observe(msg, 10)
        self.assertIn('12.34 A', state.power_status(10))
        self.assertIn('24.00 V', state.power_status(10))
        msg.current_battery = -1
        state.observe(msg, 11)
        self.assertIn('电流计', state.power_status(11))
        msg.id = 1
        msg.current_battery = 999
        state.observe(msg, 11)
        self.assertIn('电流计', state.power_status(11))
        self.assertIn('超时', state.power_status(15))

    def test_power_uses_freshest_valid_current_instead_of_fixed_message_priority(self):
        state = FlightModeController()
        state.observe(heartbeat(19), 10)
        # BATTERY_STATUS commonly arrives more slowly than SYS_STATUS.  A fresh
        # but older primary packet must not hide a newer current measurement.
        state.battery['primary'] = (3.0, 24.0, 10.0)
        state.battery['system'] = (8.25, 23.8, 10.8)
        self.assertEqual(state.power_sample(11), (8.25, 23.8, 10.8))
        self.assertIn('8.25 A', state.power_status(11))
        # If the newest packet has no current field, retain the newest valid
        # current rather than showing an artificial missing/zero value.
        state.battery['system'] = (None, 23.7, 11.0)
        self.assertEqual(state.power_sample(11.1), (3.0, 24.0, 10.0))

    def test_rejected_command_reports_failure(self):
        state = FlightModeController()
        state.observe(heartbeat(), 10)
        state.request("depth_toggle", 10)
        ack = SimpleNamespace(get_srcSystem=lambda: 1, get_srcComponent=lambda: 1,
                              get_type=lambda: "COMMAND_ACK", command=176, result=2)
        state.observe(ack, 11)
        self.assertEqual(state.mode, 0)
        self.assertIsNone(state.pending)
        self.assertIn("拒绝", state.notice)

    def test_monitor_never_requests_mode(self):
        state = FlightModeController()
        state.observe(heartbeat(), 10)
        request_mode(None, state, "depth_toggle", 10)
        self.assertIsNone(state.pending)
        self.assertEqual(state.mode, 0)

    def test_transport_command_and_readback(self):
        state = FlightModeController()
        receiver = Mock()
        receiver.recv_match.side_effect = [heartbeat(), None, heartbeat(2), None]
        poll_telemetry(receiver, state, 10)
        link = Mock()
        request_mode(link, state, "depth_toggle", 10)
        link.mav.command_long_send.assert_called_once_with(1, 1, 176, 0, 1, 2, 0, 0, 0, 0, 0)
        self.assertEqual(state.mode, 0)
        poll_telemetry(receiver, state, 11)
        self.assertEqual(state.mode, 2)

    def test_power_telemetry_is_requested_at_five_hz(self):
        link = Mock()
        request_power_telemetry(link, 1)
        self.assertEqual(link.mav.command_long_send.call_count, 2)
        calls = link.mav.command_long_send.call_args_list
        self.assertEqual(calls[0].args[4:6], (1, 200000))
        self.assertEqual(calls[1].args[4:6], (147, 200000))

    def test_pilot_gain_is_actual_telemetry_not_boot_parameter_guess(self):
        state = FlightModeController()
        self.assertIn("未知", state.pilot_gain_status(10))
        packet = SimpleNamespace(get_srcSystem=lambda: 1, get_srcComponent=lambda: 1,
                                 get_type=lambda: "NAMED_VALUE_FLOAT", name=b"PilotGain\x00", value=0.5)
        state.observe(packet, 10)
        self.assertIn("50%", state.pilot_gain_status(11))
        self.assertIn("未知", state.pilot_gain_status(14))
        self.assertIsNone(state.pending)


if __name__ == "__main__":
    unittest.main()
