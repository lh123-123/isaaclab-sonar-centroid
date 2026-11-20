import sys
import threading
import unittest
from unittest import mock

sys.modules.setdefault("serial", mock.Mock())
from pi_runtime.imu_depth_combined import DepthSensor


class DepthSensorConversionTests(unittest.TestCase):
    def _sensor_at_pressure(self, pressure_mpa, *, zero_mpa=0.0, meters_per_mpa=101.972):
        sensor = DepthSensor.__new__(DepthSensor)
        sensor._lock = threading.Lock()
        sensor.pressure_mpa = pressure_mpa
        sensor.depth_raw_zero_mpa = zero_mpa
        sensor.depth_m_per_mpa = meters_per_mpa
        return sensor

    def test_pressure_is_converted_using_configured_surface_reference(self):
        sensor = self._sensor_at_pressure(0.0982, zero_mpa=0.09705)
        self.assertAlmostEqual(sensor.get_depth(), 0.1172678, places=6)

    def test_default_scale_preserves_legacy_gauge_conversion(self):
        sensor = self._sensor_at_pressure(0.01)
        self.assertAlmostEqual(sensor.get_depth(), 1.01972, places=6)

    def test_surface_reference_can_be_updated_without_changing_pressure(self):
        sensor = self._sensor_at_pressure(0.0038, zero_mpa=0.0033)
        sensor.set_surface_reference(0.0038)
        self.assertEqual(sensor.get_pressure(), 0.0038)
        self.assertAlmostEqual(sensor.get_depth(), 0.0, places=9)

    def test_raw_frame_parsing_preserves_air_value_for_telemetry(self):
        sensor = DepthSensor.__new__(DepthSensor)
        sensor._lock = threading.Lock()
        sensor.address = 1
        sensor.pressure_mpa = 0.0
        sensor.temperature_c = 0.0

        result = sensor._parse(b"sD01100001974026750043")

        self.assertEqual(result, (-0.1974, 26.75))
        self.assertEqual(sensor.get_pressure(), -0.1974)

    def test_raw_frame_with_invalid_vendor_checksum_is_rejected(self):
        sensor = DepthSensor.__new__(DepthSensor)
        sensor._lock = threading.Lock()
        sensor.address = 1
        sensor.pressure_mpa = 0.0
        sensor.temperature_c = 0.0

        self.assertIsNone(sensor._parse(b"sD01100001974026750044"))

    def test_stream_extractor_resynchronizes_after_noise_and_fragmentation(self):
        sensor = DepthSensor.__new__(DepthSensor)
        sensor._lock = threading.Lock()
        sensor.address = 1
        sensor.pressure_mpa = 0.0
        sensor.temperature_c = 0.0
        sensor._rx_buffer = bytearray(b"echo\\r\\n")
        sensor.depth_invalid_frames = 0

        self.assertIsNone(sensor._extract_frame())
        sensor._rx_buffer.extend(b"sD01100001974026750043"[:9])
        self.assertIsNone(sensor._extract_frame())
        sensor._rx_buffer.extend(b"sD01100001974026750043"[9:])
        self.assertEqual(sensor._extract_frame(), (-0.1974, 26.75))

    def test_stream_extractor_skips_invalid_candidate_and_finds_next_frame(self):
        sensor = DepthSensor.__new__(DepthSensor)
        sensor._lock = threading.Lock()
        sensor.address = 1
        sensor.pressure_mpa = 0.0
        sensor.temperature_c = 0.0
        sensor._rx_buffer = bytearray(
            b"sD01100001974026750044sD01100001974026750043"
        )
        sensor.depth_invalid_frames = 0

        self.assertEqual(sensor._extract_frame(), (-0.1974, 26.75))


if __name__ == "__main__":
    unittest.main()
