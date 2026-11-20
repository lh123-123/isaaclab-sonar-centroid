import sys
import threading
import unittest
from unittest import mock

sys.modules.setdefault("serial", mock.Mock())
from pi_runtime.imu_depth_combined import DepthSensor


FRAME = b"sD01100001974026750043"


class _BrokenSerial:
    is_open = True

    def write(self, _data):
        return None

    def flush(self):
        return None

    @property
    def in_waiting(self):
        raise TypeError("argument must be an int, or have a fileno() method")

    def close(self):
        self.is_open = False


class _WorkingSerial:
    is_open = True

    def __init__(self):
        self._read = FRAME

    def write(self, _data):
        return None

    def flush(self):
        return None

    @property
    def in_waiting(self):
        return len(self._read)

    def read(self, count):
        data, self._read = self._read[:count], self._read[count:]
        return data

    def close(self):
        self.is_open = False


class DepthSerialRecoveryTests(unittest.TestCase):
    def test_in_waiting_type_error_is_reported_without_raising(self):
        with mock.patch("pi_runtime.imu_depth_combined.serial.Serial", return_value=_BrokenSerial()):
            sensor = DepthSensor("/dev/depth", 19200, 1)
            self.assertIsNone(sensor.query())
            self.assertEqual(sensor.depth_last_error, "serial_io_error:TypeError")

    def test_next_query_reopens_after_a_broken_serial_handle(self):
        broken = _BrokenSerial()
        working = _WorkingSerial()
        with mock.patch(
            "pi_runtime.imu_depth_combined.serial.Serial",
            side_effect=[broken, working],
        ) as serial_open:
            sensor = DepthSensor("/dev/depth", 19200, 1)
            self.assertIsNone(sensor.query())
            self.assertEqual(sensor.depth_last_error, "serial_io_error:TypeError")
            self.assertEqual(sensor.query(), (-0.1974, 26.75))
            self.assertEqual(serial_open.call_count, 2)

    def test_missing_port_is_retryable_and_does_not_block_startup(self):
        with mock.patch(
            "pi_runtime.imu_depth_combined.serial.Serial",
            side_effect=OSError("no such device"),
        ):
            sensor = DepthSensor("/dev/missing", 19200, 1)
            self.assertIsNone(sensor.ser)
            self.assertIsNone(sensor.query())
            self.assertEqual(sensor.depth_last_error, "serial_open_failed:OSError")


if __name__ == "__main__":
    unittest.main()
