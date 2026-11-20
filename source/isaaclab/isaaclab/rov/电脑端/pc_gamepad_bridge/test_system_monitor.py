import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from mount_settings import rotation_matrix, parameter_files
from system_monitor import SystemMonitor, validate_report

script = Path(__file__).resolve().parent.parent / 'pi_competition/rov_system_stats.py'
spec = importlib.util.spec_from_file_location('pi_system_stats', script)
pi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pi)


class SystemTests(unittest.TestCase):
    def test_real_counter_math_cache_full_duplex_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nic = root / 'sys/class/net/eth0'
            (nic / 'statistics').mkdir(parents=True)
            (root / 'proc').mkdir()
            (root / 'proc/meminfo').write_text('MemTotal: 8192000 kB\nMemAvailable: 6144000 kB\n')
            (nic / 'speed').write_text('1000')
            (nic / 'carrier').write_text('1')
            rx, tx = nic / 'statistics/rx_bytes', nic / 'statistics/tx_bytes'
            rx.write_text('0'); tx.write_text('0')
            sampler = pi.Sampler(root=root)
            first = sampler.sample(1)
            self.assertIsNone(first['rx_mbps'])
            rx.write_text('12500000'); tx.write_text('6250000')
            second = sampler.sample(2)
            self.assertEqual(second['memory_percent'], 25)
            self.assertEqual(second['rx_mbps'], 100)
            self.assertEqual(second['tx_mbps'], 50)
            self.assertEqual(second['link_percent'], 10)  # not RX+TX=15%
            validate_report(second)
            rx.write_text('0')
            self.assertIsNone(sampler.sample(3)['link_percent'])
            (nic / 'carrier').write_text('0')
            self.assertIsNone(sampler.sample(4)['link_mbps'])

    def test_monitor_stale_or_invalid_is_unknown(self):
        report = dict(schema=1, interface='eth0', memory_percent=25, memory_total_mb=8000,
                      link_mbps=1000, rx_mbps=0, tx_mbps=3, link_percent=.3, temperature_c=60)
        validate_report(report)
        client = SystemMonitor('127.0.0.1')
        client.report, client.received = report, 10
        with patch('system_monitor.windows_memory', return_value=(40, 16)):
            self.assertIn('3.00 Mbps', client.text(10))
            self.assertIn('未知', client.text(14))
        for value in (float('nan'), -1, True):
            with self.assertRaises(ValueError):
                validate_report({**report, 'memory_percent': value})
        client.close()

    def test_sensor_to_body_rotation_and_parameter_units(self):
        matrix = rotation_matrix(0, 0, 90)
        self.assertAlmostEqual(matrix[0][0], 0, places=7)
        self.assertAlmostEqual(matrix[1][0], 1)  # sensor forward -> vehicle right
        matrix = rotation_matrix(15, -10, 25)
        for a in range(3):
            for b in range(3):
                self.assertAlmostEqual(sum(matrix[a][k]*matrix[b][k] for k in range(3)), int(a == b))
        first, second = parameter_files((15, -10, 25))
        self.assertIn('CUST_ROT_ENABLE\t1\t2', first)
        self.assertIn('CUST_ROT1_PITCH\t-10\t9', second)
        self.assertIn('AHRS_ORIENTATION\t101\t2', second)
        self.assertNotIn('TRIM', first + second)
        with self.assertRaises(ValueError):
            parameter_files((float('nan'), 0, 0))


if __name__ == '__main__':
    unittest.main()
