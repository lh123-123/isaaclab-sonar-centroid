"""Read-only Pi SSH stream + Windows memory; no work on the control thread."""
import ctypes
import ipaddress
import json
import math
import os
import subprocess
import threading
import time


def windows_memory():
    if os.name != 'nt':
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
            (name, ctypes.c_uint64) for name in ('physical', 'available', 'page', 'page_available',
                                               'virtual', 'virtual_available', 'extended')]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.load, status.physical / 1024**3


def validate_report(report):
    if not isinstance(report, dict) or report.get('schema') != 1 or report.get('interface') != 'eth0':
        raise ValueError('监测数据版本或网口不符')
    bounds = {'memory_percent': (0, 100), 'memory_total_mb': (1, 1e7),
              'link_mbps': (1, 1e6), 'rx_mbps': (0, 1e6), 'tx_mbps': (0, 1e6),
              'link_percent': (0, 1000), 'temperature_c': (-50, 150)}
    for key, (lo, hi) in bounds.items():
        if key not in report:
            raise ValueError('监测字段缺失')
        value = report[key]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi):
            raise ValueError('监测数值异常')
    return report


class SystemMonitor:
    def __init__(self, host, enabled=False, source_ip='192.168.1.4'):
        self.host = str(ipaddress.IPv4Address(host))
        self.source_ip = str(ipaddress.IPv4Address(source_ip))
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.report, self.received, self.error = None, float('-inf'), '等待树莓派监测'
        self.process = None
        self.pc_memory, self.next_pc = None, 0
        if enabled:
            threading.Thread(target=self._run, daemon=True, name='rov-system-monitor').start()

    def _run(self):
        while not self.stop.is_set():
            try:
                process = subprocess.Popen([
                    'ssh', '-b', self.source_ip, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4',
                    '-o', 'ServerAliveInterval=2', '-o', 'ServerAliveCountMax=2', 'liang@' + self.host,
                    'python3 -u /home/liang/underwater_robot/rov_system_stats.py --stream'],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                with self.lock:
                    self.process = process
                    if self.stop.is_set():
                        process.terminate()
                for line in process.stdout:
                    if self.stop.is_set():
                        break
                    if len(line) > 4096:
                        raise ValueError('监测数据过长')
                    report = validate_report(json.loads(line))
                    with self.lock:
                        self.report, self.received, self.error = report, time.monotonic(), ''
                if not self.stop.is_set():
                    raise RuntimeError('树莓派监测连接中断或脚本未部署')
            except (OSError, ValueError, RuntimeError) as exc:
                with self.lock:
                    self.error = str(exc)
            finally:
                if self.process is not None:
                    self.process.terminate() if self.process.poll() is None else None
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=2)
                    if self.process.stdout:
                        self.process.stdout.close()
                    with self.lock:
                        self.process = None
            self.stop.wait(5)

    def snapshot(self, now):
        """Return numeric read-only fields; never reuse expired Pi samples."""
        if now >= self.next_pc:
            self.pc_memory, self.next_pc = windows_memory(), now + 1
        with self.lock:
            report, received = self.report, self.received
        result = dict(report) if report is not None and 0 <= now - received <= 3 else {}
        result['pc_memory_percent'] = self.pc_memory[0] if self.pc_memory else None
        return result

    def text(self, now):
        self.snapshot(now)
        pc = ('PC内存：未知' if self.pc_memory is None else
              f'PC内存：{self.pc_memory[0]:.0f}% / {self.pc_memory[1]:.1f} GB')
        with self.lock:
            report, received, error = self.report, self.received, self.error
        if report is None or not 0 <= now - received <= 3:
            return pc + '\n树莓派内存/网口：未知（' + (error or '回传超时') + '）'
        memory, temp = report['memory_percent'], report['temperature_c']
        lines = [pc, '树莓派内存：' + (f'{memory:.1f}%' if memory is not None else '未知') +
                 (f' · 温度 {temp:.1f}°C' if temp is not None else '')]
        rx, tx, percent, speed = (report[k] for k in ('rx_mbps', 'tx_mbps', 'link_percent', 'link_mbps'))
        lines.append('Pi网口：' + (f'↓接收 {rx:.2f} / ↑发送 {tx:.2f} Mbps' if rx is not None and tx is not None else '采样中'))
        lines.append('带宽占用：' + (f'{percent:.2f}% / 协商 {speed:g} Mbps（双工单向峰值）'
                                       if percent is not None and speed else '未知（链路速度未提供）'))
        return '\n'.join(lines)

    def close(self):
        self.stop.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
