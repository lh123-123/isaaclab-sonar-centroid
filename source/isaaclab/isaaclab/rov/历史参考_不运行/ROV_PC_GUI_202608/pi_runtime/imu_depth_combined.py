"""
IMU + 深度计 联合读取程序
IMU:  GPIO串口 /dev/serial0, 115200bps, 二进制协议
深度计: TTL转USB /dev/ttyUSB0, 19200bps, ASCII协议
"""

import serial
import time
import math
import json
import threading

from attitude_math import tilt_compensated_magnetic_heading_deg


# ============================================================
#  IMU Reader（复用原有逻辑）
# ============================================================
class IMUReader:
    CMD_PACKET_BEGIN = 0x49
    CMD_PACKET_END = 0x4D
    CMD_PACKET_MAX_DATA_SIZE_RX = 73

    def __init__(self, port="/dev/serial0", baudrate=115200, timeout=0.2):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.cs = 0
        self.rx_index = 0
        self.buf = bytearray(5 + self.CMD_PACKET_MAX_DATA_SIZE_RX)
        self.i = 0
        self.cmd_len = 0
        self.latest = None  # 最新解析结果

        self.mag_calibration = {
            "offset_x": 0.0, "offset_y": 0.0, "offset_z": 0.0,
            "scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0,
        }

    @staticmethod
    def _to_int16(lo, hi):
        value = ((hi & 0xFF) << 8) | (lo & 0xFF)
        if value >= 0x8000:
            value -= 0x10000
        return value

    @staticmethod
    def _wrap_angle_180(angle):
        while angle > 180: angle -= 360
        while angle < -180: angle += 360
        return angle

    @staticmethod
    def _wrap_angle_360(angle):
        while angle >= 360: angle -= 360
        while angle < 0: angle += 360
        return angle

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def pack_and_send(self, payload):
        dlen = len(payload)
        tx_buf = (
            bytearray([0x00] * 46)
            + bytearray([0x00, 0xFF, 0x00, 0xFF, 0x49, 0xFF, dlen])
            + bytearray(payload)
        )
        checksum = sum(tx_buf[51:51 + dlen + 2]) & 0xFF
        tx_buf.append(checksum)
        tx_buf.append(0x4D)
        self.ser.write(tx_buf)

    def configure(self, report_tag=0x004C, report_rate=20):
        params = bytearray(11)
        params[0] = 0x12
        params[1] = 5
        params[2] = 255
        params[3] = 0
        params[4] = 0
        params[5] = report_rate
        params[6] = 1
        params[7] = 3
        params[8] = 5
        params[9] = report_tag & 0xFF
        params[10] = (report_tag >> 8) & 0xFF
        self.pack_and_send(params)
        time.sleep(0.2)
        self.pack_and_send([0x03])       # 唤醒
        time.sleep(0.2)
        self.pack_and_send([0x19])       # 开启主动上报
        time.sleep(0.2)

    def _require_len(self, buf, L, need):
        return len(buf) >= L + need

    def apply_mag_calibration(self, mx, my, mz):
        mx_cal = (mx - self.mag_calibration["offset_x"]) * self.mag_calibration["scale_x"]
        my_cal = (my - self.mag_calibration["offset_y"]) * self.mag_calibration["scale_y"]
        mz_cal = (mz - self.mag_calibration["offset_z"]) * self.mag_calibration["scale_z"]
        return mx_cal, my_cal, mz_cal

    def calc_mag_heading_tilt_compensated(self, mx, my, mz, roll_deg, pitch_deg):
        heading_deg = tilt_compensated_magnetic_heading_deg(
            (mx, my, mz), roll_deg, pitch_deg
        )
        return self._wrap_angle_180(heading_deg), self._wrap_angle_360(heading_deg)

    def unpack_euler_mag_gyro(self, buf):
        scale_angle = 0.0054931640625
        scale_mag = 0.15106201171875
        scale_angle_speed = 0.06103515625

        if len(buf) < 7 or buf[0] != 0x11:
            return None

        ctl = (buf[2] << 8) | buf[1]
        ms = (buf[6] << 24) | (buf[5] << 16) | (buf[4] << 8) | buf[3]
        L = 7

        result = {"ctl": ctl, "timestamp_ms": ms}
        mag = None
        euler = None
        gyro = None

        # acc_no_g  (含重力加速度)
        if ctl & 0x0001:
            if not self._require_len(buf, L, 6): return None
            L += 6
        # acc_with_g  (不含重力, 只有运动加速度)
        if ctl & 0x0002:
            if not self._require_len(buf, L, 6): return None
            scale_acc = 0.0001220703125  # 加速度量纲: g
            result["acc_mps2"] = {
                "x": self._to_int16(buf[L], buf[L + 1]) * scale_acc * 9.80665,
                "y": self._to_int16(buf[L + 2], buf[L + 3]) * scale_acc * 9.80665,
                "z": self._to_int16(buf[L + 4], buf[L + 5]) * scale_acc * 9.80665,
            }
            L += 6
        # gyro
        if ctl & 0x0004:
            if not self._require_len(buf, L, 6): return None
            gyro = {
                "x": self._to_int16(buf[L], buf[L + 1]) * scale_angle_speed,
                "y": self._to_int16(buf[L + 2], buf[L + 3]) * scale_angle_speed,
                "z": self._to_int16(buf[L + 4], buf[L + 5]) * scale_angle_speed,
            }
            result["gyro_dps"] = gyro
            L += 6
        # mag
        if ctl & 0x0008:
            if not self._require_len(buf, L, 6): return None
            mag = {
                "x": self._to_int16(buf[L], buf[L + 1]) * scale_mag,
                "y": self._to_int16(buf[L + 2], buf[L + 3]) * scale_mag,
                "z": self._to_int16(buf[L + 4], buf[L + 5]) * scale_mag,
            }
            result["mag"] = mag
            L += 6
        # temp + pressure + height
        if ctl & 0x0010:
            if not self._require_len(buf, L, 8): return None
            L += 8
        # quat
        if ctl & 0x0020:
            if not self._require_len(buf, L, 8): return None
            L += 8
        # euler
        if ctl & 0x0040:
            if not self._require_len(buf, L, 6): return None
            euler = {
                "roll": self._to_int16(buf[L], buf[L + 1]) * scale_angle,
                "pitch": self._to_int16(buf[L + 2], buf[L + 3]) * scale_angle,
                "yaw": self._to_int16(buf[L + 4], buf[L + 5]) * scale_angle,
            }
            result["euler_deg"] = euler
            L += 6

        if mag is not None:
            mx_cal, my_cal, mz_cal = self.apply_mag_calibration(mag["x"], mag["y"], mag["z"])
            result["mag_cal"] = {"x": mx_cal, "y": my_cal, "z": mz_cal}

        if mag is not None and euler is not None:
            heading_180, heading_360 = self.calc_mag_heading_tilt_compensated(
                result["mag_cal"]["x"], result["mag_cal"]["y"], result["mag_cal"]["z"],
                euler["roll"], euler["pitch"],
            )
            result["heading_mag_deg"] = heading_180
            result["heading_mag_360"] = heading_360

        return result

    def parse_byte(self, byte):
        self.cs += byte
        if self.rx_index == 0:
            if byte == self.CMD_PACKET_BEGIN:
                self.i = 0
                self.buf[self.i] = byte
                self.i += 1
                self.cs = 0
                self.rx_index = 1
        elif self.rx_index == 1:
            self.buf[self.i] = byte
            self.i += 1
            if byte == 0xFF:
                self.rx_index = 0
            else:
                self.rx_index = 2
        elif self.rx_index == 2:
            self.buf[self.i] = byte
            self.i += 1
            if byte > self.CMD_PACKET_MAX_DATA_SIZE_RX or byte == 0:
                self.rx_index = 0
            else:
                self.cmd_len = byte
                self.rx_index = 3
        elif self.rx_index == 3:
            self.buf[self.i] = byte
            self.i += 1
            if self.i >= self.cmd_len + 3:
                self.rx_index = 4
        elif self.rx_index == 4:
            self.cs -= byte
            if (self.cs & 0xFF) == byte:
                self.buf[self.i] = byte
                self.i += 1
                self.rx_index = 5
            else:
                self.rx_index = 0
        elif self.rx_index == 5:
            self.rx_index = 0
            if byte == self.CMD_PACKET_END:
                self.buf[self.i] = byte
                self.i += 1
                data_body = self.buf[3:self.i - 2]
                return self.unpack_euler_mag_gyro(data_body)
        return None

    def read_pending(self):
        """非阻塞读取所有可用数据，更新 latest；返回本次是否收到新帧。"""
        updated = False
        while self.ser.in_waiting > 0:
            data = self.ser.read(1)
            if data:
                parsed = self.parse_byte(data[0])
                if parsed:
                    self.latest = parsed
                    updated = True
        return updated

    def calibrate_magnetometer(self, duration_sec=20):
        print("=" * 60)
        print("开始磁力计校准")
        print("请手持 IMU 缓慢旋转，尽量覆盖各个方向")
        print("=" * 60)
        mags = []
        start_time = time.time()
        last_print_time = 0
        while time.time() - start_time < duration_sec:
            data = self.ser.read(1)
            if data:
                parsed = self.parse_byte(data[0])
                if parsed and "mag" in parsed:
                    m = parsed["mag"]
                    mags.append((m["x"], m["y"], m["z"]))
                    now = time.time()
                    if now - last_print_time > 1.0:
                        last_print_time = now
                        print(f"采集中... {int(now - start_time)}/{duration_sec}s")
        if len(mags) < 50:
            print("采样点太少，校准失败")
            return False
        xs = [p[0] for p in mags]
        ys = [p[1] for p in mags]
        zs = [p[2] for p in mags]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        offset_x = (max_x + min_x) / 2.0
        offset_y = (max_y + min_y) / 2.0
        offset_z = (max_z + min_z) / 2.0
        radius_x = (max_x - min_x) / 2.0
        radius_y = (max_y - min_y) / 2.0
        radius_z = (max_z - min_z) / 2.0
        if radius_x <= 1e-6 or radius_y <= 1e-6 or radius_z <= 1e-6:
            print("旋转覆盖不足，校准失败")
            return False
        avg_radius = (radius_x + radius_y + radius_z) / 3.0
        self.mag_calibration = {
            "offset_x": offset_x, "offset_y": offset_y, "offset_z": offset_z,
            "scale_x": avg_radius / radius_x,
            "scale_y": avg_radius / radius_y,
            "scale_z": avg_radius / radius_z,
        }
        print("\n磁力计校准完成")
        print(json.dumps(self.mag_calibration, indent=2, ensure_ascii=False))
        return True

    def save_mag_calibration(self, filepath="mag_calibration.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.mag_calibration, f, indent=2, ensure_ascii=False)
        print(f"校准参数已保存: {filepath}")

    def load_mag_calibration(self, filepath="mag_calibration.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                self.mag_calibration = json.load(f)
            print(f"已加载校准参数: {filepath}")
        except FileNotFoundError:
            print("未找到校准文件，使用默认值")


# ============================================================
#  深度计 Reader
# ============================================================
class DepthSensor:
    """Read and validate the vendor ASCII-CRC pressure/temperature frame."""

    MPA_TO_WATER_DEPTH_M = 101.972

    def __init__(
        self,
        port="/dev/ttyUSB0",
        baudrate=19200,
        address=1,
        timeout=0.3,
        depth_raw_zero_mpa=0.0,
        depth_m_per_mpa=MPA_TO_WATER_DEPTH_M,
    ):
        if not math.isfinite(depth_raw_zero_mpa):
            raise ValueError("depth_raw_zero_mpa must be finite")
        if not math.isfinite(depth_m_per_mpa) or depth_m_per_mpa <= 0:
            raise ValueError("depth_m_per_mpa must be a positive finite number")
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.address = address
        self.depth_raw_zero_mpa = float(depth_raw_zero_mpa)
        self.depth_m_per_mpa = float(depth_m_per_mpa)
        self.pressure_mpa = 0.0
        self.temperature_c = 0.0
        self._lock = threading.Lock()
        self._serial_lock = threading.RLock()
        self.ser = None
        # Responses can be fragmented, echoed, or concatenated by USB/serial
        # bridges. Keep a byte stream and resynchronize on the vendor prefix.
        self._rx_buffer = bytearray()
        self.depth_frames_received = 0
        self.depth_invalid_frames = 0
        self.depth_last_error = "awaiting_response"
        self._serial_error = ""
        self._open_serial()

    def close(self):
        with self._serial_lock:
            ser = self.ser
            self.ser = None
            if ser is not None:
                try:
                    if getattr(ser, "is_open", True):
                        ser.close()
                except Exception:
                    pass

    def _open_serial(self):
        """Open the depth port if available; startup must not depend on it."""
        with self._serial_lock:
            if self.ser is not None and getattr(self.ser, "is_open", True):
                return True
            try:
                self.ser = serial.Serial(
                    self.port, self.baudrate, timeout=self.timeout
                )
            except Exception as exc:
                self.ser = None
                self._serial_error = f"{type(exc).__name__}: {exc}"
                self.depth_last_error = f"serial_open_failed:{type(exc).__name__}"
                return False
            self._serial_error = ""
            return True

    def _mark_serial_unavailable(self, exc):
        """Drop a broken handle so the next query can reopen the device."""
        self.depth_last_error = f"serial_io_error:{type(exc).__name__}"
        self._serial_error = f"{type(exc).__name__}: {exc}"
        ser = self.ser
        self.ser = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def query(self):
        """发送ASCII查询帧，返回 (压力MPa, 温度°C) 或 None"""
        with self._serial_lock:
            if not self._open_serial():
                return None
            addr_str = f"{self.address:02d}"
            frame = f"sD{addr_str}\r\n"
            try:
                self.ser.write(frame.encode("ascii"))
                self.ser.flush()

                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    # Some pyserial/platform combinations raise TypeError here
                    # while a USB-serial handle is being removed. Treat it as a
                    # transient sensor outage instead of killing the worker.
                    available = int(self.ser.in_waiting)
                    if available:
                        self._rx_buffer.extend(self.ser.read(min(available, 128)))
                        parsed = self._extract_frame()
                        if parsed is not None:
                            self.depth_frames_received += 1
                            self.depth_last_error = ""
                            return parsed
                    else:
                        time.sleep(0.005)
            except Exception as exc:
                self._mark_serial_unavailable(exc)
                return None

            self.depth_last_error = "timeout_or_invalid_frame"
            return None

    def _extract_frame(self):
        """Extract one valid 22-byte frame, discarding noise and bad frames."""
        while True:
            start = self._rx_buffer.find(b"sD")
            if start < 0:
                # Keep a possible split prefix (the final ``s``) for the next
                # serial read while bounding memory if arbitrary noise arrives.
                keep = 1 if self._rx_buffer[-1:] == b"s" else 0
                if len(self._rx_buffer) > keep:
                    if keep:
                        del self._rx_buffer[:-1]
                    else:
                        self._rx_buffer.clear()
                return None
            if start:
                del self._rx_buffer[:start]
            if len(self._rx_buffer) < 22:
                return None
            candidate = bytes(self._rx_buffer[:22])
            parsed = self._parse(candidate)
            if parsed is not None:
                del self._rx_buffer[:22]
                return parsed
            self.depth_invalid_frames += 1
            self.depth_last_error = "invalid_frame"
            # Drop one byte only; a second valid frame may already be buffered.
            del self._rx_buffer[0]

    def _parse(self, data):
        # Vendor format: 22 ASCII bytes: sD AA S PPPPPPPP S TTTTT CCC.
        # CCC is the decimal digit sum of bytes 2..18.
        if len(data) != 22 or data[:2] != b"sD":
            return None
        if not data[2:4].isdigit() or int(data[2:4]) != self.address:
            return None
        try:
            if data[4] not in (ord('0'), ord('1')) or data[13] not in (ord('0'), ord('1')):
                return None
            if not data[5:13].isdigit() or not data[14:19].isdigit() or not data[19:22].isdigit():
                return None
            if int(data[19:22]) != sum(byte - ord('0') for byte in data[2:19]):
                return None

            press_sign = -1.0 if data[4] == ord('1') else 1.0
            press_raw = int(data[5:13])
            with self._lock:
                self.pressure_mpa = press_sign * press_raw / 10000.0

            temp_sign = -1.0 if data[13] == ord('1') else 1.0
            temp_raw = int(data[14:19])
            with self._lock:
                self.temperature_c = temp_sign * temp_raw / 1000.0
        except (ValueError, UnicodeDecodeError):
            return None

        return (self.pressure_mpa, self.temperature_c)

    def get_depth(self):
        with self._lock:
            # Keep the legacy scale, while allowing a measured surface reference
            # for sensors that report absolute rather than gauge pressure.
            return (self.pressure_mpa - self.depth_raw_zero_mpa) * self.depth_m_per_mpa

    def set_surface_reference(self, pressure_mpa):
        """Use a verified current pressure sample as this run's depth zero."""
        pressure_mpa = float(pressure_mpa)
        if not math.isfinite(pressure_mpa):
            raise ValueError("surface reference pressure must be finite")
        with self._lock:
            self.depth_raw_zero_mpa = pressure_mpa

    def get_pressure(self):
        with self._lock:
            return self.pressure_mpa

    def get_temperature(self):
        with self._lock:
            return self.temperature_c


# ============================================================
#  联合读取主程序
# ============================================================
def main():
    # --- 初始化 IMU ---
    print("[1/3] 初始化 IMU ...")
    imu = IMUReader(port="/dev/serial0", baudrate=115200, timeout=0.2)
    imu.load_mag_calibration("mag_calibration.json")
    imu.configure(report_tag=0x004C, report_rate=20)  # gyro + mag + euler
    print("      IMU 配置完成")

    # --- 初始化深度计 ---
    print("[2/3] 初始化深度计 ...")
    depth = DepthSensor(port="/dev/ttyUSB0", baudrate=19200, address=1, timeout=0.3)
    print("      深度计初始化完成")

    # --- 清空缓冲区 ---
    time.sleep(0.5)
    imu.ser.reset_input_buffer()

    print("[3/3] 开始联合读取...")
    print("-" * 85)
    print(f"{'roll':>7} {'pitch':>7} {'yaw':>7} {'h360':>7} "
          f"{'gx':>7} {'gy':>7} {'gz':>7} {'水深m':>8} {'温C':>6}")
    print("-" * 70)

    last_depth_query = 0
    DEPTH_INTERVAL = 0.5  # 深度计每0.5秒查询一次

    try:
        while True:
            # IMU: 非阻塞读取所有待处理数据
            imu.read_pending()

            # 深度计: 定时查询
            now = time.time()
            if now - last_depth_query >= DEPTH_INTERVAL:
                depth.query()
                last_depth_query = now

            # 输出最新数据
            if imu.latest and "euler_deg" in imu.latest:
                e = imu.latest["euler_deg"]
                g = imu.latest.get("gyro_dps", {"x": 0.0, "y": 0.0, "z": 0.0})
                h = imu.latest.get("heading_mag_360", 0.0)
                d = depth.get_depth()
                t = depth.get_temperature()

                print(f"{e['roll']:7.1f} {e['pitch']:7.1f} {e['yaw']:7.1f} "
                      f"{h:7.1f} {g['x']:7.1f} {g['y']:7.1f} {g['z']:7.1f} "
                      f"{d:8.3f} {t:6.2f}")

            time.sleep(0.02)  # 50Hz循环，保证IMU数据不丢

    except KeyboardInterrupt:
        print("\n程序停止")
    finally:
        imu.close()
        depth.close()


if __name__ == "__main__":
    main()
