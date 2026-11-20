"""Prepare the FC's sensor-to-body rotation. Never writes to the vehicle."""
import json
import math
from pathlib import Path
import tkinter as tk

PARAMETERS = Path.home() / 'Documents/QGroundControl/Parameters'
PROFILE = Path(__file__).with_name('mount_profile.json')


def rotation_matrix(roll, pitch, yaw):
    """R_body_sensor = Rz(yaw) Ry(pitch) Rx(roll), matching AP_CustomRotations."""
    values = (roll, pitch, yaw)
    if not all(type(v) in (int, float) and math.isfinite(v) and -180 <= v <= 180 for v in values):
        raise ValueError('安装角必须是−180～180°的有限数值')
    r, p, y = map(math.radians, values)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
            (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr), (-sp, cp*sr, cp*cr))


def parameter_files(angles):
    rotation_matrix(*angles)
    first = '# Enable official custom rotation parameters. Reboot and refresh before step 2.\n1\t1\tCUST_ROT_ENABLE\t1\t2\n'
    second = '# Absolute sensor-to-body rotation in degrees, Euler ZYX. NOT a startup zero.\n'
    for name, value in zip(('ROLL', 'PITCH', 'YAW'), angles):
        second += f'1\t1\tCUST_ROT1_{name}\t{value:.8g}\t9\n'
    second += '1\t1\tAHRS_ORIENTATION\t101\t2\n'
    return first, second


class MountSettings:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title('飞控安装角校正 · 导出参数')
        self.window.geometry('650x510')
        self.window.configure(bg='#1f2937')
        box = tk.Frame(self.window, bg='#1f2937', padx=20, pady=18)
        box.pack(fill='both', expand=True)
        text = ('飞控坐标 → 车身坐标的固定旋转（绝对配置，非叠加偏移）\n'
                '车身坐标：X向前、Y向右、Z向下；右手正方向，单位度。\n'
                '从飞控/机体轴的安装关系测量，不能直接填QGC当前角度。\n'
                '水平摆放只能校验横滚/俯仰；偏航安装角须对齐车头测量。')
        tk.Label(box, text=text, justify='left', bg='#1f2937', fg='#edf5ff',
                 font=('Microsoft YaHei UI', 10)).pack(anchor='w')
        try:
            saved = json.loads(PROFILE.read_text(encoding='utf-8'))['angles_deg']
            rotation_matrix(*saved)
        except (OSError, ValueError, KeyError, TypeError):
            saved = (0, 0, 0)
        self.entries = []
        for label, value in zip(('横滚 Roll', '俯仰 Pitch', '偏航 Yaw'), saved):
            row = tk.Frame(box, bg='#1f2937')
            row.pack(fill='x', pady=6)
            tk.Label(row, text=label, width=15, anchor='w', bg='#1f2937', fg='white').pack(side='left')
            var = tk.StringVar(value=str(value))
            tk.Entry(row, textvariable=var, width=16).pack(side='left')
            self.entries.append(var)
        self.status = tk.StringVar(value='这里保存的是待导入设置，当前飞控是否已应用请在QGC读回确认。')
        tk.Label(box, textvariable=self.status, justify='left', wraplength=590,
                 bg='#1f2937', fg='#38bdf8').pack(fill='x', pady=12)
        tk.Button(box, text='保存安装角并导出两步参数文件', command=self.export).pack(fill='x', pady=6)
        tk.Label(box, text='第1步：导入启用文件 → 飞控重启、刷新参数。\n'
                 '第2步：确认CUST_ROT1_*出现，再导入安装角文件 → 重启。\n'
                 '然后在机器人实际水平时做QGC水平校准，检查抬头/横滚/转向符号。\n'
                 '飞控应用后，QGC和闭环统一按车身姿态工作，无须再在PC叠加校正。',
                 justify='left', bg='#1f2937', fg='#9ca3af', wraplength=590).pack(anchor='w', pady=8)

    def export(self):
        try:
            values = tuple(float(var.get()) for var in self.entries)
            first, second = parameter_files(values)
            PARAMETERS.mkdir(parents=True, exist_ok=True)
            (PARAMETERS / 'ROV安装校正_步骤1_启用旋转.params').write_text(first, encoding='utf-8')
            (PARAMETERS / 'ROV安装校正_步骤2_安装角.params').write_text(second, encoding='utf-8')
            PROFILE.write_text(json.dumps({'angles_deg': values, 'status': 'PREPARED_NOT_APPLIED'}, indent=2), encoding='utf-8')
        except (OSError, ValueError) as exc:
            self.status.set('未导出：' + str(exc))
            return
        self.status.set('已保存并导出到QGC Parameters文件夹。尚未写入飞控；按下面两步导入。')
