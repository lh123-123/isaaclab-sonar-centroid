"""Offline, reversible QGC v5.1.4 instrument installation. QGC must be closed."""
import argparse
import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess

BASE = Path(__file__).resolve().parent
DEFAULT_QML = 'qrc:/qml/QGroundControl/FlightMap/Widgets/IntegratedCompassAttitude.qml'


def replace_section(text, name, content):
    """Replace only an explicit section; preserve every other Qt setting byte."""
    pattern = re.compile(r'^\[' + re.escape(name) + r'\]\r?\n.*?(?=^\[|\Z)', re.M | re.S)
    section = '[' + name + ']\n' + content.rstrip('\n') + '\n\n'
    if pattern.search(text):
        return pattern.sub(lambda _match: section, text, count=1)
    return text.rstrip('\n') + '\n\n' + section


def set_instrument(text, url):
    pattern = re.compile(r'^\[FlyView\]\r?\n.*?(?=^\[|\Z)', re.M | re.S)
    match = pattern.search(text)
    if not match:
        return replace_section(text, 'FlyView', 'instrumentQmlFile2=' + url)
    old = match.group(0)
    key = re.compile(r'^instrumentQmlFile2=.*$', re.M)
    new = key.sub(lambda _m: 'instrumentQmlFile2=' + url, old, count=1) if key.search(old) else old.rstrip('\n') + '\ninstrumentQmlFile2=' + url + '\n\n'
    return text[:match.start()] + new + text[match.end():]


def competition_settings(text, qml):
    text = set_instrument(text, Path(qml).resolve().as_uri())
    # Leave the independent native QGC flight timer, remove unreliable GPS
    # ground speed / home distance and duplicate attitude from the Sub bar only.
    return replace_section(text, 'TelemetryBarUserSettings-12', '''version=1
fontSize=0
rowCount=1
columns\\size=1
columns\\1\\rows\\size=1
columns\\1\\rows\\1\\text=解锁计时
columns\\1\\rows\\1\\showUnits=false
columns\\1\\rows\\1\\icon=
columns\\1\\rows\\1\\rangeType=0
columns\\1\\rows\\1\\factGroupName=Vehicle
columns\\1\\rows\\1\\factName=FlightTime''')


def qgc_running():
    if os.name != 'nt':
        return False
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq QGroundControl.exe', '/FO', 'CSV', '/NH'],
                            capture_output=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError('无法检查QGC进程，请确认QGC已退出')
    return b'"qgroundcontrol.exe"' in result.stdout.lower()


def install(path, *, restore=False):
    if qgc_running():
        raise RuntimeError('请先正常退出QGC（不会自动关闭或中断控制）。')
    path = Path(path)
    if not path.is_file():
        raise ValueError('找不到QGC设置文件：' + str(path))
    backup = path.with_name('QGroundControl.before_rov_fly.ini')
    if restore:
        if not backup.is_file():
            raise ValueError('没有安装前备份，未修改任何设置')
        # Restore these two UI sections only; do not roll back later link/video
        # changes or erase any other settings made after installation.
        prior = backup.read_text(encoding='utf-8-sig')
        original = path.read_text(encoding='utf-8-sig')
        for section in ('FlyView', 'TelemetryBarUserSettings-12'):
            found = re.search(r'^\[' + re.escape(section) + r'\]\r?\n(.*?)(?=^\[|\Z)', prior, re.M | re.S)
            if section == 'FlyView':
                key = re.search(r'^instrumentQmlFile2=(.*)$', found[1], re.M) if found else None
                original = set_instrument(original, key[1].strip() if key else DEFAULT_QML)
            elif found:
                original = replace_section(original, section, found[1])
            else:
                original = re.sub(r'^\[' + re.escape(section) + r'\]\r?\n.*?(?=^\[|\Z)', '', original, flags=re.M | re.S)
        result = original
    else:
        qml = BASE / 'qgc' / 'ROVCompetitionInstrument.qml'
        if not qml.is_file() or not qml.with_name('ROVCompetitionCard.qml').is_file():
            raise ValueError('比赛仪表QML文件不完整，未修改设置')
        if not backup.exists():
            shutil.copy2(path, backup)
        result = competition_settings(path.read_text(encoding='utf-8-sig'), qml)
    # Application-generated settings transaction, original remains recoverable.
    dated = path.with_name('QGroundControl.rov_fly_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.ini')
    shutil.copy2(path, dated)
    temporary = path.with_suffix('.rov.tmp')
    temporary.write_text(result, encoding='utf-8')
    temporary.replace(path)
    return backup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--settings', type=Path, default=Path(os.environ.get('APPDATA', '')) / 'QGroundControl' / 'QGroundControl.ini')
    args = parser.parse_args()
    try:
        backup = install(args.settings, restore=args.restore)
        print('已恢复原仪表。' if args.restore else '已启用Fly比赛仪表。重新运行比赛启动脚本，保持桥接运行。')
        print('安装前备份：', backup)
    except (OSError, ValueError, RuntimeError) as exc:
        print('未安装：', exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
