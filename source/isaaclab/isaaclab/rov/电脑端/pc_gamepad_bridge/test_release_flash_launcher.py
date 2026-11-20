import hashlib
import contextlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location(
    'release_flash_launcher', Path(__file__).parent / 'firmware/launch_release_flash.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class ReleaseFlashLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name)
        self.firmware = self.bundle / 'source/build/Pixhawk1/bin/ardusub.apj'
        self.firmware.parent.mkdir(parents=True)
        self.firmware.write_bytes(b'fake artifact, never uploaded')
        self.built = {'status': 'BUILT_NOT_FLASHED', 'source_commit': launcher.COMMIT,
                      'patch_sha256': launcher.PATCH, 'board_id': 9,
                      'apj': str(self.firmware),
                      'apj_sha256': hashlib.sha256(self.firmware.read_bytes()).hexdigest()}
        (self.bundle / 'build_status.json').write_text(json.dumps(self.built))

    def read_text(self, path, *args, **kwargs):
        if path.as_posix() == '/sys/class/thermal/thermal_zone0/temp':
            return '51000'
        if path.as_posix() == '/proc/sys/kernel/random/boot_id':
            return 'mock-boot'
        return self.real_read(path, *args, **kwargs)

    def check_context(self, state='inactive\n'):
        self.real_read = Path.read_text
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(launcher, 'BUNDLE', self.bundle))
        stack.enter_context(patch.object(Path, 'read_text',
            lambda path, *args, **kwargs: self.read_text(path, *args, **kwargs)))
        stack.enter_context(patch.object(launcher.subprocess, 'check_output', return_value=state))
        return stack

    def test_completed_exact_artifact_allowed(self):
        with self.check_context():
            built, firmware, temp = launcher.check_bundle(self.bundle)
            self.assertEqual((built, firmware, temp), (self.built, self.firmware, 51.0))

    def test_changed_artifact_rejected(self):
        self.firmware.write_bytes(b'changed')
        with self.check_context(), self.assertRaisesRegex(RuntimeError, 'artifact changed'):
            launcher.check_bundle(self.bundle)

    def test_active_service_rejected(self):
        with self.check_context('active\n'), self.assertRaisesRegex(RuntimeError, 'fully stopped'):
            launcher.check_bundle(self.bundle)

    def test_wrong_board_rejected(self):
        self.built['board_id'] = 255
        (self.bundle / 'build_status.json').write_text(json.dumps(self.built))
        with self.check_context(), self.assertRaisesRegex(RuntimeError, 'release build missing'):
            launcher.check_bundle(self.bundle)

    def test_repeated_launch_never_starts_child(self):
        (self.bundle / 'flash_launch.json').write_text('{}')
        with self.check_context(), patch.object(launcher.subprocess, 'Popen') as child, \
                patch('sys.argv', ['launcher', '--bundle', str(self.bundle)]), \
                self.assertRaisesRegex(RuntimeError, 'already launched'):
            launcher.main()
        child.assert_not_called()

    def test_launch_detached_with_no_stdin(self):
        with self.check_context(), patch.object(launcher.subprocess, 'Popen',
                                               return_value=Mock(pid=77)) as child, \
                patch('sys.argv', ['launcher', '--bundle', str(self.bundle)]):
            launcher.main()
        self.assertTrue(child.call_args.kwargs['start_new_session'])
        self.assertEqual(child.call_args.kwargs['stdin'], subprocess.DEVNULL)
        record = json.loads((self.bundle / 'flash_launch.json').read_text())
        self.assertEqual(record['pid'], 77)

    def test_failed_uploader_never_recorded_as_success(self):
        with self.check_context(), patch.object(launcher.subprocess, 'run',
                                               return_value=Mock(returncode=1)), \
                patch('sys.argv', ['launcher', '--bundle', str(self.bundle), '--run']), \
                self.assertRaises(SystemExit) as exit_result:
            launcher.main()
        self.assertEqual(exit_result.exception.code, 1)
        self.assertEqual(json.loads((self.bundle / 'flash_status.json').read_text())['status'],
                         'FLASH_FAILED')


if __name__ == '__main__':
    unittest.main()
