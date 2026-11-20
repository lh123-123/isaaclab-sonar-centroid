import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent / 'firmware'
sys.path.insert(0, str(ROOT))
from prepare_independent_bundle import sources


class IndependentFirmwareTests(unittest.TestCase):
    def test_previous_prepared_and_flashed_evidence_is_not_overwritten(self):
        for name in ('prepared_control_20261004', 'prepared_independent_20261004'):
            bundle = ROOT / name
            manifest = json.loads((bundle / 'manifest.json').read_text())
            self.assertEqual(hashlib.sha256((bundle / 'ardusub_rov_control.patch').read_bytes()).hexdigest(),
                             manifest['patch_sha256'])
            for path, digest in manifest['source_sha256'].items():
                self.assertEqual(hashlib.sha256((bundle / 'source' / path).read_bytes()).hexdigest(), digest)

    def test_staged_patch_and_fast_loop_controller_ownership(self):
        old, new = sources(ROOT / 'baseline_dbe79216')
        bundle = ROOT / 'prepared_tolerance_20261005'
        manifest = json.loads((bundle / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'OFFLINE_NOT_BUILT_NOT_FLASHED')
        for path, content in new.items():
            self.assertEqual((bundle / 'source' / path).read_text(encoding='utf-8'), content)
            self.assertEqual(manifest['source_sha256'][path], hashlib.sha256(content.encode()).hexdigest())
        self.assertIn('control_mode != Mode::Number::DEPTH_ONLY', new['ArduSub/Sub.cpp'])
        self.assertIn('friend class ModeDepthOnly;', new['ArduSub/Sub.h'])
        self.assertIn('control_mode(Mode::Number::MANUAL)', new['ArduSub/Sub.cpp'])
        self.assertIn('sub.control_mode != Mode::Number::DEPTH_ONLY', new['ArduSub/mode_althold.cpp'])
        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory)
            shutil.copytree(ROOT / 'baseline_dbe79216', tree, dirs_exist_ok=True)
            subprocess.run(['git', 'apply', '--check', str(bundle / 'ardusub_rov_control.patch')], cwd=tree, check=True)
            subprocess.run(['git', 'apply', str(bundle / 'ardusub_rov_control.patch')], cwd=tree, check=True)
            for path, content in new.items():
                self.assertEqual((tree / path).read_text(encoding='utf-8'), content)

    @unittest.skipUnless(shutil.which('g++'), 'Host C++ compiler unavailable')
    def test_actual_depth_only_cpp_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / 'depth-only.exe'
            result = subprocess.run(['g++', '-std=c++11', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(ROOT / 'depth_only_test_stubs'), str(ROOT / 'test_depth_only.cpp'),
                    str(ROOT / 'mode_depth_only.cpp'), '-o', str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
