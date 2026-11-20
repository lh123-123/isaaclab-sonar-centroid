import json
import tkinter as tk
from unittest import TestCase
from unittest.mock import patch, Mock
from camera_focus import CameraFocusClient, CameraFocusPanel


class CameraFocusTests(TestCase):
    def response(self, **overrides):
        report = dict(configured_profile='water', water_focus_value=256, autofocus=False,
                      minimum=0, maximum=1023, step=1, focus_value=256,
                      focus_readable=True, saved=True)
        report.update(overrides)
        return Mock(returncode=0, stdout=json.dumps(report), stderr='')

    def test_query_has_no_mutation_or_motor_commands(self):
        client = CameraFocusClient('192.168.1.3')
        with patch('camera_focus.shutil.which', return_value='ssh'), patch('camera_focus.subprocess.run', return_value=self.response()) as run:
            actual = client.request()
        command = run.call_args.args[0]
        self.assertIn('--focus-status', command[-1])
        self.assertNotIn('--set-focus', command[-1])
        self.assertNotIn('sudo', command[-1])
        self.assertFalse(actual['autofocus'])

    def test_apply_is_validated_and_requires_readback(self):
        client = CameraFocusClient('192.168.1.3')
        with patch('camera_focus.shutil.which', return_value='ssh'), patch('camera_focus.subprocess.run', return_value=self.response()) as run:
            client.request('water', 256)
            self.assertIn('--set-focus water --save-focus --water-focus 256', run.call_args.args[0][-1])
            run.return_value = self.response(configured_profile='auto', autofocus=True)
            client.request('auto')
            self.assertIn('--set-focus auto --save-focus', run.call_args.args[0][-1])
            self.assertNotIn('--water-focus', run.call_args.args[0][-1])
            with self.assertRaises(ValueError):
                client.request('water;shutdown', 256)
            with self.assertRaises(ValueError):
                client.request('water', '256;shutdown')
            run.return_value = self.response(saved=False)
            with self.assertRaisesRegex(RuntimeError, '未确认保存'):
                client.request('water', 256)
            run.return_value = self.response(autofocus=False)
            with self.assertRaisesRegex(RuntimeError, '未确认保存'):
                client.request('auto')
            run.return_value = self.response(focus_value=68)
            with self.assertRaisesRegex(RuntimeError, '未确认保存'):
                client.request('water', 256)

    def test_connection_failure_is_not_claimed_success(self):
        with patch('camera_focus.shutil.which', return_value='ssh'), patch('camera_focus.subprocess.run', return_value=Mock(returncode=255, stderr='Connection closed', stdout='')):
            with self.assertRaisesRegex(RuntimeError, 'Connection closed'):
                CameraFocusClient('192.168.1.3').request()

    def test_inactive_auto_position_cannot_be_saved_or_reported_as_live(self):
        client = CameraFocusClient('192.168.1.3')
        unavailable = self.response(configured_profile='auto', autofocus=True,
                                    focus_value=None, focus_readable=False)
        with patch('camera_focus.shutil.which', return_value='ssh'), \
                patch('camera_focus.subprocess.run', return_value=unavailable) as run:
            report = client.request()
            self.assertIsNone(report['focus_value'])
            self.assertEqual(run.call_count, 1)
            self.assertTrue(all('--focus-status' in call.args[0][-1] for call in run.call_args_list))


class CameraFocusPanelTests(TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.client_patch = patch('camera_focus.CameraFocusClient')
        self.client = self.client_patch.start().return_value
        self.thread_patch = patch('camera_focus.threading.Thread')
        self.thread = self.thread_patch.start()
        self.client.request.return_value = self.report(False, 200)
        self.panel = CameraFocusPanel(self.root, '192.168.1.3', Mock())
        self.finish_request()

    def tearDown(self):
        self.panel.close()
        self.root.destroy()
        self.thread_patch.stop()
        self.client_patch.stop()

    def report(self, auto, focus, preset=200):
        return dict(configured_profile='auto' if auto else 'water',
                    water_focus_value=preset, autofocus=auto, minimum=0,
                    maximum=1023, step=1, focus_value=focus,
                    focus_readable=focus is not None, saved=True)

    def finish_request(self):
        self.thread.call_args.kwargs['target']()
        self.panel.poll()

    def test_auto_disables_editor_without_showing_cached_value_as_live(self):
        self.panel.mode.set('auto')
        self.panel.begin(True)
        self.assertEqual(str(self.panel.input.cget('state')), 'disabled')
        self.client.request.return_value = self.report(True, None)
        self.finish_request()
        self.assertEqual(self.panel.input.get(), '200')
        self.assertEqual(str(self.panel.input.cget('state')), 'disabled')
        self.assertEqual(str(self.panel.apply.cget('state')), 'disabled')
        self.assertIn('不可读取', self.panel.readback.get())

    def test_manual_preset_remains_available_after_auto(self):
        self.panel.mode.set('auto')
        self.panel.begin(True)
        self.client.request.return_value = self.report(True, None)
        self.finish_request()
        self.assertEqual(self.panel.value.get(), '200')
        self.panel.mode.set('water')
        self.panel.begin(True)
        self.client.request.return_value = self.report(False, 200)
        self.finish_request()
        self.assertEqual(self.panel.input.get(), '200')
        self.assertEqual(str(self.panel.input.cget('state')), 'normal')

    def test_auto_does_not_depend_on_manual_editor_contents(self):
        self.panel.value.set('invalid unfinished draft')
        self.panel.mode.set('auto')
        self.panel.begin(True)
        self.client.request.return_value = self.report(True, 341)
        self.finish_request()
        self.assertEqual(self.client.request.call_args.args[0], 'auto')
        self.assertEqual(self.panel.input.get(), '200')

    def test_unreadable_autofocus_position_is_not_displayed_or_saved(self):
        self.panel.mode.set('auto')
        self.panel.begin(True)
        self.client.request.return_value = self.report(True, None)
        self.finish_request()
        self.assertEqual(self.panel.input.get(), '200')
        self.assertIn('不可读取', self.panel.readback.get())

    def test_auto_does_not_poll_unavailable_position(self):
        self.panel.mode.set('auto')
        self.panel.begin(True)
        self.client.request.return_value = self.report(True, None)
        self.finish_request()
        self.assertEqual(self.client.request.call_count, 2)
