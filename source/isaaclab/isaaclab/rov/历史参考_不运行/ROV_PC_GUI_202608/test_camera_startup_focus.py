from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("start_usb_camera_stream.sh")


class CameraStartupFocusTests(unittest.TestCase):
    def test_manual_focus_is_applied_before_camera_server_starts(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('focus="${4:-256}"', text)
        auto_index = text.index('-c focus_auto=0')
        focus_index = text.index('-c focus_absolute="$focus"')
        server_index = text.index('exec python3 "$script_dir/usb_mjpeg_server.py"')
        self.assertLess(auto_index, focus_index)
        self.assertLess(focus_index, server_index)

    def test_focus_control_failure_does_not_prevent_video_startup(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('if command -v v4l2-ctl', text)
        self.assertIn('warning: unable to disable camera autofocus', text)
        self.assertIn('warning: unable to set manual camera focus', text)


if __name__ == "__main__":
    unittest.main()
