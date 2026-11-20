from types import SimpleNamespace
from pathlib import Path
import unittest

from usb_mjpeg_server import build_capture_command


SCRIPT = Path(__file__).with_name("start_usb_camera_stream.sh")


class CameraStreamQualityTests(unittest.TestCase):
    def test_capture_uses_fluid_720p_display_profile(self):
        args = SimpleNamespace(
            device="/dev/video7",
            video_size="1280x720",
            framerate=30,
            output_framerate=30,
            jpeg_quality=3,
        )

        command = build_capture_command(args)

        self.assertIn("1280x720", command)
        self.assertNotIn("fps=15", command)
        self.assertEqual(command[command.index("-framerate") + 1], "30")
        self.assertEqual(command[command.index("-q:v") + 1], "3")

    def test_startup_uses_fluid_720p_display_profile(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--video-size 1280x720", text)
        self.assertIn("--framerate 30", text)
        self.assertIn("--output-framerate 30", text)
        self.assertIn("--jpeg-quality 3", text)


if __name__ == "__main__":
    unittest.main()
