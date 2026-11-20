import unittest
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import rov_video_stream as video


def controls(auto=0, focus=256, name="focus_automatic_continuous", inactive=False, volatile=False):
    flags = " flags=" + ",".join(x for x, enabled in (("inactive", inactive), ("volatile", volatile)) if enabled) if inactive or volatile else ""
    return Mock(returncode=0, stdout=(
        f"focus_absolute 0x009a090a (int) : min=0 max=1023 step=1 default=68 value={focus}{flags}\n"
        f"{name} 0x009a090c (bool) : default=1 value={auto}\n"), stderr="")


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.config = video.load_profile()
        # Keep the existing 720p test fixture independent of the deployed profile.
        self.config.update(width=1280, height=720, fps=30, bitrate_kbps=2500,
                           enhancement_mode="off", focus_profile="water")

    def test_1080p_keeps_screen_ratio_and_low_latency_settings(self):
        self.config.update(width=1920, height=1080, fps=30, bitrate_kbps=8000)
        cmd = video.build_command(self.config)
        self.assertEqual(cmd[cmd.index('-video_size') + 1], '1920x1080')
        self.assertIn('scale=1920:1080,fps=30', cmd[cmd.index('-vf') + 1])
        self.assertEqual(cmd[cmd.index('-b:v') + 1], '8000k')
        self.assertEqual(cmd[cmd.index('-bufsize') + 1], '1600k')
        self.assertEqual(cmd[cmd.index('-bf') + 1], '0')

    def test_qgc_stream_is_rtp_h264_with_repeat_headers_and_no_b_frames(self):
        cmd = video.build_command(self.config)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "libx264")
        self.assertEqual(cmd[cmd.index("-bf") + 1], "0")
        self.assertEqual(cmd[cmd.index("-tune") + 1], "zerolatency")
        self.assertIn("repeat-headers=1", cmd[cmd.index("-x264-params") + 1])
        self.assertEqual(cmd[-1], "rtp://192.168.1.4:5600?pkt_size=1200")
        self.assertEqual(cmd[-5:-1], ["-f", "rtp", "-payload_type", "96"])

    def test_upside_down_camera_rotates_180_without_single_axis_mirroring(self):
        self.config["rotation_degrees"] = 180
        cmd = video.build_command(self.config)
        self.assertEqual(cmd[cmd.index("-vf") + 1], "hflip,vflip,scale=1280:720,fps=30")
        self.assertEqual(cmd[cmd.index("-video_size") + 1], "1280x720")
        self.assertEqual(cmd[cmd.index("-threads") + 1], "2")

    @unittest.skipUnless(shutil.which("ffmpeg"), "Requires real FFmpeg; run on Pi")
    def test_real_filter_rotates_pixels_180(self):
        self.config.update(rotation_degrees=180, width=16, height=16, fps=1)
        command = video.build_command(self.config)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
        pixels = [colors[(y // 8) * 2 + (x // 8)] for y in range(16) for x in range(16)]
        source = bytes(channel for pixel in pixels for channel in pixel)
        expected = bytes(channel for pixel in reversed(pixels) for channel in pixel)
        result = subprocess.run(
            [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-filter_threads", "1",
             "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "16x16",
             "-framerate", "1", "-i", "pipe:0", "-vf", command[command.index("-vf") + 1],
             "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
            input=source, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(result.stdout, expected)

    def test_old_profile_keeps_unrotated_image(self):
        self.config.pop("rotation_degrees", None)
        cmd = video.build_command(self.config)
        self.assertEqual(cmd[cmd.index("-vf") + 1], "scale=1280:720,fps=30")

    def test_mild_enhancement_is_explicit_and_reversible(self):
        self.config["enhancement_mode"] = "mild"
        cmd = video.build_command(self.config)
        self.assertIn("eq=contrast=1.04:saturation=1.04", cmd[cmd.index("-vf") + 1])
        self.config["enhancement_mode"] = "off"
        self.assertNotIn("eq=", video.build_command(self.config)[video.build_command(self.config).index("-vf") + 1])

    def test_invalid_enhancement_mode_rejected_before_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "video_profile.json"
            self.config["enhancement_mode"] = "ultra"
            profile.write_text(json.dumps(self.config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "enhancement_mode"):
                video.load_profile(profile)

    def test_invalid_rotation_rejected_before_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "video_profile.json"
            for angle in (90, -180, 360, 180.5, "180", True):
                self.config["rotation_degrees"] = angle
                profile.write_text(json.dumps(self.config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "rotation_degrees"):
                    video.load_profile(profile)

    def test_legacy_source_does_not_open_device_or_change_focus(self):
        self.config["source"] = "legacy-mjpeg"
        cmd = video.build_command(self.config)
        self.assertNotIn("v4l2", cmd)
        self.assertNotIn("/dev/video0", cmd)
        runner = Mock()
        video.apply_focus(self.config, runner=runner)
        runner.assert_not_called()

    def test_focus_calls_are_separate_with_settle_delay(self):
        runner = Mock(side_effect=[controls(), Mock(returncode=0), Mock(returncode=0), controls()])
        pause = Mock()
        with patch("rov_video_stream.shutil.which", return_value="v4l2-ctl"):
            video.apply_focus(self.config, runner=runner, pause=pause)
        self.assertEqual(runner.call_args_list[1].args[0][-1], "focus_automatic_continuous=0")
        self.assertEqual(runner.call_args_list[2].args[0][-1], "focus_absolute=256")
        self.assertEqual(runner.call_args_list[3].args[0][-1], "--list-ctrls")
        pause.assert_called_once_with(0.2)

    def test_auto_focus_enables_camera_auto_without_manual_position(self):
        self.config["focus_profile"] = "auto"
        runner = Mock(side_effect=[controls(), Mock(returncode=0), controls(auto=1)])
        pause = Mock()
        with patch("rov_video_stream.shutil.which", return_value="v4l2-ctl"):
            video.apply_focus(self.config, runner=runner, pause=pause)
        self.assertEqual(runner.call_count, 3)
        self.assertEqual(runner.call_args_list[1].args[0][-1], "focus_automatic_continuous=1")
        pause.assert_not_called()

    def test_legacy_focus_auto_alias_is_supported(self):
        runner = Mock(side_effect=[controls(name="focus_auto"), Mock(returncode=0),
                                  Mock(returncode=0), controls(name="focus_auto")])
        with patch("rov_video_stream.shutil.which", return_value="v4l2-ctl"):
            actual = video.apply_focus(self.config, runner=runner, pause=Mock())
        self.assertFalse(actual["autofocus"])
        self.assertEqual(runner.call_args_list[1].args[0][-1], "focus_auto=0")

    def test_manual_out_of_range_never_sends_controls(self):
        self.config["water_focus_value"] = 2000
        runner = Mock(return_value=controls())
        with patch("rov_video_stream.shutil.which", return_value="v4l2-ctl"):
            with self.assertRaisesRegex(ValueError, "range/step"):
                video.apply_focus(self.config, runner=runner)
        self.assertEqual(runner.call_count, 1)

    def test_failed_readback_is_not_success(self):
        runner = Mock(side_effect=[controls(), Mock(returncode=0), Mock(returncode=0), controls(focus=68)])
        with patch("rov_video_stream.shutil.which", return_value="v4l2-ctl"):
            with self.assertRaisesRegex(RuntimeError, "readback"):
                video.apply_focus(self.config, runner=runner, pause=Mock())

    def test_save_focus_only_after_successful_hardware_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "video_profile.json"
            profile.write_text(json.dumps(self.config), encoding="utf-8")
            with patch("rov_video_stream.apply_focus", side_effect=RuntimeError("rejected")):
                with self.assertRaises(RuntimeError):
                    video.main(["--config", str(profile), "--set-focus", "auto", "--save-focus"])
            self.assertEqual(video.load_profile(profile)["focus_profile"], "water")
            with patch("rov_video_stream.apply_focus", return_value={"autofocus": True}), patch("builtins.print"):
                video.main(["--config", str(profile), "--set-focus", "auto", "--save-focus"])
            self.assertEqual(video.load_profile(profile)["focus_profile"], "auto")

    def test_inactive_focus_is_not_a_live_autofocus_position_or_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "video_profile.json"
            self.config["focus_profile"] = "auto"
            profile.write_text(json.dumps(self.config), encoding="utf-8")
            runner = Mock(return_value=controls(auto=1, focus=256, inactive=True))
            status = video.focus_status(self.config, runner=runner)
            self.assertTrue(status["autofocus"])
            self.assertFalse(status["focus_readable"])
            self.assertIsNone(status["focus_value"])
            self.assertEqual(video.load_profile(profile)["water_focus_value"], 256)

    def test_nonvolatile_auto_focus_value_is_not_claimed_as_live(self):
        actual = video.focus_status(self.config, runner=Mock(return_value=controls(auto=1, focus=256)))
        self.assertFalse(actual["focus_readable"])
        self.assertIsNone(actual["focus_value"])

    def test_inactive_manual_focus_is_not_claimed_as_current(self):
        actual = video.focus_status(self.config, runner=Mock(return_value=controls(auto=0, focus=256, inactive=True)))
        self.assertFalse(actual["focus_readable"])
        self.assertIsNone(actual["focus_value"])

    def test_encoder_progress_is_bounded_file_and_not_a_latency_claim(self):
        process = Mock()
        process.pid = 1234
        process.stdout = iter(["frame=30\n", "fps=29.97\n", "out_time_us=1000000\n", "progress=continue\n"])
        process.wait.return_value = process.returncode = 0
        process.poll.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory) / "video_status.json"
            with patch("rov_video_stream.subprocess.Popen", return_value=process), patch("rov_video_stream.signal.signal"):
                video.run_encoder(["ffmpeg", "-i", "camera"], self.config, {}, status)
            report = json.loads(status.read_text(encoding="utf-8"))
            self.assertEqual(report["encoder_progress"]["fps"], "29.97")
            self.assertEqual(report["state"], "stopped")
            self.assertNotIn("latency_ms", report)

    def test_default_is_dry_run_without_camera_or_encoder(self):
        with patch("rov_video_stream.os.execvpe") as execute, \
             patch("rov_video_stream.apply_focus") as focus, patch("builtins.print"):
            self.assertEqual(video.main([]), 0)
            execute.assert_not_called()
            focus.assert_not_called()


if __name__ == "__main__":
    unittest.main()
