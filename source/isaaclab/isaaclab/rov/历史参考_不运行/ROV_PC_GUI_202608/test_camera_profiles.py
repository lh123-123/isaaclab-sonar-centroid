import inspect
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

from pc_gui import (
    CAMERA_PROFILE_LABELS,
    VideoStream,
    build_camera_profile_request,
    camera_profile_endpoint,
)
from usb_mjpeg_server import (
    CAMERA_PROFILES,
    Frames,
    apply_camera_profile,
    handler_factory,
)


class CameraProfileDefinitionTests(unittest.TestCase):
    def test_water_profile_preserves_the_measured_underwater_settings(self):
        controls = dict(CAMERA_PROFILES["water"])

        self.assertEqual(controls["focus_auto"], 0)
        self.assertEqual(controls["focus_absolute"], 256)
        self.assertEqual(controls["sharpness"], 98)
        self.assertEqual(controls["exposure_auto"], 3)
        self.assertEqual(controls["white_balance_temperature_auto"], 1)

    def test_shore_profile_preserves_the_measured_shore_settings(self):
        controls = dict(CAMERA_PROFILES["shore"])

        self.assertEqual(controls["focus_auto"], 0)
        self.assertEqual(controls["focus_absolute"], 222)
        self.assertEqual(controls["sharpness"], 98)

    def test_autofocus_profile_restores_continuous_focus(self):
        controls = dict(CAMERA_PROFILES["autofocus"])

        self.assertEqual(controls["focus_auto"], 1)
        self.assertNotIn("focus_absolute", controls)
        self.assertEqual(controls["sharpness"], 98)

    def test_controls_are_applied_one_ioctl_at_a_time(self):
        calls = []
        pauses = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))

        apply_camera_profile(
            "/dev/video7", "water", runner=runner, pause=pauses.append
        )

        self.assertTrue(calls)
        self.assertTrue(all(call[0][0] == "v4l2-ctl" for call in calls))
        self.assertTrue(all(call[0][2] == "/dev/video7" for call in calls))
        settings = [call[0][-1] for call in calls]
        self.assertLess(settings.index("focus_auto=0"), settings.index("focus_absolute=256"))
        self.assertEqual(pauses, [0.2])


class CameraProfileHttpTests(unittest.TestCase):
    def setUp(self):
        self.applied = []
        state = {"current": "water"}

        def apply_profile(device, profile):
            self.applied.append((device, profile))

        handler = handler_factory(
            Frames(),
            "127.0.0.1",
            device="/dev/video9",
            profile_state=state,
            profile_lock=threading.Lock(),
            profile_applier=apply_profile,
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.opener = build_opener(ProxyHandler({}))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=2)

    def test_valid_profile_is_applied_and_returned(self):
        request = build_camera_profile_request(
            self.base + "/stream.mjpg", "shore"
        )
        with self.opener.open(request, timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload, {"ok": True, "profile": "shore"})
        self.assertEqual(self.applied, [("/dev/video9", "shore")])

    def test_unknown_profile_is_rejected_without_touching_camera(self):
        request = build_camera_profile_request(
            self.base + "/stream.mjpg", "unknown"
        )
        with self.assertRaises(HTTPError) as caught:
            self.opener.open(request, timeout=1)

        try:
            self.assertEqual(caught.exception.code, 400)
        finally:
            caught.exception.close()
        self.assertEqual(self.applied, [])


class CameraProfileGuiTests(unittest.TestCase):
    def test_profile_endpoint_reuses_the_direct_camera_origin(self):
        self.assertEqual(
            camera_profile_endpoint("http://192.168.1.3:8080/stream.mjpg"),
            "http://192.168.1.3:8080/camera/profile",
        )

    def test_video_toolbar_exposes_three_compact_chinese_presets(self):
        source = inspect.getsource(VideoStream.__init__)

        self.assertEqual(
            CAMERA_PROFILE_LABELS,
            {
                "水下（已标定）": "water",
                "岸上（已标定）": "shore",
                "自动对焦（默认）": "autofocus",
            },
        )
        self.assertIn('text="画质预设"', source)
        self.assertIn('state="readonly"', source)
        self.assertIn('width=14', source)
        self.assertIn('bind("<<ComboboxSelected>>"', source)


if __name__ == "__main__":
    unittest.main()
