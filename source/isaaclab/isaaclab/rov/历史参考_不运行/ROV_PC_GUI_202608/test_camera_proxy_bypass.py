import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from pc_gui import open_direct_http_stream


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"direct-camera-ok")

    def log_message(self, _format, *_args):
        pass


class CameraProxyBypassTests(unittest.TestCase):
    def test_rov_camera_ignores_global_http_proxy(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}/stream.mjpg"
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "HTTP_PROXY": "http://127.0.0.1:1",
                    "HTTPS_PROXY": "http://127.0.0.1:1",
                    "NO_PROXY": "",
                },
                clear=False,
            ):
                with open_direct_http_stream(url, timeout=1) as response:
                    self.assertEqual(response.read(), b"direct-camera-ok")
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
