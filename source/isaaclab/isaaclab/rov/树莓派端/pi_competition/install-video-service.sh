#!/bin/sh
set -eu
BASE_DIR=/home/liang/underwater_robot
UNIT=/etc/systemd/system/rov-video.service
if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo after verifying the video profile."
    exit 1
fi
command -v ffmpeg >/dev/null
command -v v4l2-ctl >/dev/null
test -f "$BASE_DIR/rov_video_stream.py"
test -f "$BASE_DIR/video_profile.json"
test -f "$BASE_DIR/rov-video.service"
# Only install and enable; do not stop an old camera service or start capture here.
if [ -f "$UNIT" ]; then
    cp -p "$UNIT" "$UNIT.bak.$(date +%Y%m%d-%H%M%S)"
fi
install -m 0644 "$BASE_DIR/rov-video.service" "$UNIT"
systemctl daemon-reload
systemctl enable rov-video.service
echo "Installed for next boot. Test manually, then start: sudo systemctl start rov-video.service"
