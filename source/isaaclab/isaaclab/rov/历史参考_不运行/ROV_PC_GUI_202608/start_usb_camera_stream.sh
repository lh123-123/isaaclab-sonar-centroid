#!/usr/bin/env bash
# Display-only USB-camera MJPEG fan-out server for the ROV PC GUI.
# Run on the Pi: ./start_usb_camera_stream.sh
# Optional: ./start_usb_camera_stream.sh /dev/video0 192.168.1.3 8080 256
set -euo pipefail

device="${1:-/dev/video0}"
host="${2:-192.168.1.3}"
port="${3:-8080}"
focus="${4:-256}"

case "$focus" in
  ''|*[!0-9]*)
    echo "warning: invalid camera focus '$focus'; using 256" >&2
    focus=256
    ;;
esac
if (( focus > 1023 )); then
  echo "warning: camera focus '$focus' exceeds 1023; using 256" >&2
  focus=256
fi

# This UVC driver rejects changing autofocus and absolute focus in one ioctl,
# so apply the two controls separately.  A camera without these controls must
# still be allowed to start streaming.
if command -v v4l2-ctl >/dev/null 2>&1; then
  if ! v4l2-ctl -d "$device" -c focus_auto=0; then
    echo "warning: unable to disable camera autofocus" >&2
  else
    sleep 0.2
    if ! v4l2-ctl -d "$device" -c focus_absolute="$focus"; then
      echo "warning: unable to set manual camera focus $focus" >&2
    fi
  fi
else
  echo "warning: v4l2-ctl unavailable; camera focus unchanged" >&2
fi

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec python3 "$script_dir/usb_mjpeg_server.py" \
  --device "$device" --bind-host "$host" --port "$port" \
  --allow-client 192.168.1.4 \
  --video-size 1280x720 --framerate 30 --output-framerate 30 \
  --jpeg-quality 3
