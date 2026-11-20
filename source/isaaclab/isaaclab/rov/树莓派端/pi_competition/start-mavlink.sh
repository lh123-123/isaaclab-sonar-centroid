#!/bin/sh
set -eu

BASE_DIR=/home/liang/underwater_robot
PID_FILE=$BASE_DIR/mavlink-bridge.pid
LOG_FILE=$BASE_DIR/logs/mavlink-bridge.log

mkdir -p "$BASE_DIR/logs"

if [ -f "$PID_FILE" ]; then
    bridge_pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$bridge_pid" ] && kill -0 "$bridge_pid" 2>/dev/null; then
        echo "MAVLink bridge is already running: PID $bridge_pid"
        exit 0
    fi
fi

nohup /usr/bin/python3 "$BASE_DIR/rov_mavlink_bridge.py" \
    --serial /dev/serial/by-id/usb-ArduPilot_Pixhawk1_3B0042000450325635393420-if00 \
    --baud 115200 \
    --gcs-ip 192.168.1.4 \
    --udp-port 14550 \
    >> "$LOG_FILE" 2>&1 &

bridge_pid=$!
echo "$bridge_pid" > "$PID_FILE"
sleep 1
kill -0 "$bridge_pid"
echo "MAVLink bridge started: PID $bridge_pid"
