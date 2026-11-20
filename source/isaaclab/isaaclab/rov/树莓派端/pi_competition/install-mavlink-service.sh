#!/bin/sh
set -eu

BASE_DIR=/home/liang/underwater_robot
SERVICE_NAME=rov-mavlink.service

if [ "$(id -u)" -ne 0 ]; then
    echo "请使用 sudo 运行：sudo $BASE_DIR/install-mavlink-service.sh" >&2
    exit 1
fi

if [ -f "$BASE_DIR/mavlink-bridge.pid" ]; then
    bridge_pid=$(cat "$BASE_DIR/mavlink-bridge.pid" 2>/dev/null || true)
    if [ -n "$bridge_pid" ] && [ -r "/proc/$bridge_pid/cmdline" ]; then
        bridge_command=$(tr '\000' ' ' < "/proc/$bridge_pid/cmdline")
        case "$bridge_command" in
            *rov_mavlink_bridge.py*)
                kill "$bridge_pid" 2>/dev/null || true
                ;;
        esac
    fi
    rm -f "$BASE_DIR/mavlink-bridge.pid"
fi

install -m 0644 "$BASE_DIR/$SERVICE_NAME" "/etc/systemd/system/$SERVICE_NAME"
systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"
systemctl --no-pager --full status "$SERVICE_NAME"
