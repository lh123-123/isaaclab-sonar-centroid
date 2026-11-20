#!/bin/sh
set -eu

sudo systemctl stop rov-usst.service
stamp=$(date +%Y%m%d-%H%M%S)
backup="/etc/rov/config.ini.before-servo-$stamp"
sudo cp -a /etc/rov/config.ini "$backup"
sudo sed -i \
  -e '/^\[servo\]/,/^\[/ { s/^output_enabled = .*/output_enabled = true/; s/^pulse_min_us = .*/pulse_min_us = 960/; s/^pulse_center_us = .*/pulse_center_us = 1130/; s/^pulse_max_us = .*/pulse_max_us = 1500/; s/^angle_min_deg = .*/angle_min_deg = -50.0/; s/^angle_max_deg = .*/angle_max_deg = 50.0/; s/^failsafe_position = .*/failsafe_position = 0.0/; }' \
  /etc/rov/config.ini

cd /opt/rov/usst
sudo -u rov env ROV_CONFIG=/etc/rov/config.ini ./.venv/bin/python -c \
  'from config_loader import load_config; c=load_config(); print("validated", dict(c["servo"]))'
./.venv/bin/python -c \
  'import pathlib; [compile(pathlib.Path(name).read_text(), name, "exec") for name in ("main.py", "config_loader.py", "secure_protocol.py")]'
sudo systemctl start rov-usst.service
sleep 2
systemctl is-active rov-usst.service
systemctl --no-pager --full status rov-usst.service | head -n 12
echo "BACKUP=$backup"
