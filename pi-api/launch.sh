#!/bin/bash
# launch.sh — Auto-start EMG API on boot with logging
# Place in /home/ahmed/muscle-sensor-ui/pi-api/
# Add to crontab: @reboot /home/ahmed/muscle-sensor-ui/pi-api/launch.sh

LOG_DIR="/var/log/emg-api"
LOG_FILE="$LOG_DIR/emg-api.log"
API_DIR="/home/ahmed/muscle-sensor-ui/pi-api"

mkdir -p "$LOG_DIR"

echo "========================================" >> "$LOG_FILE"
echo "[$(date)] EMG API starting..." >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

cd "$API_DIR"
/usr/bin/python3 "$API_DIR/server.py" >> "$LOG_FILE" 2>&1
