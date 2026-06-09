# EMG API — Auto-Start & Logging Guide

## 1. Set Raspberry Pi Hostname (for `ahmed@emg` prompt)

```bash
sudo hostnamectl set-hostname emg
sudo sed -i 's/127.0.1.1.*/127.0.1.1 emg/' /etc/hosts
# Reboot or run: exec bash
```

After reboot, your prompt will show: `ahmed@emg:~ $`

---

## 2. Install & Enable systemd Service (Auto-Start on Boot)

```bash
# Copy service file
sudo cp /home/ahmed/Downloads/muscle-sensor-ui/pi-api/emg-api.service /etc/systemd/system/

# Reload + enable + start
sudo systemctl daemon-reload
sudo systemctl enable emg-api
sudo systemctl start emg-api

# Check status
sudo systemctl status emg-api
```

---

## 3. View Live Logs

```bash
# Real-time (follow)
sudo journalctl -u emg-api -f

# Last 50 lines
sudo journalctl -u emg-api -n 50 --no-pager

# Search for errors
sudo journalctl -u emg-api --no-pager | grep -i error
```

---

## 4. Manual Control

```bash
sudo systemctl start emg-api      # Start
sudo systemctl stop emg-api       # Stop
sudo systemctl restart emg-api    # Restart
sudo systemctl status emg-api     # Status
sudo systemctl disable emg-api    # Disable auto-start
```

---

## 5. Connect / Disconnect Arduino Serial

```bash
# Connect (replace port if needed — check with: ls /dev/ttyUSB* /dev/ttyACM*)
curl -X POST http://localhost:3000/connect \
     -H "Content-Type: application/json" \
     -d '{"port":"/dev/ttyUSB0","baudRate":9600}'

# Disconnect
curl -X POST http://localhost:3000/disconnect
```

---

## 6. Alternative: crontab @reboot

```bash
crontab -e
# Add line:
@reboot /home/ahmed/Downloads/muscle-sensor-ui/pi-api/launch.sh
```

---

## 7. Test API

```bash
curl http://localhost:3000/health
curl http://localhost:3000/ports
curl http://localhost:3000/readings
curl http://localhost:3000/stats
```

---

## File Locations

| File | Path |
|------|------|
| Server | `/home/ahmed/Downloads/muscle-sensor-ui/pi-api/server.py` |
| DB layer | `/home/ahmed/Downloads/muscle-sensor-ui/pi-api/db.py` |
| Serial reader | `/home/ahmed/Downloads/muscle-sensor-ui/pi-api/serial_reader.py` |
| GPIO control | `/home/ahmed/Downloads/muscle-sensor-ui/pi-api/gpio_control.py` |
| .env | `/home/ahmed/Downloads/emg-api/.env` |
| Service | `/etc/systemd/system/emg-api.service` |
| Logs | `sudo journalctl -u emg-api` |
| Launch script | `/home/ahmed/Downloads/muscle-sensor-ui/pi-api/launch.sh` |
