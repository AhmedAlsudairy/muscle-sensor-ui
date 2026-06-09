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
sudo cp /home/ahmed/muscle-sensor-ui/pi-api/emg-api.service /etc/systemd/system/

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
tail -f /var/log/emg-api.log

# Or via journalctl
sudo journalctl -u emg-api -f

# Last 50 lines
tail -50 /var/log/emg-api.log

# Search for errors
grep -i error /var/log/emg-api.log
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

## 5. Alternative: crontab @reboot

```bash
crontab -e
# Add line:
@reboot /home/ahmed/muscle-sensor-ui/pi-api/launch.sh
```

---

## 6. Test API

```bash
curl http://localhost:3001/health
curl http://localhost:3001/ports
```

---

## File Locations

| File | Path |
|------|------|
| Server | `/home/ahmed/muscle-sensor-ui/pi-api/server.py` |
| .env | `/home/ahmed/muscle-sensor-ui/pi-api/.env` |
| Service | `/etc/systemd/system/emg-api.service` |
| Logs | `/var/log/emg-api.log` |
| Launch script | `/home/ahmed/muscle-sensor-ui/pi-api/launch.sh` |
