# EMG API — Arduino to Serial to Neon DB

Reads live EMG values from an Arduino over USB serial and stores them in a **Neon (PostgreSQL)** database. Exposes a REST + SSE API for the Muscle Sensor UI dashboard.

---

## Quick Start

### 1. Install
```bash
cd pi-api
npm install
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your Neon DB URL and serial port
```

### 3. Flash Arduino
Upload this sketch to your Arduino UNO (Grove EMG Sensor on A0):
```cpp
int EMGPin = A0;
int EMGVal = 0;

void setup() { Serial.begin(9600); }

void loop() {
  EMGVal = analogRead(EMGPin);
  Serial.println(EMGVal);
  delay(30);
}
```

### 4. Start
```bash
npm start       # production
npm run dev     # auto-reload (nodemon)
```

Server runs on `http://localhost:3001` (avoiding conflict with Next.js on 3000).

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status + serial state |
| `GET` | `/ports` | List available serial ports |
| `POST` | `/connect` | Open Arduino serial `{ "port": "/dev/ttyUSB0", "baudRate": 9600 }` |
| `POST` | `/disconnect` | Close serial port |
| `GET` | `/readings?limit=100` | Recent readings from DB |
| `GET` | `/stats?minutes=5` | Min/max/avg over N minutes |
| `POST` | `/readings` | Manual test insert `{ "rawValue": 512 }` |
| `GET` | `/stream` | **SSE** real-time live feed |

---

## Status Classification (Two-Class)

| Status | Threshold | LED |
|--------|-----------|-----|
| **Normal** | `< 30%` | Green ON |
| **Fatigue** | `>= 30%` | Red ON + Buzzer |

Matches the dashboard's two-class muscle state indicator.

---

## DB Schema

```sql
CREATE TABLE muscle_readings (
  id                SERIAL PRIMARY KEY,
  signal_value      DECIMAL(10, 2) NOT NULL,   -- 0-1023 from ADC
  signal_percentage DECIMAL(5, 2)  NOT NULL,   -- 0-100%
  status            VARCHAR(20)    NOT NULL,   -- 'normal' | 'fatigue'
  peak_value        DECIMAL(10, 2),
  average_value     DECIMAL(10, 2),
  created_at        TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);
```

Auto-created on startup via `initDB()`.

---

## Config (`config.json`)

```json
{
  "port": 3001,
  "serial": { "baudRate": 9600 },
  "emg": { "thresholdFatigue": 30, "labels": { "normal": "Normal", "fatigue": "Fatigue" } },
  "gpio": { "pins": { "redLed": 14, "greenLed": 15, "buzzer": 18 } }
}
```

---

## GPIO Wiring (Raspberry Pi 5)

| GPIO Pin | Component | Purpose |
|----------|-----------|---------|
| 14 | Red LED | Fatigue indicator |
| 15 | Green LED | Normal indicator |
| 18 | Buzzer | Audible fatigue alert |

---

## systemd Service

```bash
sudo cp emg-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable emg-api
sudo systemctl start emg-api
```
