# EMG API — Arduino → Serial → Neon DB

Reads live EMG values from an Arduino over USB serial and stores them in a **Neon (PostgreSQL)** database. Exposes a REST + SSE API for the [Muscle Sensor UI](https://v0-muscle-sensor-ui.vercel.app/).

---

## Setup

### 1. Install dependencies
```bash
cd emg-api
npm install
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in your Neon DB URL and serial port
```

Get your `DATABASE_URL` from [console.neon.tech](https://console.neon.tech) → your project → **Connection string**.

### 3. Find your Arduino port (Linux)
```bash
ls /dev/tty{USB,ACM}*
# Usually /dev/ttyUSB0 or /dev/ttyACM0
```

If you get a **Permission denied** error:
```bash
sudo usermod -aG dialout $USER
# Then log out and back in
```

### 4. Flash the Arduino
Upload this sketch to your Arduino UNO:
```cpp
int EMGPin = A0;
int EMGVal = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  EMGVal = analogRead(EMGPin);
  Serial.println(EMGVal);
  delay(30);
}
```

### 5. Start the server
```bash
npm start
# or for auto-reload during development:
npm run dev
```

The server starts on `http://localhost:3000` and **auto-connects** to the serial port defined in `.env`.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status + serial connection state |
| `GET` | `/ports` | List available serial ports |
| `POST` | `/connect` | Open Arduino serial port |
| `POST` | `/disconnect` | Close serial port |
| `GET` | `/readings` | Recent readings from DB |
| `GET` | `/stats` | Min / max / avg over last N minutes |
| `POST` | `/readings` | Manual insert (testing) |
| `GET` | `/stream` | **SSE** live feed (real-time waveform) |

### POST /connect
```json
{ "port": "/dev/ttyUSB0", "baudRate": 9600 }
```

### GET /readings
```
/readings?limit=100&offset=0
```
Returns array of `{ id, raw_value, percentage, created_at }`.

### GET /stats
```
/stats?minutes=5
```
Returns `{ total, min_raw, max_raw, avg_raw, avg_pct }`.

### POST /readings (test without Arduino)
```json
{ "rawValue": 512 }
```

### GET /stream (Server-Sent Events)
Connect from the dashboard frontend:
```js
const es = new EventSource("http://localhost:3000/stream");
es.onmessage = (e) => {
  const { raw_value, percentage, created_at } = JSON.parse(e.data);
  // update chart...
};
```

---

## DB Schema

```sql
CREATE TABLE emg_readings (
  id          BIGSERIAL PRIMARY KEY,
  raw_value   INTEGER        NOT NULL,   -- 0–1023 from Arduino 10-bit ADC
  percentage  NUMERIC(5,2)   NOT NULL,   -- 0.00–100.00 %
  created_at  TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);
```

Created automatically on first startup.
