# AI-Based Muscle Fatigue Detection using EMG and IoT

Real-time EMG signal monitoring dashboard with an in-browser AI classifier that detects **Normal** vs **Fatigue** muscle states.

---

## Overview

This project streams electromyography (EMG) data from an Arduino sensor through a Raspberry Pi to a Next.js web dashboard. A neural network trained on the incoming data classifies each window of readings as either **Normal** (low EMG activity) or **Fatigue** (elevated EMG activity), and reports **Accuracy** and **F1 Score** so you can evaluate model performance directly in the browser.

```
Arduino (EMG sensor) → USB Serial → Raspberry Pi (Node.js API) → Neon PostgreSQL → Next.js Dashboard → brain.js AI
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Sensor | Grove EMG Sensor v3 on Arduino UNO |
| Edge API | Node.js + serialport (Raspberry Pi) |
| Database | Neon (serverless PostgreSQL) |
| Frontend | Next.js 16 · TypeScript · Tailwind CSS · shadcn/ui |
| AI / ML | **brain.js** — feedforward neural network (in-browser, CPU mode) |
| Charts | Recharts |

---

## AI Model — brain.js Neural Network

The AI panel uses [brain.js](https://github.com/BrainJS/brain.js), a pure-JavaScript neural network library that runs entirely in the browser.

### Architecture

```
Input (20 normalised EMG values)
        ↓
  Hidden layer — 16 neurons, ReLU activation
        ↓
  Output layer — 2 neurons (softmax)
        ↓
  [P(Normal), P(Fatigue)]
```

- **Input window**: last 20 readings, each normalised to `[0, 1]` by dividing `signal_percentage / 100`
- **Output**: two-class probability distribution
- **Optimiser**: SGD · learning rate 0.05 · 100 epochs
- **Train / val split**: 80 % training, 20 % validation

### Classes

| Label | Signal % | Meaning |
|-------|----------|---------|
| **Normal** (0) | < 30 % | Low muscle activity, no fatigue |
| **Fatigue** (1) | ≥ 30 % | Elevated activity, fatigue detected |

### Metrics displayed

| Metric | Formula | What it means |
|--------|---------|---------------|
| **Accuracy** | (TP + TN) / N | Overall % of correct predictions |
| **F1 Score** | 2 · P · R / (P + R) | Harmonic mean of precision and recall; robust to class imbalance |

Where **P** = Precision = TP / (TP + FP) and **R** = Recall = TP / (TP + FN).  
Positive class = **Fatigue**.

Both metrics are computed on the held-out validation set and displayed as large cards in the AI panel after training.

---

## Project Structure

```
muscle-sensor-ui/
├── app/
│   ├── page.tsx                # Main dashboard (LED indicators, waveform, stats)
│   ├── layout.tsx
│   └── api/
│       ├── readings/route.ts   # GET/DELETE readings from Neon DB
│       └── setup/route.ts      # Create muscle_readings table
├── components/
│   ├── ai-panel.tsx            # brain.js classifier — training, inference, metrics
│   ├── muscle-indicator.tsx    # Animated muscle state widget (Normal / Moderate / Fatigue)
│   ├── muscle-waveform.tsx     # Real-time EMG waveform chart
│   ├── history-chart.tsx       # 30-point signal history
│   ├── readings-table.tsx      # Paginated readings table
│   └── stats-card.tsx          # Current / Peak / Average / Session cards
├── lib/
│   ├── neural-net.ts           # Dataset builder, computeMetrics (Accuracy + F1), shuffleIndices
│   ├── db.ts                   # Neon DB connection
│   └── utils.ts
├── pi-api/                     # Raspberry Pi edge server (see pi-api/README.md)
│   ├── server.js               # Express API + serial reader
│   ├── serial.js               # Arduino serial port handler
│   ├── db.js                   # Neon DB writes
│   └── gpio.js                 # Optional GPIO LED output
└── next.config.mjs             # brain.js excluded from server bundle
```

---

## Getting Started

### Prerequisites

- Node.js ≥ 18
- A [Neon](https://neon.tech) database (free tier is sufficient)
- Arduino UNO with Grove EMG Sensor (or the Pi-API mock endpoint)

### 1. Clone and install

```bash
git clone <repo-url>
cd muscle-sensor-ui
npm install
```

### 2. Configure environment

```bash
cp .env.example .env.local
```

Set your Neon connection string:

```env
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
```

### 3. Run the development server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

On first load the dashboard automatically creates the `muscle_readings` table in your Neon database.

### 4. Connect the Pi API (optional)

See [pi-api/README.md](pi-api/README.md) for full Arduino + Raspberry Pi setup instructions.

---

## Using the AI Classifier

1. **Connect** the dashboard (click the **Connect** button).
2. Let it collect at least **50 readings** — a progress bar shows how many have been gathered.
3. Click **Train** — the neural network trains for 100 epochs entirely in your browser using brain.js. A progress bar and live loss value are shown.
4. After training, the panel shows:
   - **Val Accuracy** — overall classification accuracy on the validation set
   - **F1 Score** — precision-recall balance (more reliable than accuracy for imbalanced data)
   - **Live Prediction** — Normal or Fatigue with confidence %, updated every 2 s
5. Click **Retrain** at any time to rebuild the model with the latest readings.

---

## LED Indicator States

The two-LED panel on the right mirrors a physical LED circuit on the Pi:

| LED | State | Condition |
|-----|-------|-----------|
| 🟢 Green | **NORMAL** | signal < 30 % |
| 🔴 Red | **FATIGUE** | signal ≥ 30 % |

---

## Pi API

The Raspberry Pi runs a lightweight Node.js server that:

1. Reads raw ADC values from the Arduino over USB serial
2. Converts them to `signal_value` (0–1023) and `signal_percentage` (0–100 %)
3. Derives a `status` field (`normal` / `fatigue`)
4. Writes each reading to Neon PostgreSQL
5. Optionally toggles GPIO LEDs to mirror the fatigue state

See [pi-api/README.md](pi-api/README.md) for full setup.

---

## Scripts

```bash
npm run dev      # Start Next.js dev server
npm run build    # Production build
npm run start    # Start production server
npm run lint     # ESLint
```

---

## Acknowledgements

- [brain.js](https://github.com/BrainJS/brain.js) — in-browser neural networks
- [Neon](https://neon.tech) — serverless PostgreSQL
- [shadcn/ui](https://ui.shadcn.com) — component library
- [Recharts](https://recharts.org) — charting

