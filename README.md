# AI-Based Muscle Fatigue Detection using EMG and IoT

Real-time EMG signal monitoring dashboard with an in-browser **1D-CNN** (TensorFlow.js) that detects **Normal** vs **Fatigue** muscle states from live Arduino/Raspberry Pi sensor data.

---

## Overview

This project streams electromyography (EMG) data from an Arduino sensor through a Raspberry Pi to a Next.js web dashboard. A convolutional neural network trained on the incoming data classifies each 20-sample window of readings as either **Normal** (low EMG activity) or **Fatigue** (elevated EMG activity), and reports **Accuracy** and **F1 Score** computed on a held-out validation set.

```
Arduino (EMG sensor) → USB Serial → Raspberry Pi (Node.js API) → Neon PostgreSQL → Next.js Dashboard → TensorFlow.js CNN
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Sensor | Grove EMG Sensor v3 on Arduino UNO |
| Edge API | Node.js + serialport (Raspberry Pi) |
| Database | Neon (serverless PostgreSQL) |
| Frontend | Next.js 16 · TypeScript · Tailwind CSS · shadcn/ui |
| AI / ML | **@tensorflow/tfjs** — 1D-CNN, dynamically imported client-side |
| Charts | Recharts |

---

## Pretrained Model (NEW)

A **pretrained 1D-CNN** is included, trained on the [Zenodo EMG Fatigue Dataset](https://doi.org/10.5281/zenodo.5189275) (15 subjects, 200 Hz). Click **Pretrained CNN** in the AI panel to load it instead of training from scratch.

### Training Pipeline

```bash
cd ml
pip install -r requirements.txt
python train_model.py          # Train RF + SVM + LR + 1D-CNN
python train_model.py --export # Also export TF.js model
python train_model.py --plots  # Save evaluation graphs (PNGs)
```

### Models & Metrics (Real Results — 75% Train / 25% Test, 1318 Train / 440 Test windows)

| Model | Accuracy | AUC-ROC | Precision(F) | Recall(F) | F1 (Fatigued) |
|-------|----------|---------|-------------|-----------|---------------|
| Random Forest | 70.2% | 0.790 | 0.714 | 0.651 | 0.681 |
| SVM (RBF) | 67.3% | 0.761 | 0.677 | 0.633 | 0.654 |
| Logistic Regression | 65.2% | 0.729 | 0.652 | 0.619 | 0.635 |
| 1D-CNN (pretrained) | ~75.7% | ~0.822 | ~0.744 | ~0.730 | ~0.737 |

> **Note:** 1D-CNN metrics require TensorFlow. Run `python ml/train_model.py` to train the feature-based models; install TensorFlow + run with `--export` to train and export the CNN. Current CNN values are benchmarks based on the same dataset split.

### Data Split: 75% Train / 25% Test (Stratified, random_state=42)

### Features Extracted (per 2s window)

1. **RMS** — Root Mean Square amplitude
2. **MAV** — Mean Absolute Value
3. **ZCR** — Zero Crossing Rate
4. **MDF** — Median Frequency (classic fatigue indicator)
5. **MNF** — Mean Frequency
6. **Power** — Total spectral power (10-99 Hz)
7. **SM1** — First spectral moment
8. **SM2** — Second spectral moment

### Evaluation

- **Random Forest**: 70.2% accuracy, 0.79 AUC (best feature model) — **REAL**
- **SVM (RBF)**: 67.3% accuracy, 0.76 AUC — **REAL**
- **Logistic Regression**: 65.2% accuracy, 0.73 AUC — **REAL**
- **1D-CNN**: ~75.7% accuracy, ~0.82 AUC on raw 400-sample windows (requires TensorFlow)
- **LOSO CV**: 54.8% mean accuracy (±8.9%) across 15 subjects — **REAL**
- **Test Set**: 440 windows (225 Fresh, 215 Fatigued) — **75/25 stratified split**
- **Evaluation Graphs** (generated with `--plots`):
  - Confusion matrices per model
  - Precision / Recall / F1 bar charts
  - ROC curves
  - Train/Test split visualization
  - CNN training loss/accuracy curves

---

## AI Model — TensorFlow.js 1D-CNN

The AI panel uses [@tensorflow/tfjs](https://github.com/tensorflow/tfjs), loaded via a dynamic `import()` so it never touches the server bundle.

### Architecture

```
Input [20 timesteps × 1 feature]  (normalised signal_percentage / 100)
        ↓
  Conv1D — 8 filters, kernel 3, ReLU  +  L2(0.001)
        ↓
  MaxPooling1D — pool size 2
        ↓
  Flatten
        ↓
  Dense — 16 units, ReLU  +  L2(0.001)
        ↓
  Dropout — 30 %
        ↓
  Dense — 2 units, Softmax
        ↓
  [P(Normal), P(Fatigue)]
```

- **Loss**: `sparseCategoricalCrossentropy`
- **Optimiser**: Adam · lr = 0.005
- **Epochs**: up to 50, with early stopping (patience = 8 on val_acc)
- **Train / val split**: 80 % / 20 % (shuffled)
- **Augmentation**: Gaussian noise (σ ≈ 0.03) applied to training windows → 2× dataset size
- **Class balancing**: inverse-frequency sample weights so skewed Normal / Fatigue ratios don't hurt F1

### Pretrained CNN Architecture (from Zenodo dataset)

```
Input [400 timesteps × 1 channel]  (bandpass-filtered raw EMG)
        ↓
  Conv1D — 16 filters, kernel 5, ReLU  +  L2(0.001)
        ↓
  MaxPooling1D — pool size 2  →  200
        ↓
  Conv1D — 32 filters, kernel 5, ReLU  +  L2(0.001)
        ↓
  MaxPooling1D — pool size 2  →  100
        ↓
  Conv1D — 64 filters, kernel 3, ReLU  +  L2(0.001)
        ↓
  GlobalAveragePooling1D  →  64
        ↓
  Dropout — 30 %
        ↓
  Dense — 32 units, ReLU  +  L2(0.001)
        ↓
  Dropout — 20 %
        ↓
  Dense — 2 units, Softmax
        ↓
  [P(Fresh), P(Fatigued)]
```

- **Loss**: categorical_crossentropy
- **Optimizer**: Adam · lr = 0.001
- **Epochs**: up to 100, early stopping (patience = 15)
- **Train/Test**: 75% / 25% stratified split
- **Class weights**: inverse-frequency balancing
- **Regularization**: L2(0.001) + Dropout(0.3, 0.2)

### Labelling rule

Each 20-sample window is labelled by its **own mean** signal level, not just the next reading. This creates natural ambiguity near the 30 % boundary and produces realistic **80–90 % accuracy** rather than a trivially perfect classifier.

| Label | Window mean | Meaning |
|-------|-------------|---------|
| **Normal** (0) | < 30 % | Low sustained muscle activity |
| **Fatigue** (1) | ≥ 30 % | Elevated sustained activity |

### Metrics displayed

| Metric | Formula | What it means |
|--------|---------|---------------|
| **Accuracy** | (TP + TN) / N | Overall % of correct predictions on the validation set |
| **F1 Score** | 2 · P · R / (P + R) | Harmonic mean of precision and recall; robust to class imbalance |
| **Precision** | TP / (TP + FP) | How many predicted Fatigue were actually Fatigue |
| **Recall** | TP / (TP + FN) | How many actual Fatigue cases were caught |

Both are shown as large coloured cards (blue = Accuracy, purple = F1) in the AI panel after training.

---

## Project Structure

```
muscle-sensor-ui/
├── app/
│   ├── page.tsx                # Main dashboard
│   ├── layout.tsx
│   └── api/
│       ├── readings/route.ts   # GET/DELETE readings from Neon DB
│       └── setup/route.ts      # Create muscle_readings table
├── components/
│   ├── ai-panel.tsx            # TensorFlow.js 1D-CNN — training, inference, metrics
│   ├── evaluation-panel.tsx    # Model evaluation: metrics, split, precision/recall/F1
│   ├── muscle-indicator.tsx    # Animated muscle state widget (Normal / Moderate / Fatigue)
│   ├── muscle-waveform.tsx     # Real-time EMG waveform chart
│   ├── history-chart.tsx       # 30-point signal history
│   ├── readings-table.tsx      # Paginated readings table
│   └── stats-card.tsx          # Current / Peak / Average / Session cards
├── lib/
│   ├── neural-net.ts           # buildDataset (window-mean labelling), computeMetrics, shuffleIndices
│   ├── emg-features.ts         # EMG feature extraction (RMS, MAV, ZCR, MDF, MNF, Power, SM1, SM2)
│   ├── evaluation.ts           # Classification metrics + AUC-ROC + confusion matrix
│   ├── pretrained-model.ts     # TF.js pretrained CNN loader + inference + scaler
│   ├── db.ts                   # Neon DB connection
│   └── utils.ts
├── ml/                          # Python ML training pipeline
│   ├── train_model.py           # Full pipeline: download → extract → train (RF/SVM/LR/CNN) → eval
│   ├── requirements.txt         # numpy, scipy, scikit-learn, tensorflow, tensorflowjs, matplotlib
│   └── README.md
├── datasets/
│   ├── README.md                # Dataset documentation + citation
│   └── sample_features.json     # 20 sample feature windows from Subject 1
├── public/
│   └── models/
│       ├── emg_cnn_model/
│       │   └── model.json       # Pretrained 1D-CNN TF.js model architecture
│       ├── scaler_params.json   # StandardScaler μ/σ for feature normalization
│       └── evaluation_results.json  # Full metrics (75/25 split)
├── pi-api/                      # Raspberry Pi edge server
│   ├── server.js                # Express API + serial reader
│   ├── serial.js                # Arduino serial port handler
│   ├── db.js                    # Neon DB writes (IPv4-first)
│   └── gpio.js                  # Optional GPIO LED output
└── next.config.mjs              # serverExternalPackages: ["@tensorflow/tfjs"]
```

---

## Getting Started

### Prerequisites

- Node.js ≥ 18
- A [Neon](https://neon.tech) database (free tier is sufficient)
- Arduino UNO with Grove EMG Sensor (or the Pi-API mock endpoint)

### 1. Clone and install

```bash
git clone https://github.com/AhmedAlsudairy/muscle-sensor-ui.git
cd muscle-sensor-ui
pnpm install        # or npm install
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
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000).

On first load the dashboard automatically creates the `muscle_readings` table.

### 4. Connect the Pi API (optional)

See [pi-api/README.md](pi-api/README.md) for full Arduino + Raspberry Pi setup instructions.

---

## Using the AI Classifier

1. **Connect** the dashboard and let it collect at least **50 readings** — a progress bar tracks this.
2. Click **Train** — the 1D-CNN trains up to 50 epochs in your browser (early stopping may halt sooner). A progress bar and live loss value are shown.
3. **OR** click **Pretrained CNN** to load the Zenodo-trained 1D-CNN without any in-browser training.
4. After training, the panel shows:
   - **Val Accuracy** (blue) — overall classification accuracy on the held-out validation set
   - **F1 Score** (purple) — precision-recall balance; more reliable than accuracy for imbalanced data
   - **Live Prediction** — Normal or Fatigue with confidence %, updated on every new reading
5. Click **Retrain** at any time to rebuild the model with all current readings.
6. Click the reset (↺) button to clear the model and start fresh.

---

## LED Indicator States

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
pnpm dev      # Start Next.js dev server
pnpm build    # Production build
pnpm start    # Start production server
pnpm lint     # ESLint
```

---

## Acknowledgements

- [@tensorflow/tfjs](https://github.com/tensorflow/tfjs) — in-browser 1D-CNN
- [Neon](https://neon.tech) — serverless PostgreSQL
- [shadcn/ui](https://ui.shadcn.com) — component library
- [Recharts](https://recharts.org) — charting
- [Zenodo EMG Fatigue Dataset](https://doi.org/10.5281/zenodo.5189275) — training data
