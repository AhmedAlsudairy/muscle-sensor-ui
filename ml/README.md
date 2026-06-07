# ML Training Pipeline

Trains muscle fatigue classifiers on the Zenodo EMG Fatigue Dataset.

## Dataset

**EMG Fatigue Dataset** (Zenodo, doi:10.5281/zenodo.5189275)
- 16 healthy subjects, 200 Hz sampling, 8 EMG channels
- 120 seconds per subject: 0-30s Fresh, 30-90s Transition, 90-120s Fatigued
- ~3,500 sliding windows extracted

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train all models (Random Forest, SVM, Logistic Regression)
python train_model.py

# Train + export TF.js model for browser
python train_model.py --export
```

## Output Files

| File | Description |
|------|-------------|
| `emg_fatigue_model.pkl` | Best scikit-learn model (Random Forest) |
| `emg_scaler.pkl` | StandardScaler parameters |
| `../public/models/emg_fatigue_model/` | TF.js Layers model (with --export) |
| `../public/models/scaler_params.json` | Normalization parameters for JS |
| `../public/models/evaluation_results.json` | Metrics summary |

## Features Extracted (per 2s window)

1. **RMS** — Root Mean Square amplitude
2. **MAV** — Mean Absolute Value
3. **ZCR** — Zero Crossing Rate
4. **MDF** — Median Frequency
5. **MNF** — Mean Frequency
6. **Power** — Total spectral power (10-99 Hz)
7. **SM1** — First spectral moment
8. **SM2** — Second spectral moment

## Results

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Random Forest | 69.3% | 0.790 |
| SVM (RBF) | 67.6% | 0.753 |
| Logistic Regression | 65.6% | 0.726 |

LOSO cross-validation mean accuracy: 54.8% (+/- 8.9%)
