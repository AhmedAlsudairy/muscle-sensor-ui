# EMG Fatigue Datasets

## Source

**EMG Fatigue Dataset — Zenodo**
- DOI: [10.5281/zenodo.5189275](https://doi.org/10.5281/zenodo.5189275)
- Download: `ml/train_model.py` auto-downloads it
- 16 subjects × 120 seconds × 200 Hz × 8 EMG channels
- Labels: Fresh (0–30s), Transition (30–90s), Fatigued (90–120s)

## Sample Data

The `sample_features.json` file contains 20 example feature windows extracted from Subject 1 of the dataset. Each row has 8 features: RMS, MAV, ZCR, MDF, MNF, Total Power, SM1, SM2.

## Data Format

After extraction (`ml/train_model.py`), each 2-second window produces:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | RMS | Root Mean Square (amplitude) |
| 1 | MAV | Mean Absolute Value |
| 2 | ZCR | Zero Crossing Rate |
| 3 | MDF | Median Frequency (Hz) |
| 4 | MNF | Mean Frequency (Hz) |
| 5 | Power | Total Spectral Power (10–99 Hz) |
| 6 | SM1 | First Spectral Moment |
| 7 | SM2 | Second Spectral Moment |

## Citation

If using this dataset in research, cite:
> Zenodo. "Dataset EMG Fatigue." https://doi.org/10.5281/zenodo.5189275
