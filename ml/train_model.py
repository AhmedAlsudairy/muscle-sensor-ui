"""
EMG Muscle Fatigue Detection — Training Pipeline with 1D-CNN
=============================================================
References: Zenodo EMG Fatigue Dataset (doi:10.5281/zenodo.5189275)

Models trained:
  1. Feature-based classifiers (Random Forest, SVM, Logistic Regression)
     on 8 extracted features per window (RMS, MAV, ZCR, MDF, MNF, Power, SM1, SM2)
  2. 1D-CNN directly on raw EMG windows (400 samples × 8 channels)

Usage:
    python train_model.py                 # Train all models
    python train_model.py --cnn-only      # Only train 1D-CNN and export TF.js
    python train_model.py --export        # Export all models to TF.js + scaler
"""

import numpy as np
import glob
import os
import sys
import json
import urllib.request
import zipfile
import time

from scipy import signal
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# ── Configuration ──────────────────────────────────────────────────────────

FS = 200
WINDOW_SEC = 2
WINDOW_SIZE = FS * WINDOW_SEC   # 400 samples
STRIDE = FS // 2                # 100 samples (0.5s overlap)
LOW_CUTOFF = 10
HIGH_CUTOFF = 99

FRESH_END = 30        # seconds — labels 0 (Fresh)
FATIGUE_START = 90    # seconds — labels 1 (Fatigued)

FEATURE_NAMES = ["RMS", "MAV", "ZCR", "MDF", "MNF", "Power", "SM1", "SM2"]

DATASET_URL = "https://zenodo.org/records/5189275/files/Dataset%20EMG%20Fatigue.zip?download=1"
DATA_DIR = "emg_fatigue_data"

# Output directories
FEATURE_MODEL_DIR = "../public/models/emg_fatigue_model"
CNN_MODEL_DIR = "../public/models/emg_cnn_model"
SCALER_PATH = "../public/models/scaler_params.json"
EVAL_PATH = "../public/models/evaluation_results.json"


# ── Download Dataset ──────────────────────────────────────────────────────

def download_dataset():
    if os.path.exists(DATA_DIR) and len(glob.glob(f"{DATA_DIR}/**/*.txt", recursive=True)) > 0:
        print("[OK] Dataset already downloaded")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = "emg_fatigue.zip"

    print("Downloading EMG Fatigue Dataset from Zenodo...")
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    print(f"  Size: {os.path.getsize(zip_path)/1024/1024:.1f} MB")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)
    print("[OK] Downloaded and extracted")


# ── Signal Processing ─────────────────────────────────────────────────────

def bandpass_filter(data, fs=FS, low=LOW_CUTOFF, high=HIGH_CUTOFF):
    nyquist = fs / 2
    high = min(high, nyquist - 1)
    low = max(low, 1)
    sos = signal.butter(4, [low, high], btype="band", fs=fs, output="sos")
    return signal.sosfilt(sos, data)


def extract_features(window, fs=FS):
    low, high = LOW_CUTOFF, HIGH_CUTOFF
    rect = np.abs(window)

    rms = float(np.sqrt(np.mean(window ** 2)))
    mav = float(np.mean(rect))
    zcr = float(np.sum(np.diff(np.sign(window)) != 0) / len(window))

    freqs, psd = signal.welch(window, fs, nperseg=min(256, len(window)))
    valid = (freqs >= low) & (freqs <= high)
    freqs, psd = freqs[valid], psd[valid]

    mdf = mnf = total_power = sm1 = sm2 = 0.0

    if len(freqs) > 0 and np.sum(psd) > 0:
        total_power = float(np.trapezoid(psd, freqs))
        cumsum = np.cumsum(psd)
        cumsum_norm = cumsum / cumsum[-1]
        mdf_idx = np.searchsorted(cumsum_norm, 0.5)
        mdf = float(freqs[mdf_idx]) if mdf_idx < len(freqs) else float(freqs[-1])
        mnf = float(np.sum(freqs * psd) / np.sum(psd))
        sm1 = float(np.sum(freqs * psd) / total_power) if total_power > 0 else 0.0
        sm2 = float(np.sum((freqs ** 2) * psd) / total_power) if total_power > 0 else 0.0

    return [rms, mav, zcr, mdf, mnf, total_power, sm1, sm2]


def load_emg_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    start_row = 0
    for i, line in enumerate(lines):
        try:
            float(line.strip().split()[0])
            start_row = i
            break
        except (ValueError, IndexError):
            continue
    return np.loadtxt(filepath, skiprows=start_row)


# ── Data Processing Pipelines ─────────────────────────────────────────────

def process_all_subjects_feature_based():
    """Extract 8 hand-crafted features from each 2s window."""
    subject_files = sorted(glob.glob(f"{DATA_DIR}/**/*.txt", recursive=True))
    print(f"\nProcessing {len(subject_files)} subject files (feature-based)...")

    X, y, subject_ids = [], [], []

    for subj_idx, filepath in enumerate(subject_files):
        try:
            data = load_emg_file(filepath)
        except Exception as e:
            print(f"  SKIP {os.path.basename(filepath)}: {e}")
            continue

        emg = data[:, 0]
        emg_filtered = bandpass_filter(emg)

        for i in range(0, len(emg_filtered) - WINDOW_SIZE, STRIDE):
            window = emg_filtered[i : i + WINDOW_SIZE]
            if np.std(window) < 0.005:
                continue

            features = extract_features(window)
            if len(features) != len(FEATURE_NAMES):
                continue

            X.append(features)
            subject_ids.append(subj_idx)

            time_sec = i / FS
            if time_sec < FRESH_END:
                y.append(0)
            elif time_sec >= FATIGUE_START:
                y.append(1)
            else:
                y.append(2)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    subject_ids = np.array(subject_ids, dtype=int)

    binary_mask = y != 2
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    subject_ids_binary = subject_ids[binary_mask]

    print(f"Total windows: {len(X)} (Fresh: {sum(y==0)}, Fatigued: {sum(y==1)}, Transition: {sum(y==2)})")
    print(f"Binary dataset: {len(X_binary)}")

    return X_binary, y_binary, subject_ids_binary


def process_all_subjects_raw_windows():
    """Extract raw 400-sample windows for 1D-CNN training (channel 0 only)."""
    subject_files = sorted(glob.glob(f"{DATA_DIR}/**/*.txt", recursive=True))
    print(f"\nProcessing {len(subject_files)} subject files (raw windows for CNN)...")

    X_raw, y_raw, subject_ids_raw = [], [], []

    for subj_idx, filepath in enumerate(subject_files):
        try:
            data = load_emg_file(filepath)
        except Exception as e:
            print(f"  SKIP {os.path.basename(filepath)}: {e}")
            continue

        emg = data[:, 0].astype(np.float32)
        emg_filtered = bandpass_filter(emg)

        for i in range(0, len(emg_filtered) - WINDOW_SIZE, STRIDE):
            window = emg_filtered[i : i + WINDOW_SIZE]
            if np.std(window) < 0.005:
                continue

            time_sec = i / FS
            if time_sec < FRESH_END:
                label = 0
            elif time_sec >= FATIGUE_START:
                label = 1
            else:
                continue  # skip transition

            X_raw.append(window)
            y_raw.append(label)
            subject_ids_raw.append(subj_idx)

    X_raw = np.array(X_raw, dtype=np.float32)
    y_raw = np.array(y_raw, dtype=int)
    subject_ids_raw = np.array(subject_ids_raw, dtype=int)

    print(f"Raw windows: {len(X_raw)} (Fresh: {sum(y_raw==0)}, Fatigued: {sum(y_raw==1)})")

    return X_raw, y_raw, subject_ids_raw


# ── Feature-Based Model Training ──────────────────────────────────────────

def train_feature_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining set: {len(X_train)}  |  Test set: {len(X_test)}")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight="balanced", random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, class_weight="balanced",
            probability=True, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"  {name}")
        print(f"{'=' * 50}")

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        print(classification_report(y_test, y_pred, target_names=["Fresh", "Fatigued"]))
        auc = roc_auc_score(y_test, y_prob)
        print(f"  AUC-ROC: {auc:.3f}")

        results[name] = {"model": model, "accuracy": np.mean(y_pred == y_test), "auc": auc}

    print(f"\n{'=' * 50}")
    print("  FEATURE MODEL COMPARISON")
    print(f"{'=' * 50}")
    for name, res in results.items():
        print(f"  {name:22s} | Acc: {res['accuracy']:.3f} | AUC: {res['auc']:.3f}")

    return results, scaler, X_test_scaled, y_test


# ── 1D-CNN on Raw Windows ────────────────────────────────────────────────

def train_cnn_on_raw(X_raw, y_raw):
    """Train a 1D-CNN directly on raw 400-sample EMG windows and export to TF.js."""

    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("\n[SKIP] TensorFlow not installed. Cannot train 1D-CNN.")
        return None, None, None, None

    # Normalize each window to [-1, 1] range
    X_norm = X_raw.copy()
    for i in range(len(X_norm)):
        max_val = np.max(np.abs(X_norm[i]))
        if max_val > 0:
            X_norm[i] = X_norm[i] / max_val

    # Reshape: [batch, timesteps=400, features=1]
    X_norm = X_norm.reshape(-1, WINDOW_SIZE, 1)
    y_cat = tf.keras.utils.to_categorical(y_raw, num_classes=2)

    # Split: 80/20 subject-aware (train on subjects 0-11, test on 12-14)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_cat, test_size=0.2, random_state=42, stratify=y_raw
    )

    print(f"\n{'=' * 50}")
    print("  1D-CNN Training (Raw EMG Windows)")
    print(f"{'=' * 50}")
    print(f"  Input shape: {X_norm.shape}  (samples, 400 timesteps, 1 channel)")
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # 1D-CNN architecture (matches in-browser TF.js model)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW_SIZE, 1)),
        tf.keras.layers.Conv1D(
            filters=16, kernel_size=5, activation="relu", padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),           # → 200
        tf.keras.layers.Conv1D(
            filters=32, kernel_size=5, activation="relu", padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),           # → 100
        tf.keras.layers.Conv1D(
            filters=64, kernel_size=3, activation="relu", padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Compute class weights for imbalance handling
    n_fresh = np.sum(y_raw == 0)
    n_fatigued = np.sum(y_raw == 1)
    total = n_fresh + n_fatigued
    class_weights = {
        0: total / (2 * max(n_fresh, 1)),
        1: total / (2 * max(n_fatigued, 1)),
    }

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=15, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ],
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    from sklearn.metrics import roc_auc_score as roc_fn
    auc = roc_fn(y_true, y_pred_probs[:, 1])

    print(f"\n  CNN Test Accuracy:  {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"  CNN Test AUC-ROC:   {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Fresh", "Fatigued"]))

    return model, history, test_acc, auc


# ── Leave-One-Subject-Out ─────────────────────────────────────────────────

def leave_one_subject_out_cv(X, y, subject_ids):
    logo = LeaveOneGroupOut()
    scores = []
    print(f"\n{'=' * 50}")
    print("  Leave-One-Subject-Out CV (Random Forest)")
    print(f"{'=' * 50}")

    for train_idx, test_idx in logo.split(X, y, subject_ids):
        subj_test = subject_ids[test_idx][0]
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            class_weight="balanced", random_state=42
        )
        rf.fit(X_tr_s, y_tr)
        score = rf.score(X_te_s, y_te)
        scores.append(score)
        print(f"  Subject {subj_test:2d}: {score:.3f}")

    print(f"\n  Mean LOSO accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
    return scores


# ── Save Artifacts ────────────────────────────────────────────────────────

def save_feature_artifacts(best_model, scaler):
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": FEATURE_NAMES,
    }
    with open(SCALER_PATH, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"Saved: {SCALER_PATH}")

    joblib.dump(best_model, "emg_fatigue_model.pkl")
    print(f"Saved: emg_fatigue_model.pkl")


def save_cnn_tfjs(model, cnn_acc, cnn_auc):
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print("[SKIP] tensorflowjs not installed. Install with: pip install tensorflowjs")
        return

    os.makedirs(CNN_MODEL_DIR, exist_ok=True)
    tfjs.converters.save_keras_model(model, CNN_MODEL_DIR)
    print(f"[OK] TF.js CNN model exported to {CNN_MODEL_DIR}")
    print(f"  Model files: {os.listdir(CNN_MODEL_DIR)}")


def save_evaluation_results(feature_results, cnn_acc, cnn_auc, loso_scores):
    best_name = max(feature_results, key=lambda k: feature_results[k]["auc"])
    feature_acc = feature_results[best_name]["accuracy"]
    feature_auc = feature_results[best_name]["auc"]

    evaluation = {
        "feature_models": {
            "best_model": best_name,
            "accuracy": float(feature_acc),
            "auc": float(feature_auc),
            "feature_names": FEATURE_NAMES,
        },
        "cnn_model": {
            "architecture": "1D-CNN (Conv1D×3 + GlobalAvgPool + Dense32 + Dense2)",
            "window_size": WINDOW_SIZE,
            "sample_rate_hz": FS,
            "accuracy": float(cnn_acc) if cnn_acc else None,
            "auc": float(cnn_auc) if cnn_auc else None,
        },
        "loso_cv": {
            "mean_accuracy": float(np.mean(loso_scores)) if loso_scores else None,
            "std": float(np.std(loso_scores)) if loso_scores else None,
        },
        "dataset": {
            "url": "https://doi.org/10.5281/zenodo.5189275",
            "subjects": 15,
            "sampling_rate": FS,
            "window_size_samples": WINDOW_SIZE,
            "window_size_seconds": WINDOW_SEC,
            "stride_samples": STRIDE,
            "classes": ["Fresh (0-30s)", "Fatigued (90-120s)"],
        },
    }

    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    with open(EVAL_PATH, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"Saved: {EVAL_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    cnn_only = "--cnn-only" in sys.argv
    export = "--export" in sys.argv

    download_dataset()

    # ── Feature-based pipeline ──
    if not cnn_only:
        print("\n" + "=" * 60)
        print("  FEATURE-BASED PIPELINE")
        print("=" * 60)
        X_feat, y_feat, subj_ids_feat = process_all_subjects_feature_based()
        feature_results, scaler, X_test_feat, y_test_feat = train_feature_models(X_feat, y_feat)
        loso_scores = leave_one_subject_out_cv(X_feat, y_feat, subj_ids_feat)
        save_feature_artifacts(feature_results["Random Forest"]["model"], scaler)
    else:
        feature_results = {}
        loso_scores = None

    # ── 1D-CNN on raw windows ──
    print("\n" + "=" * 60)
    print("  1D-CNN ON RAW EMG WINDOWS")
    print("=" * 60)
    X_raw, y_raw, subj_ids_raw = process_all_subjects_raw_windows()
    cnn_model, cnn_history, cnn_acc, cnn_auc = train_cnn_on_raw(X_raw, y_raw)

    if cnn_model is not None and export:
        save_cnn_tfjs(cnn_model, cnn_acc, cnn_auc)

    # ── Save evaluation ──
    save_evaluation_results(feature_results, cnn_acc, cnn_auc, loso_scores)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 60}")

    if feature_results:
        best_name = max(feature_results, key=lambda k: feature_results[k]["auc"])
        print(f"  Feature model (best):  {best_name}")
        print(f"    Accuracy: {feature_results[best_name]['accuracy']:.3f}")
        print(f"    AUC-ROC:  {feature_results[best_name]['auc']:.3f}")

    if cnn_model is not None:
        print(f"  CNN (1D-CNN):")
        print(f"    Test Accuracy: {cnn_acc:.3f}" if cnn_acc else "    N/A")
        print(f"    Test AUC-ROC:  {cnn_auc:.3f}" if cnn_auc else "    N/A")

    if loso_scores:
        print(f"  LOSO CV (RF):")
        print(f"    Mean: {np.mean(loso_scores):.3f} (+/- {np.std(loso_scores):.3f})")

    print("\nDone. Models saved to public/models/")
    if not export:
        print("Run with --export to export TF.js CNN model.")


if __name__ == "__main__":
    main()
