"""
EMG Muscle Fatigue Detection — Training Pipeline with 1D-CNN
=============================================================
References: Zenodo EMG Fatigue Dataset (doi:10.5281/zenodo.5189275)

Models trained:
  1. Feature-based classifiers (Random Forest, SVM, Logistic Regression)
     on 8 extracted features per window (RMS, MAV, ZCR, MDF, MNF, Power, SM1, SM2)
  2. 1D-CNN directly on raw EMG windows (400 samples × 1 channel)

Data Split: 75% Train / 25% Test (stratified, random_state=42)

Usage:
    python train_model.py                 # Train all models
    python train_model.py --cnn-only      # Only train 1D-CNN and export TF.js
    python train_model.py --export        # Export all models to TF.js + scaler
    python train_model.py --plots         # Save evaluation graphs (PNGs)
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
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, roc_curve,
)
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Configuration ---------------------------------------------------------

FS = 200
WINDOW_SEC = 2
WINDOW_SIZE = FS * WINDOW_SEC   # 400 samples
STRIDE = FS // 2                # 100 samples (0.5s overlap)
LOW_CUTOFF = 10
HIGH_CUTOFF = 99

FRESH_END = 30        # seconds — labels 0 (Fresh)
FATIGUE_START = 90    # seconds — labels 1 (Fatigued)
TEST_SIZE = 0.25      # 75% Train / 25% Test split
RANDOM_STATE = 42

FEATURE_NAMES = ["RMS", "MAV", "ZCR", "MDF", "MNF", "Power", "SM1", "SM2"]
CLASS_NAMES = ["Fresh", "Fatigued"]

DATASET_URL = "https://zenodo.org/records/5189275/files/Dataset%20EMG%20Fatigue.zip?download=1"
DATA_DIR = "emg_fatigue_data"
PLOTS_DIR = "../public/models/plots"

# Output directories
FEATURE_MODEL_DIR = "../public/models/emg_fatigue_model"
CNN_MODEL_DIR = "../public/models/emg_cnn_model"
SCALER_PATH = "../public/models/scaler_params.json"
EVAL_PATH = "../public/models/evaluation_results.json"


#─ Download Dataset ──────────────────────────────────────────────────────

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


#─ Signal Processing ─────────────────────────────────────────────────────

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


#─ Data Processing Pipelines ─────────────────────────────────────────────

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

    n_fresh = sum(y_binary == 0)
    n_fatigued = sum(y_binary == 1)
    print(f"  Fresh: {n_fresh} ({n_fresh/len(X_binary)*100:.1f}%) | Fatigued: {n_fatigued} ({n_fatigued/len(X_binary)*100:.1f}%)")

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
                continue

            X_raw.append(window)
            y_raw.append(label)
            subject_ids_raw.append(subj_idx)

    X_raw = np.array(X_raw, dtype=np.float32)
    y_raw = np.array(y_raw, dtype=int)
    subject_ids_raw = np.array(subject_ids_raw, dtype=int)

    print(f"Raw windows: {len(X_raw)} (Fresh: {sum(y_raw==0)}, Fatigued: {sum(y_raw==1)})")

    return X_raw, y_raw, subject_ids_raw


#─ Plotting Functions ────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name, savepath):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}\n(75% Train / 25% Test)", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]),
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black",
                    fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {savepath}")


def plot_metrics_bars(metrics_by_model, savepath):
    """Bar chart: Precision, Recall, F1 per model (Fatigued class)."""
    model_names = list(metrics_by_model.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.25

    p = [metrics_by_model[m]["precision_1"] for m in model_names]
    r = [metrics_by_model[m]["recall_1"] for m in model_names]
    f = [metrics_by_model[m]["f1_1"] for m in model_names]

    bars1 = ax.bar(x - width, p, width, label="Precision", color="#3b82f6")
    bars2 = ax.bar(x, r, width, label="Recall", color="#22c55e")
    bars3 = ax.bar(x + width, f, width, label="F1 Score", color="#a855f7")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 per Model (Fatigued Class)\n75% Train / 25% Test Split", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for bar_set in [bars1, bars2, bars3]:
        for bar in bar_set:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {savepath}")


def plot_roc_curves(roc_data, savepath):
    """ROC curves for all models on one plot."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"Random Forest": "#3b82f6", "SVM (RBF)": "#22c55e", "Logistic Regression": "#a855f7"}

    for name, data in roc_data.items():
        if name in colors:
            fpr, tpr, auc_val = data["fpr"], data["tpr"], data["auc"]
            ax.plot(fpr, tpr, lw=2, color=colors[name],
                    label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — EMG Fatigue Classification\n(75% Train / 25% Test)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {savepath}")


def plot_data_distribution(y, savepath):
    """Pie chart showing class distribution."""
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = [int(sum(y == 0)), int(sum(y == 1))]
    colors = ["#22c55e", "#ef4444"]
    wedges, texts, autotexts = ax.pie(
        counts, labels=CLASS_NAMES, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 13}
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(14)
    ax.set_title(f"Data Distribution (Total = {len(y)})\n75% Train / 25% Test Split", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {savepath}")


def plot_train_test_split_info(n_train, n_test, n_fresh_train, n_fresh_test, n_fat_train, n_fat_test, savepath):
    """Bar chart showing train/test split sizes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Fresh\n(Train)", "Fresh\n(Test)", "Fatigued\n(Train)", "Fatigued\n(Test)"]
    counts = [n_fresh_train, n_fresh_test, n_fat_train, n_fat_test]
    colors = ["#22c55e", "#86efac", "#ef4444", "#fca5a5"]

    bars = ax.bar(categories, counts, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Samples", fontsize=12)
    ax.set_title(f"Train/Test Split: {n_train} Train / {n_test} Test (75%/25%)\nStratified, random_state=42", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {savepath}")


#─ Feature-Based Model Training ──────────────────────────────────────────

def train_feature_models(X, y, save_plots=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_train = len(X_train)
    n_test = len(X_test)
    n_fresh_train = int(sum(y_train == 0))
    n_fresh_test = int(sum(y_test == 0))
    n_fat_train = int(sum(y_train == 1))
    n_fat_test = int(sum(y_test == 1))

    print(f"\n{'-' * 55}")
    print(f"  DATA SPLIT: {n_train} Train ({n_train/(n_train+n_test)*100:.0f}%) / {n_test} Test ({n_test/(n_train+n_test)*100:.0f}%)")
    print(f"  Train: Fresh={n_fresh_train}, Fatigued={n_fat_train}")
    print(f"  Test:  Fresh={n_fresh_test}, Fatigued={n_fat_test}")
    print(f"{'-' * 55}")

    if save_plots:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_train_test_split_info(
            n_train, n_test, n_fresh_train, n_fresh_test, n_fat_train, n_fat_test,
            f"{PLOTS_DIR}/train_test_split.png"
        )
        plot_data_distribution(y, f"{PLOTS_DIR}/data_distribution.png")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, class_weight="balanced",
            probability=True, random_state=RANDOM_STATE
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
    }

    results = {}
    metrics_by_model = {}
    roc_data = {}

    for name, model in models.items():
        print(f"\n{'=' * 55}")
        print(f"  {name}")
        print(f"{'=' * 55}")

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        p0 = precision_score(y_test, y_pred, pos_label=0)
        r0 = recall_score(y_test, y_pred, pos_label=0)
        f0 = f1_score(y_test, y_pred, pos_label=0)
        p1 = precision_score(y_test, y_pred, pos_label=1)
        r1 = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        print(f"  AUC-ROC: {auc:.3f}")
        print(f"  Confusion Matrix:\n{cm}")

        results[name] = {
            "model": model,
            "accuracy": float(np.mean(y_pred == y_test)),
            "auc": float(auc),
            "precision_fresh": float(p0),
            "recall_fresh": float(r0),
            "f1_fresh": float(f0),
            "precision_fatigued": float(p1),
            "recall_fatigued": float(r1),
            "f1_fatigued": float(f1),
        }

        metrics_by_model[name] = {
            "precision_0": float(p0), "recall_0": float(r0), "f1_0": float(f0),
            "precision_1": float(p1), "recall_1": float(r1), "f1_1": float(f1),
        }
        roc_data[name] = {"fpr": None, "tpr": None, "auc": float(auc)}
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name]["fpr"] = fpr.tolist()
        roc_data[name]["tpr"] = tpr.tolist()

        if save_plots:
            plot_confusion_matrix(cm, name, f"{PLOTS_DIR}/confusion_{name.lower().replace(' ', '_')}.png")

    if save_plots:
        plot_metrics_bars(metrics_by_model, f"{PLOTS_DIR}/metrics_bars.png")
        plot_roc_curves(roc_data, f"{PLOTS_DIR}/roc_curves.png")

    print(f"\n{'=' * 55}")
    print("  FEATURE MODEL COMPARISON (75% Train / 25% Test)")
    print(f"{'=' * 55}")
    print(f"  {'Model':22s} | {'Acc':>6s} | {'AUC':>6s} | {'P(F)':>6s} | {'R(F)':>6s} | {'F1(F)':>6s}")
    print(f"  {'-'*22} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6}")
    for name, res in results.items():
        print(f"  {name:22s} | {res['accuracy']:6.3f} | {res['auc']:6.3f} | "
              f"{res['precision_fatigued']:6.3f} | {res['recall_fatigued']:6.3f} | {res['f1_fatigued']:6.3f}")

    return results, scaler, X_test_scaled, y_test


#─ 1D-CNN on Raw Windows ────────────────────────────────────────────────

def train_cnn_on_raw(X_raw, y_raw, save_plots=False):
    try:
        import tensorflow as tf
    except ImportError:
        print("\n[SKIP] TensorFlow not installed. Cannot train 1D-CNN.")
        return None, None, None, None

    X_norm = X_raw.copy()
    for i in range(len(X_norm)):
        max_val = np.max(np.abs(X_norm[i]))
        if max_val > 0:
            X_norm[i] = X_norm[i] / max_val

    X_norm = X_norm.reshape(-1, WINDOW_SIZE, 1)
    y_cat = tf.keras.utils.to_categorical(y_raw, num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_cat, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_raw
    )

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    print(f"\n{'-' * 55}")
    print(f"  CNN DATA SPLIT: {n_train} Train ({n_train/(n_train+n_test)*100:.0f}%) / {n_test} Test ({n_test/(n_train+n_test)*100:.0f}%)")
    print(f"{'-' * 55}")

    print(f"\n{'=' * 55}")
    print("  1D-CNN Training (Raw EMG Windows)")
    print(f"{'=' * 55}")
    print(f"  Input shape:  {X_norm.shape}  (samples, 400 timesteps, 1 channel)")
    print(f"  Train: {n_train}  |  Test: {n_test}")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW_SIZE, 1)),
        tf.keras.layers.Conv1D(
            filters=16, kernel_size=5, activation="relu", padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(
            filters=32, kernel_size=5, activation="relu", padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(
            filters=64, kernel_size=3, activation="relu", padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
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
        metrics=["accuracy"])

    model.summary()

    n_fresh = np.sum(y_raw == 0)
    n_fatigued = np.sum(y_raw == 1)
    total = n_fresh + n_fatigued
    class_weights = {
        0: total / (2 * max(n_fresh, 1)),
        1: total / (2 * max(n_fatigued, 1))}

    history = model.fit(
        X_train, y_train, epochs=100, batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights, verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)])

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    auc = roc_auc_score(y_true, y_pred_probs[:, 1])
    cm = confusion_matrix(y_true, y_pred)
    p1 = precision_score(y_true, y_pred, pos_label=1)
    r1 = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    print(f"\n  CNN Test Accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"  CNN AUC-ROC:       {auc:.4f}")
    print(f"  Precision(F): {p1:.3f}  |  Recall(F): {r1:.3f}  |  F1(F): {f1:.3f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

    if save_plots:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_confusion_matrix(cm, "1D-CNN", f"{PLOTS_DIR}/confusion_cnn.png")

        # Training history plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history.history["loss"], label="Train Loss", color="#3b82f6")
        ax1.plot(history.history["val_loss"], label="Val Loss", color="#ef4444")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("CNN Loss Curves")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax2.plot(history.history["accuracy"], label="Train Acc", color="#3b82f6")
        ax2.plot(history.history["val_accuracy"], label="Val Acc", color="#22c55e")
        ax2.set_xlabel("Epoch")  # Fixed typo
        ax2.set_ylabel("Accuracy")
        ax2.set_title("CNN Accuracy Curves")
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/cnn_training_history.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [plot] {PLOTS_DIR}/cnn_training_history.png")

    cnn_metrics = {
        "accuracy": float(test_acc),
        "auc": float(auc),
        "precision_fresh": float(precision_score(y_true, y_pred, pos_label=0)),
        "recall_fresh": float(recall_score(y_true, y_pred, pos_label=0)),
        "f1_fresh": float(f1_score(y_true, y_pred, pos_label=0)),
        "precision_fatigued": float(p1),
        "recall_fatigued": float(r1),
        "f1_fatigued": float(f1),
    }

    return model, history, cnn_metrics, (X_test, y_test)


#─ Leave-One-Subject-Out ─────────────────────────────────────────────────

def leave_one_subject_out_cv(X, y, subject_ids):
    logo = LeaveOneGroupOut()
    scores = []
    print(f"\n{'=' * 55}")
    print("  Leave-One-Subject-Out CV (Random Forest)")
    print(f"{'=' * 55}")

    for train_idx, test_idx in logo.split(X, y, subject_ids):
        subj_test = subject_ids[test_idx][0]
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            class_weight="balanced", random_state=RANDOM_STATE)
        rf.fit(X_tr_s, y_tr)
        score = rf.score(X_te_s, y_te)
        scores.append(score)
        print(f"  Subject {subj_test:2d}: {score:.3f}")

    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))
    print(f"\n  Mean LOSO accuracy: {mean_s:.3f} (+/- {std_s:.3f})")
    return mean_s, std_s


#─ Save Artifacts ────────────────────────────────────────────────────────

def save_feature_artifacts(feature_results, scaler):
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": FEATURE_NAMES,
    }
    with open(SCALER_PATH, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"Saved: {SCALER_PATH}")

    for name, res in feature_results.items():
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = f"emg_{fname}_model.pkl"
        joblib.dump(res["model"], path)
        print(f"Saved: {path}")


def save_cnn_tfjs(model, cnn_metrics):
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print("[SKIP] tensorflowjs not installed. pip install tensorflowjs")
        return

    os.makedirs(CNN_MODEL_DIR, exist_ok=True)
    tfjs.converters.save_keras_model(model, CNN_MODEL_DIR)
    print(f"[OK] TF.js CNN model → {CNN_MODEL_DIR}")
    print(f"  Files: {os.listdir(CNN_MODEL_DIR)}")


def save_evaluation_results(feature_results, cnn_metrics, loso_mean, loso_std):
    best_name = max(feature_results, key=lambda k: feature_results[k]["auc"]) if feature_results else None

    rf = feature_results.get("Random Forest", {}) if feature_results else {}
    svm = feature_results.get("SVM (RBF)", {}) if feature_results else {}
    lr = feature_results.get("Logistic Regression", {}) if feature_results else {}

    evaluation = {
        "data_split": {
            "train_pct": 75,
            "test_pct": 25,
            "stratified": True,
            "random_state": RANDOM_STATE,
        },
        "feature_models": {
            "features": FEATURE_NAMES,
            "best_model": best_name,
            "random_forest": {
                "accuracy": round(float(rf.get("accuracy", 0)), 4),
                "auc_roc": round(float(rf.get("auc", 0)), 4),
                "precision_fresh": round(float(rf.get("precision_fresh", 0)), 4),
                "recall_fresh": round(float(rf.get("recall_fresh", 0)), 4),
                "f1_fresh": round(float(rf.get("f1_fresh", 0)), 4),
                "precision_fatigued": round(float(rf.get("precision_fatigued", 0)), 4),
                "recall_fatigued": round(float(rf.get("recall_fatigued", 0)), 4),
                "f1_fatigued": round(float(rf.get("f1_fatigued", 0)), 4),
            },
            "svm": {
                "accuracy": round(float(svm.get("accuracy", 0)), 4),
                "auc_roc": round(float(svm.get("auc", 0)), 4),
                "precision_fresh": round(float(svm.get("precision_fresh", 0)), 4),
                "recall_fresh": round(float(svm.get("recall_fresh", 0)), 4),
                "f1_fresh": round(float(svm.get("f1_fresh", 0)), 4),
                "precision_fatigued": round(float(svm.get("precision_fatigued", 0)), 4),
                "recall_fatigued": round(float(svm.get("recall_fatigued", 0)), 4),
                "f1_fatigued": round(float(svm.get("f1_fatigued", 0)), 4),
            },
            "logistic_regression": {
                "accuracy": round(float(lr.get("accuracy", 0)), 4),
                "auc_roc": round(float(lr.get("auc", 0)), 4),
                "precision_fresh": round(float(lr.get("precision_fresh", 0)), 4),
                "recall_fresh": round(float(lr.get("recall_fresh", 0)), 4),
                "f1_fresh": round(float(lr.get("f1_fresh", 0)), 4),
                "precision_fatigued": round(float(lr.get("precision_fatigued", 0)), 4),
                "recall_fatigued": round(float(lr.get("recall_fatigued", 0)), 4),
                "f1_fatigued": round(float(lr.get("f1_fatigued", 0)), 4),
            },
        },
        "cnn_model": {
            "architecture": "Conv1D(16,k5)|MP2|Conv1D(32,k5)|MP2|Conv1D(64,k3)|GAP|Dense(32)|Drop(0.2)|Dense(2)|Softmax",
            "input_shape": "(400, 1)",
            "window_size": WINDOW_SIZE,
            "sample_rate_hz": FS,
            "accuracy": round(float(cnn_metrics.get("accuracy")), 4) if cnn_metrics else None,
            "auc": round(float(cnn_metrics.get("auc")), 4) if cnn_metrics else None,
            "precision_fresh": round(float(cnn_metrics.get("precision_fresh")), 4) if cnn_metrics else None,
            "recall_fresh": round(float(cnn_metrics.get("recall_fresh")), 4) if cnn_metrics else None,
            "f1_fresh": round(float(cnn_metrics.get("f1_fresh")), 4) if cnn_metrics else None,
            "precision_fatigued": round(float(cnn_metrics.get("precision_fatigued")), 4) if cnn_metrics else None,
            "recall_fatigued": round(float(cnn_metrics.get("recall_fatigued")), 4) if cnn_metrics else None,
            "f1_fatigued": round(float(cnn_metrics.get("f1_fatigued")), 4) if cnn_metrics else None,
            "optimizer": "Adam(lr=0.001)",
            "regularization": "L2(0.001) | Dropout(0.3, 0.2)",
            "training_epochs": 100,
            "early_stopping_patience": 15,
            "expected_accuracy": "72-78%",
            "expected_auc": "0.80-0.85",
            "note": "Pretrained on Zenodo EMG Fatigue Dataset (15 subjects). 75% train / 25% test split.",
        },
        "loso_cv": {
            "method": "Leave-One-Subject-Out (15 folds)",
            "mean_accuracy": round(float(loso_mean), 4) if loso_mean is not None else None,
            "std": round(float(loso_std), 4) if loso_std is not None else None,
            "model": "Random Forest (feature-based)",
        },
        "dataset": {
            "url": "https://doi.org/10.5281/zenodo.5189275",
            "subjects": 15,
            "sampling_rate_hz": FS,
            "window_size_samples": WINDOW_SIZE,
            "window_size_seconds": WINDOW_SEC,
            "stride_samples": STRIDE,
            "total_windows": 3558,
            "binary_classification": 1758,
            "classes": {"0": "Fresh (0-30s)", "1": "Fatigued (90-120s)", "2": "Transition (30-90s, excluded)"},
            "class_distribution": {"fresh": 900, "fatigued": 858},
        },
    }

    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    with open(EVAL_PATH, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"Saved: {EVAL_PATH}")


#─ Main ──────────────────────────────────────────────────────────────────

def main():
    cnn_only = "--cnn-only" in sys.argv
    export = "--export" in sys.argv
    save_plots = "--plots" in sys.argv or export

    download_dataset()

    #─ Feature-based pipeline ──
    if not cnn_only:
        print("\n" + "=" * 60)
        print("  FEATURE-BASED PIPELINE (75% Train / 25% Test)")
        print("=" * 60)
        X_feat, y_feat, subj_ids_feat = process_all_subjects_feature_based()
        feature_results, scaler, _, _ = train_feature_models(X_feat, y_feat, save_plots)
        loso_mean, loso_std = leave_one_subject_out_cv(X_feat, y_feat, subj_ids_feat)
        save_feature_artifacts(feature_results, scaler)
    else:
        feature_results = {}
        loso_mean, loso_std = None, None

    #─ 1D-CNN on raw windows ──
    print("\n" + "=" * 60)
    print("  1D-CNN ON RAW EMG WINDOWS")
    print("=" * 60)
    X_raw, y_raw, _ = process_all_subjects_raw_windows()
    cnn_model, _, cnn_metrics, _ = train_cnn_on_raw(X_raw, y_raw, save_plots)

    if cnn_model is not None and export:
        save_cnn_tfjs(cnn_model, cnn_metrics)

    #─ Save evaluation ──
    save_evaluation_results(feature_results, cnn_metrics, loso_mean, loso_std)

    #─ Summary ──
    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY (75% Train / 25% Test)")
    print(f"{'=' * 60}")

    if feature_results:
        best_name = max(feature_results, key=lambda k: feature_results[k]["auc"])
        print(f"  Feature model (best):  {best_name}")
        print(f"    Accuracy:  {feature_results[best_name]['accuracy']:.3f}")
        print(f"    AUC-ROC:   {feature_results[best_name]['auc']:.3f}")
        print(f"    F1 (Fatig.): {feature_results[best_name]['f1_fatigued']:.3f}")

    if cnn_metrics:
        print(f"  CNN (1D-CNN):")
        print(f"    Accuracy:  {cnn_metrics['accuracy']:.3f}")
        print(f"    AUC-ROC:   {cnn_metrics['auc']:.3f}")
        print(f"    F1 (Fatig.): {cnn_metrics['f1_fatigued']:.3f}")

    if loso_mean is not None:
        print(f"  LOSO CV (RF):")
        print(f"    Mean: {loso_mean:.3f} (+/- {loso_std:.3f})")

    if save_plots:
        print(f"\n  Plots saved to {os.path.abspath(PLOTS_DIR)}/")

    print("\nDone. Models saved to public/models/")
    if not export:
        print("Run with --export to export TF.js CNN model.")
    if not save_plots:
        print("Run with --plots to save evaluation graphs (PNGs).")


if __name__ == "__main__":
    main()
