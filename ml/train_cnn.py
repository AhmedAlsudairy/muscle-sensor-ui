"""Train 1D-CNN on raw EMG windows and save model + update evaluation JSON."""
import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import json
import glob
from scipy.signal import butter, sosfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score,
)
import keras

FS = 200
WINDOW_SIZE = 400
STRIDE = 100
FRESH_END = 30
FATIGUE_START = 90
LOW_CUTOFF = 10
HIGH_CUTOFF = 99

# Load raw windows
subject_files = sorted(glob.glob("emg_fatigue_data/**/*.txt", recursive=True))
X_raw, y_raw = [], []

for fp in subject_files:
    try:
        with open(fp) as f:
            lines = f.readlines()
        start_row = 0
        for i, line in enumerate(lines):
            try:
                float(line.strip().split()[0])
                start_row = i
                break
            except (ValueError, IndexError):
                continue
        data = np.loadtxt(fp, skiprows=start_row)
    except Exception:
        continue

    emg = data[:, 0].astype(np.float32)
    nyq = FS / 2
    hi = min(HIGH_CUTOFF, nyq - 1)
    lo = max(LOW_CUTOFF, 1)
    sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
    emg_f = sosfilt(sos, emg)

    for i in range(0, len(emg_f) - WINDOW_SIZE, STRIDE):
        w = emg_f[i : i + WINDOW_SIZE]
        if np.std(w) < 0.005:
            continue
        t = i / FS
        if t < FRESH_END:
            label = 0
        elif t >= FATIGUE_START:
            label = 1
        else:
            continue
        X_raw.append(w)
        y_raw.append(label)

X = np.array(X_raw, dtype=np.float32)
y = np.array(y_raw, dtype=int)

for i in range(len(X)):
    m = np.max(np.abs(X[i]))
    if m > 0:
        X[i] = X[i] / m

X = X.reshape(-1, 400, 1)
print(f"Raw windows: {len(X)} (Fresh={sum(y == 0)}, Fatigued={sum(y == 1)})")

# 75/25 split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

# Class weights
n_fresh = np.sum(y_tr == 0)
n_fat = np.sum(y_tr == 1)
tot = n_fresh + n_fat
cw = {0: tot / (2 * max(n_fresh, 1)), 1: tot / (2 * max(n_fat, 1))}

# Build 1D-CNN
model = keras.Sequential([
    keras.layers.Input((400, 1)),
    keras.layers.Conv1D(16, 5, activation="relu", padding="same",
                        kernel_regularizer=keras.regularizers.L2(0.001)),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(32, 5, activation="relu", padding="same",
                        kernel_regularizer=keras.regularizers.L2(0.001)),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(64, 3, activation="relu", padding="same",
                        kernel_regularizer=keras.regularizers.L2(0.001)),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation="relu",
                       kernel_regularizer=keras.regularizers.L2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# Train
model.fit(
    X_tr, y_tr,
    epochs=100,
    batch_size=32,
    validation_data=(X_te, y_te),
    class_weight=cw,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ],
)

# Evaluate
loss, acc = model.evaluate(X_te, y_te, verbose=0)
yp = model.predict(X_te, verbose=0)
yp_c = np.argmax(yp, axis=1)

auc = roc_auc_score(y_te, yp[:, 1])
p1 = precision_score(y_te, yp_c, pos_label=1)
r1 = recall_score(y_te, yp_c, pos_label=1)
f1 = f1_score(y_te, yp_c, pos_label=1)
p0 = precision_score(y_te, yp_c, pos_label=0)
r0 = recall_score(y_te, yp_c, pos_label=0)
f0 = f1_score(y_te, yp_c, pos_label=0)

print(f"\nCNN Test Accuracy: {acc:.4f} ({acc * 100:.1f}%)")
print(f"CNN AUC-ROC: {auc:.4f}")
print(f"P(F)={p1:.4f} R(F)={r1:.4f} F1(F)={f1:.4f}")
print(classification_report(y_te, yp_c, target_names=["Fresh", "Fatigued"], digits=3))

# Save model
model.save("emg_cnn_model.h5")
model.save("emg_cnn_model.keras")
print(f"Saved: emg_cnn_model.h5 ({os.path.getsize('emg_cnn_model.h5') / 1024:.0f} KB)")
print(f"Saved: emg_cnn_model.keras ({os.path.getsize('emg_cnn_model.keras') / 1024:.0f} KB)")

# Update evaluation_results.json
eval_path = "../public/models/evaluation_results.json"
with open(eval_path) as f:
    ev = json.load(f)
ev["cnn_model"]["accuracy"] = round(float(acc), 4)
ev["cnn_model"]["auc"] = round(float(auc), 4)
ev["cnn_model"]["precision_fresh"] = round(float(p0), 4)
ev["cnn_model"]["recall_fresh"] = round(float(r0), 4)
ev["cnn_model"]["f1_fresh"] = round(float(f0), 4)
ev["cnn_model"]["precision_fatigued"] = round(float(p1), 4)
ev["cnn_model"]["recall_fatigued"] = round(float(r1), 4)
ev["cnn_model"]["f1_fatigued"] = round(float(f1), 4)
with open(eval_path, "w") as f:
    json.dump(ev, f, indent=2)
print(f"Updated {eval_path}")
print(f"  Acc={acc:.4f}  AUC={auc:.4f}  P(F)={p1:.4f}  R(F)={r1:.4f}  F1(F)={f1:.4f}")
