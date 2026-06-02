"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Brain, Play, Square, RefreshCw, Zap, FlaskConical, Target } from "lucide-react";
import {
  buildDataset,
  shuffleIndices,
  WINDOW_SIZE,
  CLASS_LABELS,
  computeMetrics,
} from "@/lib/neural-net";

interface Reading {
  signal_percentage: number;
  status: string;
}

interface AIPanelProps {
  readings: Reading[];
}

type Mode = "idle" | "training" | "ready";

const MIN_SAMPLES = WINDOW_SIZE + 30;
const TOTAL_EPOCHS = 50;

// Module-level tf reference — populated once after first dynamic import
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _tf: any = null;

export function AIPanel({ readings }: AIPanelProps) {
  const [mode, setMode] = useState<Mode>("idle");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const modelRef = useRef<any>(null);
  const [epoch, setEpoch] = useState(0);
  const [valAcc, setValAcc] = useState<number | null>(null);
  const [f1Score, setF1Score] = useState<number | null>(null);
  const [trainLoss, setTrainLoss] = useState<number | null>(null);
  const [probs, setProbs] = useState<[number, number] | null>(null);
  const stopRef = useRef(false);

  // Live inference whenever readings update (ready mode only)
  useEffect(() => {
    if (mode !== "ready" || !modelRef.current || !_tf || readings.length < WINDOW_SIZE) return;
    const win = readings
      .slice(-WINDOW_SIZE)
      .map((r) => parseFloat(String(r.signal_percentage)) / 100);
    _tf.tidy(() => {
      // CNN input shape: [batch=1, timesteps=20, features=1]
      const input = _tf.tensor3d([win.map((v: number) => [v])]);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const pred = modelRef.current!.predict(input) as any;
      const data: Float32Array = pred.dataSync();
      setProbs([data[0], data[1]]);
    });
  }, [readings, mode]);

  const startTraining = useCallback(async () => {
    if (readings.length < MIN_SAMPLES) return;
    setMode("training");
    setEpoch(0);
    stopRef.current = false;

    // Dynamic import — tfjs stays out of the server bundle
    if (!_tf) _tf = await import("@tensorflow/tfjs");
    const tf = _tf;

    const { X, y } = buildDataset(readings);
    if (X.length < 20) { setMode("idle"); return; }

    // 80 / 20 train / validation split
    const idx = shuffleIndices(X.length);
    const split = Math.floor(idx.length * 0.8);
    const Xtr  = idx.slice(0, split).map((i) => X[i]);
    const ytr  = idx.slice(0, split).map((i) => y[i]);
    const Xval = idx.slice(split).map((i) => X[i]);
    const yval = idx.slice(split).map((i) => y[i]);

    // Tensors — 1D CNN expects [batch, timesteps, features]
    const xTrain = tf.tensor3d(Xtr.map((row: number[]) => row.map((v: number) => [v])));
    const yTrain = tf.oneHot(tf.tensor1d(ytr, "int32"), 2).toFloat();
    const xVal   = tf.tensor3d(Xval.map((row: number[]) => row.map((v: number) => [v])));
    const yVal   = tf.oneHot(tf.tensor1d(yval, "int32"), 2).toFloat();

    // 1D CNN: Conv1D(8,k=3,relu) → MaxPool(2) → Flatten → Dense(16,relu) → Dropout(0.25) → Dense(2,softmax)
    const model = tf.sequential({
      layers: [
        tf.layers.conv1d({ inputShape: [WINDOW_SIZE, 1], filters: 8, kernelSize: 3, activation: "relu" }),
        tf.layers.maxPooling1d({ poolSize: 2 }),
        tf.layers.flatten(),
        tf.layers.dense({ units: 16, activation: "relu" }),
        tf.layers.dropout({ rate: 0.25 }),
        tf.layers.dense({ units: 2, activation: "softmax" }),
      ],
    });

    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    await model.fit(xTrain, yTrain, {
      epochs: TOTAL_EPOCHS,
      validationData: [xVal, yVal],
      batchSize: 16,
      shuffle: true,
      callbacks: {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        onEpochEnd: async (ep: number, logs: any) => {
          if (stopRef.current) model.stopTraining = true;
          setEpoch(ep + 1);
          setTrainLoss(parseFloat((logs?.loss ?? 0).toFixed(4)));
        },
      },
    });

    if (!stopRef.current) {
      // Final metrics on validation set
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const predTensor = model.predict(xVal) as any;
      const predData: Float32Array = predTensor.dataSync();
      predTensor.dispose();
      const preds = Array.from({ length: Xval.length }, (_, i) =>
        predData[i * 2 + 1] >= predData[i * 2] ? 1 : 0
      );
      const metrics = computeMetrics(preds, yval);
      setValAcc(parseFloat((metrics.accuracy * 100).toFixed(1)));
      setF1Score(parseFloat((metrics.f1 * 100).toFixed(1)));
      modelRef.current = model;
      setMode("ready");
    } else {
      setMode("idle");
    }

    xTrain.dispose(); yTrain.dispose(); xVal.dispose(); yVal.dispose();
  }, [readings]);

  const reset = () => {
    stopRef.current = true;
    modelRef.current = null;
    setMode("idle");
    setEpoch(0);
    setValAcc(null);
    setF1Score(null);
    setTrainLoss(null);
    setProbs(null);
  };

  const hasEnough = readings.length >= MIN_SAMPLES;
  const predicted = probs ? (probs[1] >= probs[0] ? 1 : 0) : null;

  return (
    <div className="bg-card rounded-xl border border-border p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">AI Fatigue Classifier</h2>
          <span className="text-xs text-muted-foreground font-normal">(TensorFlow.js CNN)</span>
        </div>
        <div className="flex items-center gap-2">
          {mode !== "training" && (
            <button
              onClick={startTraining}
              disabled={!hasEnough}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium border transition-colors ${
                hasEnough
                  ? "bg-primary/10 text-primary border-primary/30 hover:bg-primary/20"
                  : "bg-secondary text-muted-foreground border-border cursor-not-allowed"
              }`}
            >
              <Play className="w-3.5 h-3.5" />
              {mode === "ready" ? "Retrain" : "Train"}
            </button>
          )}
          {mode === "training" && (
            <button
              onClick={() => { stopRef.current = true; }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium bg-red-500/10 text-red-400 border border-red-500/30"
            >
              <Square className="w-3.5 h-3.5" />
              Stop
            </button>
          )}
          {mode === "ready" && (
            <button
              onClick={reset}
              className="p-1.5 rounded-lg border border-border text-muted-foreground hover:text-foreground transition-colors"
              title="Reset model"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Status badges */}
      <div className="flex flex-wrap gap-2 mb-5 text-xs">
        <span
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full border font-medium ${
            mode === "training"
              ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
              : mode === "ready"
              ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
              : "bg-secondary border-border text-muted-foreground"
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              mode === "training"
                ? "bg-amber-400 animate-pulse"
                : mode === "ready"
                ? "bg-emerald-400 animate-pulse"
                : "bg-muted-foreground"
            }`}
          />
          {mode === "training"
            ? `Training — epoch ${epoch}/${TOTAL_EPOCHS}`
            : mode === "ready"
            ? "Live Inference"
            : "Idle"}
        </span>

        {valAcc !== null && (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full border bg-blue-500/10 border-blue-500/30 text-blue-400 font-medium">
            <FlaskConical className="w-3 h-3" />
            Accuracy: {valAcc}%
          </span>
        )}

        {f1Score !== null && (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full border bg-purple-500/10 border-purple-500/30 text-purple-400 font-medium">
            <Target className="w-3 h-3" />
            F1 Score: {f1Score}%
          </span>
        )}

        {trainLoss !== null && mode === "training" && (
          <span className="px-2.5 py-1 rounded-full border bg-secondary border-border text-muted-foreground">
            Loss: {trainLoss}
          </span>
        )}
      </div>

      {/* Not enough data */}
      {!hasEnough && (
        <div className="text-center py-4">
          <p className="text-sm text-muted-foreground">
            Collecting training data…{" "}
            <span className="text-foreground font-medium">{readings.length}</span> /{" "}
            {MIN_SAMPLES} readings
          </p>
          <div className="mt-3 h-1.5 bg-secondary rounded-full overflow-hidden mx-8">
            <div
              className="h-full bg-primary/50 rounded-full transition-all duration-500"
              style={{ width: `${Math.min((readings.length / MIN_SAMPLES) * 100, 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Training progress */}
      {mode === "training" && (
        <div className="mb-4">
          <div className="h-2 bg-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-150"
              style={{ width: `${(epoch / TOTAL_EPOCHS) * 100}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground text-center mt-1.5">
            Training 1D-CNN · Conv1D → MaxPool → Dense(16) → Dense(2)
          </p>
        </div>
      )}

      {/* Idle with enough data */}
      {mode === "idle" && hasEnough && (
        <p className="text-sm text-muted-foreground text-center py-3">
          Press{" "}
          <span className="text-foreground font-medium">Train</span> to build a 1D-CNN from{" "}
          <span className="text-foreground font-medium">{readings.length}</span> readings.
        </p>
      )}

      {/* Live prediction */}
      {mode === "ready" && (
        <div className="space-y-4">
          {probs && predicted !== null ? (
            <>
              <div className="flex items-center justify-center gap-2 p-3 rounded-lg bg-secondary">
                <Zap className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium text-foreground">
                  AI Prediction:{" "}
                  <span className={predicted === 0 ? "text-emerald-400" : "text-red-400"}>
                    {CLASS_LABELS[predicted]}
                  </span>
                </span>
                <span className="text-xs text-muted-foreground ml-1">
                  ({(probs[predicted] * 100).toFixed(0)}% confident)
                </span>
              </div>

              {CLASS_LABELS.map((label, i) => (
                <div key={label}>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className={`font-semibold ${i === 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {label}
                    </span>
                    <span className="text-muted-foreground">{(probs[i] * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-3 bg-secondary rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        i === 0 ? "bg-emerald-400" : "bg-red-400"
                      }`}
                      style={{ width: `${probs[i] * 100}%` }}
                    />
                  </div>
                </div>
              ))}

              {/* Accuracy + F1 metric cards */}
              {(valAcc !== null || f1Score !== null) && (
                <div className="grid grid-cols-2 gap-3 pt-2">
                  {valAcc !== null && (
                    <div className="bg-blue-500/5 border border-blue-500/20 rounded-lg p-3 text-center">
                      <p className="text-xs text-muted-foreground mb-1">Val Accuracy</p>
                      <p className="text-2xl font-bold text-blue-400">{valAcc}%</p>
                    </div>
                  )}
                  {f1Score !== null && (
                    <div className="bg-purple-500/5 border border-purple-500/20 rounded-lg p-3 text-center">
                      <p className="text-xs text-muted-foreground mb-1">F1 Score</p>
                      <p className="text-2xl font-bold text-purple-400">{f1Score}%</p>
                    </div>
                  )}
                </div>
              )}

              <p className="text-xs text-muted-foreground text-center pt-1">
                Inference on last {WINDOW_SIZE} readings · updates live
              </p>
            </>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-3">
              Waiting for live readings…
            </p>
          )}
        </div>
      )}
    </div>
  );
}
