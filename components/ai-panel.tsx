"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Brain, Play, Square, RefreshCw, Zap, FlaskConical, Target } from "lucide-react";
import {
  buildDataset,
  shuffleIndices,
  WINDOW_SIZE,
  CLASS_LABELS,
  computeMetrics,
  MLP,
  type TrainingSample,
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
const TOTAL_EPOCHS = 100;

export function AIPanel({ readings }: AIPanelProps) {
  const [mode, setMode] = useState<Mode>("idle");
  const modelRef = useRef<MLP | null>(null);
  const [epoch, setEpoch] = useState(0);
  const [valAcc, setValAcc] = useState<number | null>(null);
  const [f1Score, setF1Score] = useState<number | null>(null);
  const [trainLoss, setTrainLoss] = useState<number | null>(null);
  const [probs, setProbs] = useState<[number, number] | null>(null);
  const stopRef = useRef(false);

  // Run live inference whenever readings update (ready mode only)
  useEffect(() => {
    if (mode !== "ready" || !modelRef.current || readings.length < WINDOW_SIZE) return;
    const win = readings
      .slice(-WINDOW_SIZE)
      .map((r) => parseFloat(String(r.signal_percentage)) / 100);
    const output = modelRef.current.run(win);
    setProbs([output[0] ?? 0, output[1] ?? 0]);
  }, [readings, mode]);

  const startTraining = useCallback(async () => {
    if (readings.length < MIN_SAMPLES) return;
    setMode("training");
    setEpoch(0);
    stopRef.current = false;

    const { samples, labels } = buildDataset(readings);
    if (samples.length < 20) { setMode("idle"); return; }

    // 80 / 20 train / validation split
    const idx = shuffleIndices(samples.length);
    const split = Math.floor(idx.length * 0.8);
    const trainSamples: TrainingSample[] = idx.slice(0, split).map((i) => samples[i]);
    const valSamples:   TrainingSample[] = idx.slice(split).map((i) => samples[i]);
    const valLabels:    number[]         = idx.slice(split).map((i) => labels[i]);

    const net = new MLP(WINDOW_SIZE, 16, 2);

    // trainAsync yields to the browser event loop so the UI stays responsive
    await net.trainAsync(trainSamples, {
      iterations:     TOTAL_EPOCHS,
      learningRate:   0.05,
      errorThresh:    0.005,
      callbackPeriod: 5,
      callback: (state: { iterations: number; error: number }) => {
        if (stopRef.current) return;
        setEpoch(state.iterations);
        setTrainLoss(parseFloat(state.error.toFixed(4)));
      },
    });

    if (stopRef.current) { setMode("idle"); return; }

    // Evaluate on the held-out validation set
    const preds = valSamples.map((s) => {
      const out = net.run(s.input);
      return out[1] >= out[0] ? 1 : 0; // 1 = Fatigue
    });

    const metrics = computeMetrics(preds, valLabels);
    setValAcc(parseFloat((metrics.accuracy * 100).toFixed(1)));
    setF1Score(parseFloat((metrics.f1 * 100).toFixed(1)));
    modelRef.current = net;
    setMode("ready");
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
  const predicted = probs ? ((probs[1] ?? 0) >= (probs[0] ?? 0) ? 1 : 0) : null;

  return (
    <div className="bg-card rounded-xl border border-border p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">AI Fatigue Classifier</h2>
          <span className="text-xs text-muted-foreground font-normal">(Neural Network)</span>
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

      {/* Training progress bar */}
      {mode === "training" && (
        <div className="mb-4">
          <div className="h-2 bg-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-150"
              style={{ width: `${(epoch / TOTAL_EPOCHS) * 100}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground text-center mt-1.5">
            Training on {readings.length} samples · {WINDOW_SIZE}-input → 16 hidden → 2 output
          </p>
        </div>
      )}

      {/* Idle — enough data available */}
      {mode === "idle" && hasEnough && (
        <p className="text-sm text-muted-foreground text-center py-3">
          Press{" "}
          <span className="text-foreground font-medium">Train</span> to build a neural network
          from{" "}
          <span className="text-foreground font-medium">{readings.length}</span> readings.
        </p>
      )}

      {/* Live prediction */}
      {mode === "ready" && (
        <div className="space-y-4">
          {probs && predicted !== null ? (
            <>
              {/* Predicted class */}
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

              {/* Confidence bars for each class */}
              {CLASS_LABELS.map((label, i) => (
                <div key={label}>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className={`font-semibold ${i === 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {label}
                    </span>
                    <span className="text-muted-foreground">
                      {(probs[i] * 100).toFixed(1)}%
                    </span>
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

              {/* Accuracy + F1 Score metric cards */}
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
                Inference on last {WINDOW_SIZE} readings · updates every 2 s
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


interface Reading {
  signal_percentage: number;
  status: string;
}

interface AIPanelProps {
  readings: Reading[];
}

type Mode = "idle" | "training" | "ready";

const MIN_SAMPLES = WINDOW_SIZE + 30;
const TOTAL_EPOCHS = 100;

export function AIPanel({ readings }: AIPanelProps) {
  const [mode, setMode] = useState<Mode>("idle");
  const [model, setModel] = useState<MuscleNet | null>(null);
  const [epoch, setEpoch] = useState(0);
  const [valAcc, setValAcc] = useState<number | null>(null);
  const [trainLoss, setTrainLoss] = useState<number | null>(null);
  const [probs, setProbs] = useState<[number, number] | null>(null);
  const stopRef = useRef(false);

  // Run live inference whenever readings update (test mode)
  useEffect(() => {
    if (mode !== "ready" || !model || readings.length < WINDOW_SIZE) return;
    const win = readings
      .slice(-WINDOW_SIZE)
      .map((r) => parseFloat(String(r.signal_percentage)) / 100);
    setProbs(model.predict(win));
  }, [readings, model, mode]);

  const startTraining = useCallback(async () => {
    if (readings.length < MIN_SAMPLES) return;
    setMode("training");
    setEpoch(0);
    stopRef.current = false;

    const { X, y } = buildDataset(readings);
    if (X.length < 20) { setMode("idle"); return; }

    // 80/20 train/val split
    const idx = shuffleIndices(X.length);
    const split = Math.floor(idx.length * 0.8);
    const Xtr = idx.slice(0, split).map((i) => X[i]);
    const ytr = idx.slice(0, split).map((i) => y[i]);
    const Xval = idx.slice(split).map((i) => X[i]);
    const yval = idx.slice(split).map((i) => y[i]);

    const net = new MuscleNet();

    for (let e = 0; e < TOTAL_EPOCHS && !stopRef.current; e++) {
      const { loss } = net.trainBatch(Xtr, ytr, 0.05);

      // Validation accuracy
      let correct = 0;
      for (let i = 0; i < Xval.length; i++) {
        const p = net.predict(Xval[i]);
        if ((p[0] >= p[1] ? 0 : 1) === yval[i]) correct++;
      }

      setEpoch(e + 1);
      setTrainLoss(parseFloat(loss.toFixed(4)));
      setValAcc(Xval.length > 0 ? parseFloat(((correct / Xval.length) * 100).toFixed(1)) : null);

      // Yield to UI thread every 5 epochs to keep it responsive
      if (e % 5 === 0) await new Promise((r) => setTimeout(r, 0));
    }

    if (!stopRef.current) {
      setModel(net);
      setMode("ready");
    } else {
      setMode("idle");
    }
  }, [readings]);

  const reset = () => {
    stopRef.current = true;
    setModel(null);
    setMode("idle");
    setEpoch(0);
    setValAcc(null);
    setTrainLoss(null);
    setProbs(null);
  };

  const hasEnough = readings.length >= MIN_SAMPLES;
  const predicted = probs ? (probs[0] >= probs[1] ? 0 : 1) : null;

  return (
    <div className="bg-card rounded-xl border border-border p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">AI Fatigue Classifier</h2>
          <span className="text-xs text-muted-foreground font-normal">(Neural Network)</span>
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
          {model && (
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
            Val Accuracy: {valAcc}%
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
            Training on {readings.length} samples · {WINDOW_SIZE}-input → 16 hidden → 2 output
          </p>
        </div>
      )}

      {/* Idle with enough data */}
      {mode === "idle" && hasEnough && (
        <p className="text-sm text-muted-foreground text-center py-3">
          Press{" "}
          <span className="text-foreground font-medium">Train</span> to build a neural network
          from{" "}
          <span className="text-foreground font-medium">{readings.length}</span> readings.
        </p>
      )}

      {/* Live prediction (test mode) */}
      {mode === "ready" && (
        <div className="space-y-4">
          {probs && predicted !== null ? (
            <>
              {/* Predicted class */}
              <div className="flex items-center justify-center gap-2 p-3 rounded-lg bg-secondary">
                <Zap className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium text-foreground">
                  AI Prediction:{" "}
                  <span className={predicted === 0 ? "text-emerald-400" : "text-red-400"}>
                    {CLASS_LABELS[predicted]}
                  </span>
                </span>
                <span className="text-xs text-muted-foreground ml-1">
                  ({((probs[predicted]) * 100).toFixed(0)}% confident)
                </span>
              </div>

              {/* Confidence bars */}
              {CLASS_LABELS.map((label, i) => (
                <div key={label}>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span
                      className={`font-semibold ${
                        i === 0 ? "text-emerald-400" : "text-red-400"
                      }`}
                    >
                      {label}
                    </span>
                    <span className="text-muted-foreground">
                      {((probs[i]) * 100).toFixed(1)}%
                    </span>
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

              <p className="text-xs text-muted-foreground text-center pt-1">
                Inference on last {WINDOW_SIZE} readings · updates every 2 s
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
