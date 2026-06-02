// EMG Muscle Fatigue Classifier — utility module
// Powered by brain.js (loaded dynamically client-side via AIPanel).
//
// Two classes:
//   Class 0 = "Normal"  — low EMG activity  (signal_percentage < 30 %)
//   Class 1 = "Fatigue" — elevated activity  (signal_percentage ≥ 30 %)
//
// Feature window: last WINDOW_SIZE readings, each normalised to [0, 1].

export const WINDOW_SIZE = 20;
export const CLASS_LABELS = ["Normal", "Fatigue"] as const;
export type ClassLabel = (typeof CLASS_LABELS)[number];

/** One training sample in brain.js { input, output } format */
export interface TrainingSample {
  input: number[];
  /** One-hot encoding: [1, 0] = Normal | [0, 1] = Fatigue */
  output: number[];
}

/**
 * Build labelled brain.js training samples from database readings.
 *
 * Labelling rule (per-window):
 *   signal_percentage < 30  →  Normal  (label 0)
 *   signal_percentage ≥ 30  →  Fatigue (label 1)
 */
export function buildDataset(
  readings: { signal_percentage: number; status: string }[]
): { samples: TrainingSample[]; labels: number[] } {
  const samples: TrainingSample[] = [];
  const labels: number[] = [];

  for (let i = WINDOW_SIZE; i < readings.length; i++) {
    // Sliding window of normalised EMG values
    const window = readings
      .slice(i - WINDOW_SIZE, i)
      .map((r) => parseFloat(String(r.signal_percentage)) / 100);

    // Fatigue when the current reading crosses the 30 % threshold
    const pct = parseFloat(String(readings[i].signal_percentage));
    const isFatigue = pct >= 30;
    const label = isFatigue ? 1 : 0;

    samples.push({
      input: window,
      output: isFatigue ? [0, 1] : [1, 0], // one-hot: [P(Normal), P(Fatigue)]
    });
    labels.push(label);
  }

  return { samples, labels };
}

/**
 * Compute binary classification metrics.
 * Positive class = Fatigue (label 1).
 *
 * Returns:
 *   accuracy  — (TP + TN) / N
 *   precision — TP / (TP + FP)
 *   recall    — TP / (TP + FN)
 *   f1        — 2 · precision · recall / (precision + recall)
 */
export function computeMetrics(
  predictions: number[],
  labels: number[]
): { accuracy: number; f1: number; precision: number; recall: number } {
  let tp = 0, fp = 0, fn = 0, tn = 0;

  for (let i = 0; i < predictions.length; i++) {
    const pred = predictions[i];   // 0 = Normal, 1 = Fatigue
    const actual = labels[i];
    if (pred === 1 && actual === 1) tp++;
    else if (pred === 1 && actual === 0) fp++;
    else if (pred === 0 && actual === 1) fn++;
    else tn++;
  }

  const total = tp + tn + fp + fn;
  const accuracy  = total > 0        ? (tp + tn) / total                           : 0;
  const precision = tp + fp > 0      ? tp / (tp + fp)                              : 0;
  const recall    = tp + fn > 0      ? tp / (tp + fn)                              : 0;
  const f1        = precision + recall > 0
    ? (2 * precision * recall) / (precision + recall)
    : 0;

  return { accuracy, f1, precision, recall };
}

/** Fisher-Yates shuffle returning a shuffled index array */
export function shuffleIndices(n: number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}

// ─── Minimal MLP (zero external dependencies) ────────────────────────────────
// Architecture: INPUT → HIDDEN (ReLU) → OUTPUT (sigmoid)
// Implements the same trainAsync / run interface previously provided by brain.js.

function sigmoid(x: number): number { return 1 / (1 + Math.exp(-x)); }

export class MLP {
  private w1: number[][];
  private b1: number[];
  private w2: number[][];
  private b2: number[];

  constructor(
    private inputSize: number,
    private hiddenSize: number,
    private outputSize: number,
  ) {
    // He (Kaiming) initialisation
    const s1 = Math.sqrt(2 / inputSize);
    const s2 = Math.sqrt(2 / hiddenSize);
    this.w1 = Array.from({ length: hiddenSize }, () =>
      Array.from({ length: inputSize }, () => (Math.random() * 2 - 1) * s1)
    );
    this.b1 = Array(hiddenSize).fill(0);
    this.w2 = Array.from({ length: outputSize }, () =>
      Array.from({ length: hiddenSize }, () => (Math.random() * 2 - 1) * s2)
    );
    this.b2 = Array(outputSize).fill(0);
  }

  private forward(input: number[]): { z1: number[]; h: number[]; out: number[] } {
    // Hidden: z1 = W1·x + b1,  h = ReLU(z1)
    const z1 = this.w1.map((row, j) =>
      row.reduce((s, w, i) => s + w * input[i], 0) + this.b1[j]
    );
    const h = z1.map(x => (x > 0 ? x : 0));
    // Output: out = sigmoid(W2·h + b2)
    const out = this.w2.map((row, k) =>
      sigmoid(row.reduce((s, w, j) => s + w * h[j], 0) + this.b2[k])
    );
    return { z1, h, out };
  }

  /** Run inference — returns [P(Normal), P(Fatigue)] */
  run(input: number[]): number[] {
    return this.forward(input).out;
  }

  /** SGD training with async yielding so the browser UI stays responsive */
  async trainAsync(
    samples: TrainingSample[],
    opts: {
      iterations: number;
      learningRate: number;
      errorThresh?: number;
      callbackPeriod: number;
      callback: (state: { iterations: number; error: number }) => void;
    }
  ): Promise<void> {
    const lr = opts.learningRate;
    for (let iter = 1; iter <= opts.iterations; iter++) {
      let totalLoss = 0;
      for (const s of samples) {
        const { z1, h, out } = this.forward(s.input);
        // MSE output error
        const dOut = out.map((o, k) => o - s.output[k]);
        totalLoss += dOut.reduce((acc, e) => acc + e * e, 0) / dOut.length;
        // Output layer gradient (sigmoid′ = o·(1−o))
        const dZ2 = dOut.map((d, k) => d * out[k] * (1 - out[k]));
        for (let k = 0; k < this.outputSize; k++) {
          for (let j = 0; j < this.hiddenSize; j++) this.w2[k][j] -= lr * dZ2[k] * h[j];
          this.b2[k] -= lr * dZ2[k];
        }
        // Hidden layer gradient (ReLU′ = z1 > 0 ? 1 : 0)
        const dH = Array.from({ length: this.hiddenSize }, (_, j) =>
          this.w2.reduce((sum, row, k) => sum + row[j] * dZ2[k], 0)
        );
        const dZ1 = dH.map((d, j) => d * (z1[j] > 0 ? 1 : 0));
        for (let j = 0; j < this.hiddenSize; j++) {
          for (let i = 0; i < this.inputSize; i++) this.w1[j][i] -= lr * dZ1[j] * s.input[i];
          this.b1[j] -= lr * dZ1[j];
        }
      }
      if (iter % opts.callbackPeriod === 0 || iter === opts.iterations) {
        opts.callback({ iterations: iter, error: totalLoss / samples.length });
        await new Promise<void>(resolve => setTimeout(resolve, 0));
      }
    }
  }
}
