// Lightweight feedforward neural network for EMG muscle fatigue classification
// Architecture: WINDOW_SIZE → 16 hidden (ReLU) → 2 output (softmax)
// Class 0 = relaxed  |  Class 1 = active (moderate or contracted)
// No external dependencies — runs entirely in the browser.

export const WINDOW_SIZE = 20;
export const CLASS_LABELS = ["Relaxed", "Active"] as const;

function rand(scale: number): number {
  return (Math.random() * 2 - 1) * scale;
}

function relu(x: number): number {
  return x > 0 ? x : 0;
}

function softmax(x: number[]): number[] {
  const m = Math.max(...x);
  const e = x.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export class MuscleNet {
  private w1: number[][];
  private b1: number[];
  private w2: number[][];
  private b2: number[];

  static readonly HIDDEN = 16;
  static readonly OUTPUTS = 2;

  constructor() {
    const s1 = Math.sqrt(2 / WINDOW_SIZE);
    const s2 = Math.sqrt(2 / MuscleNet.HIDDEN);
    this.w1 = Array.from({ length: MuscleNet.HIDDEN }, () =>
      Array.from({ length: WINDOW_SIZE }, () => rand(s1))
    );
    this.b1 = new Array(MuscleNet.HIDDEN).fill(0);
    this.w2 = Array.from({ length: MuscleNet.OUTPUTS }, () =>
      Array.from({ length: MuscleNet.HIDDEN }, () => rand(s2))
    );
    this.b2 = new Array(MuscleNet.OUTPUTS).fill(0);
  }

  /** Returns [P(relaxed), P(active)] for a normalised input window */
  predict(window: number[]): [number, number] {
    const hRaw = this.w1.map((row, i) =>
      row.reduce((s, w, j) => s + w * window[j], 0) + this.b1[i]
    );
    const h = hRaw.map(relu);
    const outRaw = this.w2.map((row, i) =>
      row.reduce((s, w, j) => s + w * h[j], 0) + this.b2[i]
    );
    const p = softmax(outRaw);
    return [p[0], p[1]];
  }

  /** Mini-batch SGD step. Returns loss and training accuracy. */
  trainBatch(
    X: number[][],
    y: number[],
    lr = 0.05
  ): { loss: number; accuracy: number } {
    let totalLoss = 0;
    let correct = 0;

    const dW1 = this.w1.map((r) => r.map(() => 0));
    const db1 = new Array(MuscleNet.HIDDEN).fill(0);
    const dW2 = this.w2.map((r) => r.map(() => 0));
    const db2 = new Array(MuscleNet.OUTPUTS).fill(0);

    for (let n = 0; n < X.length; n++) {
      const x = X[n];
      const label = y[n];

      // Forward
      const hRaw = this.w1.map((row, i) =>
        row.reduce((s, w, j) => s + w * x[j], 0) + this.b1[i]
      );
      const h = hRaw.map(relu);
      const outRaw = this.w2.map((row, i) =>
        row.reduce((s, w, j) => s + w * h[j], 0) + this.b2[i]
      );
      const prob = softmax(outRaw);

      totalLoss += -Math.log(Math.max(prob[label], 1e-7));
      if (prob.indexOf(Math.max(...prob)) === label) correct++;

      // Backprop — output (softmax + CE shortcut)
      const dOut = prob.map((p, i) => p - (i === label ? 1 : 0));
      for (let i = 0; i < MuscleNet.OUTPUTS; i++) {
        db2[i] += dOut[i];
        for (let j = 0; j < MuscleNet.HIDDEN; j++) dW2[i][j] += dOut[i] * h[j];
      }

      // Hidden layer
      const dH = new Array(MuscleNet.HIDDEN).fill(0);
      for (let j = 0; j < MuscleNet.HIDDEN; j++)
        for (let i = 0; i < MuscleNet.OUTPUTS; i++) dH[j] += this.w2[i][j] * dOut[i];
      const dHRelu = dH.map((v, i) => v * (hRaw[i] > 0 ? 1 : 0));

      for (let i = 0; i < MuscleNet.HIDDEN; i++) {
        db1[i] += dHRelu[i];
        for (let j = 0; j < WINDOW_SIZE; j++) dW1[i][j] += dHRelu[i] * x[j];
      }
    }

    const n = X.length;
    for (let i = 0; i < MuscleNet.HIDDEN; i++) {
      this.b1[i] -= (lr * db1[i]) / n;
      for (let j = 0; j < WINDOW_SIZE; j++) this.w1[i][j] -= (lr * dW1[i][j]) / n;
    }
    for (let i = 0; i < MuscleNet.OUTPUTS; i++) {
      this.b2[i] -= (lr * db2[i]) / n;
      for (let j = 0; j < MuscleNet.HIDDEN; j++) this.w2[i][j] -= (lr * dW2[i][j]) / n;
    }

    return { loss: totalLoss / n, accuracy: correct / n };
  }
}

/** Build a labelled dataset from DB readings */
export function buildDataset(
  readings: { signal_percentage: number; status: string }[]
): { X: number[][]; y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];

  for (let i = WINDOW_SIZE; i < readings.length; i++) {
    const window = readings
      .slice(i - WINDOW_SIZE, i)
      .map((r) => parseFloat(String(r.signal_percentage)) / 100);
    const label = readings[i].status === "relaxed" ? 0 : 1;
    X.push(window);
    y.push(label);
  }

  return { X, y };
}

/** Fisher-Yates shuffle on indices */
export function shuffleIndices(n: number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}
