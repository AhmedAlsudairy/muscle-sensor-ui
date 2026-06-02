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
