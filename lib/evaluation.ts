// Evaluation module for EMG fatigue classification models.
// Computes Accuracy, Precision, Recall, F1 Score, AUC-ROC, and Confusion Matrix.

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  tp: number;
  tn: number;
  fp: number;
  fn: number;
  total: number;
}

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface ExtendedMetrics extends ClassificationMetrics {
  auc: number;
  confusionMatrix: ConfusionMatrix;
  classDistribution: { fresh: number; fatigued: number };
}

/**
 * Compute binary classification metrics.
 * Positive class = 1 (Fatigued), Negative class = 0 (Fresh).
 */
export function computeMetrics(predictions: number[], labels: number[]): ClassificationMetrics {
  let tp = 0, fp = 0, fn = 0, tn = 0;

  for (let i = 0; i < predictions.length; i++) {
    const pred = predictions[i];
    const actual = labels[i];
    if (pred === 1 && actual === 1) tp++;
    else if (pred === 1 && actual === 0) fp++;
    else if (pred === 0 && actual === 1) fn++;
    else tn++;
  }

  const total = tp + tn + fp + fn;
  const accuracy = total > 0 ? (tp + tn) / total : 0;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return { accuracy, precision, recall, f1, tp, tn, fp, fn, total };
}

/**
 * Compute AUC-ROC using the trapezoidal method.
 * @param scores - Predicted probabilities for positive class (Fatigued)
 * @param labels - True labels (0 = Fresh, 1 = Fatigued)
 */
export function computeAUCROC(scores: number[], labels: number[]): number {
  const n = scores.length;

  // Sort by score descending, with tie-breaking
  const indexed = scores.map((s, i) => ({ score: s, label: labels[i] }));
  indexed.sort((a, b) => b.score - a.score || a.label - b.label);

  let auc = 0;
  let tpCount = 0;
  let fpCount = 0;
  let prevTp = 0;
  let prevFp = 0;

  for (let i = 0; i < n; i++) {
    if (i > 0 && indexed[i].score !== indexed[i - 1].score) {
      auc += trapeaoidArea(fpCount, prevFp, tpCount, prevTp);
      prevFp = fpCount;
      prevTp = tpCount;
    }
    if (indexed[i].label === 1) tpCount++;
    else fpCount++;
  }

  auc += trapeaoidArea(fpCount, prevFp, tpCount, prevTp);
  const pos = labels.filter((l) => l === 1).length;
  const neg = labels.filter((l) => l === 0).length;

  if (pos === 0 || neg === 0) return 0;
  return auc / (pos * neg);
}

function trapeaoidArea(x1: number, x2: number, y1: number, y2: number): number {
  const base = Math.abs(x1 - x2);
  const height = (y1 + y2) / 2;
  return base * height;
}

/**
 * Compute extended metrics including AUC and confusion matrix.
 */
export function computeExtendedMetrics(
  predictions: number[],
  scores: number[],
  labels: number[]
): ExtendedMetrics {
  const base = computeMetrics(predictions, labels);
  const auc = computeAUCROC(scores, labels);

  const fresh = labels.filter((l) => l === 0).length;
  const fatigued = labels.filter((l) => l === 1).length;

  return {
    ...base,
    auc,
    confusionMatrix: { tn: base.tn, fp: base.fp, fn: base.fn, tp: base.tp },
    classDistribution: { fresh, fatigued },
  };
}

/**
 * Format metrics as percentages for display.
 */
export function formatMetrics(metrics: ClassificationMetrics): Record<string, string> {
  return {
    accuracy: (metrics.accuracy * 100).toFixed(1) + "%",
    precision: (metrics.precision * 100).toFixed(1) + "%",
    recall: (metrics.recall * 100).toFixed(1) + "%",
    f1: (metrics.f1 * 100).toFixed(1) + "%",
  };
}
