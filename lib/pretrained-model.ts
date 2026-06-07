// Pretrained EMG Fatigue Model Loader
// Loads the 1D-CNN model trained on the Zenodo EMG Fatigue Dataset
// via TensorFlow.js Layers format (model.json + sharded weights).
//
// Two model types supported:
//   1. CNN (raw):   1D-CNN on raw 400-sample EMG windows
//   2. Feature (ML): Feature-based model using 8 extracted features + scaler

export interface PretrainedModelConfig {
  cnnModelPath: string;
  scalerPath: string;
}

export interface ScalerParams {
  mean: number[];
  scale: number[];
  feature_names: string[];
}

export interface EvaluationResults {
  data_split: {
    train_pct: number;
    test_pct: number;
    stratified: boolean;
    random_state: number;
  };
  dataset: {
    url: string;
    subjects: number;
    sampling_rate_hz: number;
    window_size_samples: number;
    window_size_seconds: number;
    binary_classification: number;
    classes: Record<string, string>;
    class_distribution: { fresh: number; fatigued: number };
  };
  feature_models: Record<string, unknown>;
  cnn_model: {
    architecture: string;
    input_shape: string;
    optimizer: string;
    expected_accuracy: string;
    expected_auc: string;
    training_epochs: number;
    note: string;
    window_size?: number;
    sample_rate_hz?: number;
    accuracy?: number | null;
    auc?: number | null;
    precision_fresh?: number | null;
    recall_fresh?: number | null;
    f1_fresh?: number | null;
    precision_fatigued?: number | null;
    recall_fatigued?: number | null;
    f1_fatigued?: number | null;
    early_stopping_patience?: number;
    regularization?: string;
  };
  loso_cv: {
    method: string;
    mean_accuracy: number;
    std: number;
    model: string;
  };
}

const DEFAULT_MODEL_PATH = "/models/emg_cnn_model/model.json";
const DEFAULT_SCALER_PATH = "/models/scaler_params.json";
const DEFAULT_EVAL_PATH = "/models/evaluation_results.json";

// Module-level cache — loaded once
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _cnnModel: any = null;
let _scalerParams: ScalerParams | null = null;
let _evalResults: EvaluationResults | null = null;
let _tf: any = null;

/**
 * Ensure TensorFlow.js is loaded (dynamic import to avoid server bundle).
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function ensureTF(): Promise<any> {
  if (!_tf) {
    _tf = await import("@tensorflow/tfjs");
  }
  return _tf;
}

/**
 * Load the pretrained 1D-CNN model from TF.js Layers format.
 * Returns the loaded model, or null if loading fails.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function loadPretrainedCNN(modelPath?: string): Promise<any | null> {
  if (_cnnModel) return _cnnModel;

  try {
    const tf = await ensureTF();
    const path = modelPath || DEFAULT_MODEL_PATH;
    _cnnModel = await tf.loadLayersModel(path);
    console.log("[PretrainedModel] 1D-CNN loaded from", path);
    return _cnnModel;
  } catch (e) {
    console.warn("[PretrainedModel] Failed to load CNN model:", e);
    return null;
  }
}

/**
 * Load StandardScaler parameters for feature-based normalization.
 */
export async function loadScalerParams(scalerPath?: string): Promise<ScalerParams | null> {
  if (_scalerParams) return _scalerParams;

  try {
    const path = scalerPath || DEFAULT_SCALER_PATH;
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    _scalerParams = await res.json();
    return _scalerParams;
  } catch (e) {
    console.warn("[PretrainedModel] Failed to load scaler params:", e);
    return null;
  }
}

/**
 * Load evaluation results (accuracy, AUC, LOSO, etc.).
 */
export async function loadEvaluationResults(evalPath?: string): Promise<EvaluationResults | null> {
  if (_evalResults) return _evalResults;

  try {
    const path = evalPath || DEFAULT_EVAL_PATH;
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    _evalResults = await res.json();
    return _evalResults;
  } catch (e) {
    console.warn("[PretrainedModel] Failed to load evaluation results:", e);
    return null;
  }
}

/**
 * Run inference with the pretrained CNN model on a raw EMG window.
 *
 * @param window - Array of 400 normalized signal values (range [-1, 1] or [0, 1])
 * @returns Object with predicted class (0=Normal/Fresh, 1=Fatigue) and confidence
 */
export async function predictWithCNN(
  window: number[]
): Promise<{ classIndex: number; className: string; confidence: number; probabilities: [number, number] } | null> {
  const model = await loadPretrainedCNN();
  if (!model) return null;

  try {
    const tf = await ensureTF();
    const input = tf.tensor3d([window.map((v) => [v])]);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const pred = model.predict(input) as any;
    const data = new Float32Array(await pred.data());
    input.dispose();
    pred.dispose();

    const p0 = data[0];
    const p1 = data[1];
    const classIndex = p1 >= p0 ? 1 : 0;

    return {
      classIndex,
      className: classIndex === 0 ? "Normal" : "Fatigue",
      confidence: data[classIndex],
      probabilities: [p0, p1],
    };
  } catch (e) {
    console.error("[PretrainedModel] Inference error:", e);
    return null;
  }
}

/**
 * Scale extracted features using StandardScaler params, then run a simple
 * logistic regression classifier (in-browser approximation of the RF model).
 *
 * An alternative lightweight approach when the full pipeline is not available.
 * Returns class prediction with confidence.
 */
export function predictWithFeatures(
  features: number[],
  scalerParams: ScalerParams
): { classIndex: number; className: string } | null {
  if (!scalerParams || features.length !== scalerParams.feature_names.length) {
    return null;
  }

  // Scale features
  const scaled = features.map((v, i) => (v - scalerParams.mean[i]) / (scalerParams.scale[i] || 1));

  // Heuristic: MDF (index 3) is the strongest fatigue indicator.
  // Lower median frequency → more Fatigue
  const mdfZScore = scaled[3];   // MDF z-score
  const rmsZScore = scaled[0];   // RMS z-score

  // Simple linear combination (approximate logistic regression decision boundary)
  const score = -0.6 * mdfZScore + 0.4 * rmsZScore;
  const prob = 1 / (1 + Math.exp(-score));

  return {
    classIndex: prob >= 0.5 ? 1 : 0,
    className: prob >= 0.5 ? "Fatigue" : "Normal",
  };
}

/**
 * Check if the pretrained CNN model is available (model.json loads successfully).
 */
export async function isPretrainedModelAvailable(modelPath?: string): Promise<boolean> {
  const model = await loadPretrainedCNN(modelPath);
  return model !== null;
}

/**
 * Reset the module cache (e.g., to force reload).
 */
export function resetModelCache(): void {
  _cnnModel = null;
  _scalerParams = null;
  _evalResults = null;
}
