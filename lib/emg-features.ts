// EMG Feature Extraction — JavaScript port of the Python ml/train_model.py pipeline.
// Extracts 8 fatigue-relevant features from raw EMG signal windows.
//
// Reference: Zenodo EMG Fatigue Dataset (doi:10.5281/zenodo.5189275)

export const EMG_FEATURE_NAMES = ["RMS", "MAV", "ZCR", "MDF", "MNF", "Power", "SM1", "SM2"] as const;
export type EMGFeatureName = (typeof EMG_FEATURE_NAMES)[number];

export const DEFAULT_SAMPLE_RATE = 200;
export const DEFAULT_WINDOW_SIZE = 400; // 2 seconds at 200 Hz

// StandardScaler parameters (loaded from scaler_params.json)
export interface ScalerParams {
  mean: number[];
  scale: number[];
  feature_names: string[];
}

/**
 * Compute Root Mean Square of a signal window.
 */
export function rms(window: number[]): number {
  let sum = 0;
  for (let i = 0; i < window.length; i++) sum += window[i] * window[i];
  return Math.sqrt(sum / window.length);
}

/**
 * Compute Mean Absolute Value of a signal window.
 */
export function mav(window: number[]): number {
  let sum = 0;
  for (let i = 0; i < window.length; i++) sum += Math.abs(window[i]);
  return sum / window.length;
}

/**
 * Compute Zero Crossing Rate.
 */
export function zcr(window: number[]): number {
  let count = 0;
  for (let i = 1; i < window.length; i++) {
    if ((window[i] >= 0 && window[i - 1] < 0) || (window[i] < 0 && window[i - 1] >= 0)) {
      count++;
    }
  }
  return count / window.length;
}

/**
 * Compute Power Spectral Density using a simple periodogram approach.
 * Returns [frequencies, psd] arrays for the given frequency band.
 */
export function welch(
  window: number[],
  fs: number = DEFAULT_SAMPLE_RATE,
  lowCutoff: number = 10,
  highCutoff: number = 99
): { freqs: number[]; psd: number[]; totalPower: number } {
  const n = Math.min(256, window.length);

  // Simple periodogram via DFT magnitude squared
  const psd: number[] = [];
  const freqs: number[] = [];

  for (let k = 0; k < Math.floor(n / 2); k++) {
    const freq = (k * fs) / n;
    if (freq >= lowCutoff && freq <= highCutoff) {
      freqs.push(freq);

      let real = 0;
      let imag = 0;
      for (let t = 0; t < n; t++) {
        const angle = (-2 * Math.PI * k * t) / n;
        real += window[t] * Math.cos(angle);
        imag += window[t] * Math.sin(angle);
      }
      psd.push((real * real + imag * imag) / (n * fs));
    }
  }

  let totalPower = 0;
  for (let i = 1; i < psd.length; i++) {
    totalPower += ((freqs[i] - freqs[i - 1]) * (psd[i] + psd[i - 1])) / 2;
  }

  return { freqs, psd, totalPower };
}

/**
 * Compute Median Frequency from PSD.
 */
export function medianFrequency(freqs: number[], psd: number[]): number {
  if (freqs.length === 0 || psd.length === 0) return 0;

  const cumsum: number[] = [];
  let running = 0;
  for (let i = 0; i < psd.length; i++) {
    running += psd[i];
    cumsum.push(running);
  }

  const total = cumsum[cumsum.length - 1];
  if (total === 0) return 0;

  const halfTotal = total / 2;
  for (let i = 0; i < cumsum.length; i++) {
    if (cumsum[i] >= halfTotal) return freqs[i];
  }

  return freqs[freqs.length - 1];
}

/**
 * Compute Mean Frequency from PSD.
 */
export function meanFrequency(freqs: number[], psd: number[]): number {
  if (freqs.length === 0 || psd.length === 0) return 0;

  let num = 0;
  let den = 0;
  for (let i = 0; i < freqs.length; i++) {
    num += freqs[i] * psd[i];
    den += psd[i];
  }
  return den === 0 ? 0 : num / den;
}

/**
 * Apply StandardScaler normalization to extracted features.
 *   normalized = (raw - mean) / scale
 */
export function scaleFeatures(features: number[], params: ScalerParams): number[] {
  return features.map((v, i) => (v - params.mean[i]) / (params.scale[i] || 1));
}

/**
 * Extract all 8 EMG fatigue features from a raw signal window.
 * Returns [RMS, MAV, ZCR, MDF, MNF, TotalPower, SM1, SM2]
 */
export function extractFeatures(
  window: number[],
  fs: number = DEFAULT_SAMPLE_RATE,
  lowCutoff: number = 10,
  highCutoff: number = 99
): number[] {
  const rmsVal = rms(window);
  const mavVal = mav(window);
  const zcrVal = zcr(window);

  const { freqs, psd, totalPower } = welch(window, fs, lowCutoff, highCutoff);
  const mdfVal = medianFrequency(freqs, psd);
  const mnfVal = meanFrequency(freqs, psd);

  const sm1Val = totalPower > 0 ? meanFrequency(freqs, psd.map((v, i) => v * freqs[i])) : 0;
  const sm2 = totalPower > 0
    ? (() => {
        let num = 0;
        for (let i = 0; i < freqs.length; i++) {
          num += freqs[i] * freqs[i] * psd[i];
        }
        return num / totalPower;
      })()
    : 0;

  return [rmsVal, mavVal, zcrVal, mdfVal, mnfVal, totalPower, sm1Val, sm2];
}
