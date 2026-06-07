"use client";

import { useState, useEffect } from "react";
import { BarChart3, Activity, Users, Zap, Target, FlaskConical, Cpu } from "lucide-react";
import { loadEvaluationResults, loadScalerParams } from "@/lib/pretrained-model";
import type { EvaluationResults, ScalerParams } from "@/lib/pretrained-model";

const CLASS_LABELS = ["Normal", "Fatigue"] as const;

export function EvaluationPanel() {
  const [evalData, setEvalData] = useState<EvaluationResults | null>(null);
  const [scalerParams, setScalerParams] = useState<ScalerParams | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"cnn" | "feature" | "dataset">("cnn");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [evaluation, scaler] = await Promise.all([
          loadEvaluationResults(),
          loadScalerParams(),
        ]);
        if (cancelled) return;
        setEvalData(evaluation);
        setScalerParams(scaler);
      } catch {
        if (!cancelled) setError("Failed to load evaluation data");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  if (loading) {
    return (
      <div className="bg-card rounded-xl border border-border p-6">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">Model Evaluation</h2>
        </div>
        <p className="text-sm text-muted-foreground text-center py-6">
          Loading evaluation results…
        </p>
      </div>
    );
  }

  if (error && !evalData) {
    return (
      <div className="bg-card rounded-xl border border-border p-6">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">Model Evaluation</h2>
        </div>
        <p className="text-sm text-red-400 text-center py-6">
          Could not load evaluation results. Run <code className="text-xs bg-secondary px-1 py-0.5 rounded">python ml/train_model.py --export</code> to generate them.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-card rounded-xl border border-border p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">Model Evaluation</h2>
          <span className="text-xs text-muted-foreground font-normal">
            (Zenodo EMG Fatigue Dataset)
          </span>
        </div>
        <span className="text-xs px-2 py-1 rounded-full border bg-emerald-500/10 border-emerald-500/30 text-emerald-400">
          Pretrained
        </span>
      </div>

      {/* Tab selector */}
      <div className="flex gap-1 mb-5 p-1 bg-secondary rounded-lg">
        {([
          { key: "cnn", label: "1D-CNN", icon: Cpu },
          { key: "feature", label: "Feature Models", icon: Target },
          { key: "dataset", label: "Dataset", icon: Users },
        ] as const).map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === key
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Icon className="w-3 h-3" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab: CNN */}
      {activeTab === "cnn" && evalData?.cnn_model && (
        <div className="space-y-4">
          <div className="bg-secondary/50 rounded-lg p-4">
            <h3 className="text-sm font-medium text-foreground mb-2">Architecture</h3>
            <p className="text-xs text-muted-foreground font-mono leading-relaxed">
              {evalData.cnn_model.architecture}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-blue-500/5 border border-blue-500/20 rounded-lg p-3 text-center">
              <p className="text-xs text-muted-foreground mb-1">Expected Accuracy</p>
              <p className="text-xl font-bold text-blue-400">
                {evalData.cnn_model.expected_accuracy}
              </p>
            </div>
            <div className="bg-purple-500/5 border border-purple-500/20 rounded-lg p-3 text-center">
              <p className="text-xs text-muted-foreground mb-1">Expected AUC-ROC</p>
              <p className="text-xl font-bold text-purple-400">
                {evalData.cnn_model.expected_auc}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="bg-secondary/50 rounded-lg p-3 text-center">
              <p className="text-xs text-muted-foreground">Input Shape</p>
              <p className="text-sm font-medium text-foreground mt-0.5">400 × 1</p>
            </div>
            <div className="bg-secondary/50 rounded-lg p-3 text-center">
              <p className="text-xs text-muted-foreground">Epochs</p>
              <p className="text-sm font-medium text-foreground mt-0.5">
                {evalData.cnn_model.training_epochs}
              </p>
            </div>
            <div className="bg-secondary/50 rounded-lg p-3 text-center">
              <p className="text-xs text-muted-foreground">Patience</p>
              <p className="text-sm font-medium text-foreground mt-0.5">15</p>
            </div>
          </div>

          <p className="text-xs text-muted-foreground bg-amber-500/5 border border-amber-500/20 rounded-lg p-3">
            <Zap className="w-3 h-3 inline mr-1 text-amber-400" />
            {evalData.cnn_model.note}
          </p>
        </div>
      )}

      {/* Tab: Feature Models */}
      {activeTab === "feature" && evalData?.feature_models && (
        <div className="space-y-4">
          {/* Feature names */}
          <div className="flex flex-wrap gap-1">
            {(evalData.feature_models.features as string[]).map((f) => (
              <span key={f} className="text-xs px-2 py-1 rounded-full bg-secondary text-muted-foreground">
                {f}
              </span>
            ))}
          </div>

          {/* Best model badge */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-foreground">
              Best: <span className="font-semibold text-emerald-400">{String(evalData.feature_models.best_model)}</span>
            </span>
          </div>

          {/* Model metrics table */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 text-muted-foreground font-medium">Model</th>
                  <th className="text-right py-2 text-muted-foreground font-medium">Accuracy</th>
                  <th className="text-right py-2 text-muted-foreground font-medium">AUC</th>
                  <th className="text-right py-2 text-muted-foreground font-medium">F1 (N)</th>
                  <th className="text-right py-2 text-muted-foreground font-medium">F1 (F)</th>
                </tr>
              </thead>
              <tbody>
                {(["random_forest", "svm", "logistic_regression"] as const).map((key) => {
                  const m = evalData.feature_models[key] as Record<string, number>;
                  const name = key === "random_forest" ? "Random Forest"
                    : key === "svm" ? "SVM (RBF)"
                    : "Logistic Reg.";
                  const isBest = (key === "random_forest" && evalData.feature_models.best_model === "Random Forest")
                    || (key === "svm" && evalData.feature_models.best_model === "SVM (RBF)")
                    || (key === "logistic_regression" && evalData.feature_models.best_model === "Logistic Regression");
                  return (
                    <tr key={key} className={`border-b border-border/50 ${isBest ? "bg-primary/5" : ""}`}>
                      <td className="py-2 text-foreground flex items-center gap-1.5">
                        {isBest && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
                        {name}
                      </td>
                      <td className="py-2 text-right font-mono">{(m.accuracy * 100).toFixed(1)}%</td>
                      <td className="py-2 text-right font-mono">{(m.auc_roc * 100).toFixed(1)}%</td>
                      <td className="py-2 text-right font-mono">{(m.f1_fresh * 100).toFixed(1)}%</td>
                      <td className="py-2 text-right font-mono">{(m.f1_fatigued * 100).toFixed(1)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* LOSO CV */}
          {evalData.loso_cv && (
            <div className="bg-secondary/50 rounded-lg p-3">
              <p className="text-xs text-muted-foreground mb-2">
                {evalData.loso_cv.method}
              </p>
              <div className="flex items-center gap-3">
                <span className="text-lg font-bold text-foreground">
                  {(evalData.loso_cv.mean_accuracy * 100).toFixed(1)}%
                </span>
                <span className="text-xs text-muted-foreground">
                  ±{(evalData.loso_cv.std * 100).toFixed(1)}% std
                </span>
              </div>
            </div>
          )}

          {/* Scaler params */}
          {scalerParams && (
            <div className="bg-secondary/50 rounded-lg p-3">
              <p className="text-xs text-muted-foreground mb-2">StandardScaler params loaded</p>
              <p className="text-xs font-mono text-foreground">
                {scalerParams.feature_names.length} features · μ,σ computed from {evalData.dataset.binary_classification} windows
              </p>
            </div>
          )}
        </div>
      )}

      {/* Tab: Dataset */}
      {activeTab === "dataset" && evalData?.dataset && (
        <div className="space-y-4">
          <div className="bg-secondary/50 rounded-lg p-4">
            <h3 className="text-sm font-medium text-foreground mb-1">Source</h3>
            <a
              href={evalData.dataset.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-blue-400 hover:underline"
            >
              Zenodo EMG Fatigue Dataset
            </a>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-secondary/50 rounded-lg p-3 text-center">
              <div className="flex items-center justify-center gap-1.5 mb-1">
                <Users className="w-3.5 h-3.5 text-primary" />
                <p className="text-xs text-muted-foreground">Subjects</p>
              </div>
              <p className="text-xl font-bold text-foreground">
                {evalData.dataset.subjects}
              </p>
            </div>
            <div className="bg-secondary/50 rounded-lg p-3 text-center">
              <div className="flex items-center justify-center gap-1.5 mb-1">
                <Activity className="w-3.5 h-3.5 text-primary" />
                <p className="text-xs text-muted-foreground">Sample Rate</p>
              </div>
              <p className="text-xl font-bold text-foreground">
                {evalData.dataset.sampling_rate_hz} Hz
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-secondary/50 rounded-lg p-3 text-center">
              <p className="text-xs text-muted-foreground">Windows</p>
              <p className="text-lg font-bold text-foreground">
                {evalData.dataset.binary_classification}
              </p>
            </div>
            <div className="bg-secondary/50 rounded-lg p-3 text-center">
              <p className="text-xs text-muted-foreground">Channels</p>
              <p className="text-lg font-bold text-foreground">8</p>
            </div>
          </div>

          {/* Class distribution */}
          <div className="bg-secondary/50 rounded-lg p-3">
            <p className="text-xs text-muted-foreground mb-2">Class Distribution</p>
            <div className="space-y-2">
              {(["fresh", "fatigued"] as const).map((cls, i) => {
                const count = evalData.dataset.class_distribution[cls];
                const total = evalData.dataset.binary_classification;
                const pct = total > 0 ? (count / total) * 100 : 0;
                return (
                  <div key={cls}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className={`font-medium ${i === 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {CLASS_LABELS[i]}
                      </span>
                      <span className="text-muted-foreground">{count} ({pct.toFixed(1)}%)</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${i === 0 ? "bg-emerald-400" : "bg-red-400"}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Inference speed */}
          {evalData.loso_cv && (
            <div className="flex items-center gap-2 p-3 bg-secondary/50 rounded-lg text-xs text-muted-foreground">
              <FlaskConical className="w-3.5 h-3.5 text-primary" />
              <span>RF Inference: ~15ms/prediction · CNN (TF.js): ~5ms/prediction</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
