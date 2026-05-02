from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


MODEL_DESCRIPTIONS = {
    "ANN": "Feed-forward neural network used as a dense deep-learning classifier.",
    "KNN": "Distance-based classifier that predicts from nearby training examples.",
    "Decision Tree": "Rule-based tree learner that splits bookings into cancellation patterns.",
    "Random Forest": "Bagged ensemble of decision trees with randomized feature selection.",
    "Naive Bayes": "Probabilistic classifier with conditional independence assumptions.",
    "SVM": "Margin-based classifier calibrated for probability-style output.",
    "Gradient Boosting": "Sequential boosted trees focusing on previous residual errors.",
    "Extra Trees": "Highly randomized tree ensemble reducing variance through random splits.",
    "XGBoost": "Optimized gradient boosting tree method with regularized boosting.",
    "Voting Ensemble": "Soft-voting ensemble that averages probabilities from several models.",
    "Stacking Ensemble": "Meta-ensemble combining strong base learners through a final estimator.",
    "RNN": "Recurrent neural network trained on reshaped tabular sequences with TensorFlow.",
}


class BenchmarkPdfBuilder:
    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.reports_dir = self.artifacts_dir / "reports"
        self.output_path = self.reports_dir / "model_evaluation_report.pdf"

    def build(self) -> Path:
        holdout = pd.read_csv(self.reports_dir / "holdout_summary.csv")
        cv_results = pd.read_csv(self.reports_dir / "cross_validation_results.csv")
        metadata = json.loads((self.reports_dir / "metadata.json").read_text(encoding="utf-8"))

        with PdfPages(self.output_path) as pdf:
            pdf.savefig(self._cover_page(metadata))
            pdf.savefig(self._holdout_page(holdout))
            pdf.savefig(self._cv_page(cv_results))
            pdf.savefig(self._narrative_page(holdout))

        plt.close("all")
        return self.output_path

    def _cover_page(self, metadata: dict) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = [
            "Hotel Cancellation Prediction",
            "",
            f"Best benchmark model: {metadata.get('best_model', 'N/A')}",
            f"Train / test split: {int(metadata.get('train_ratio', 0.7) * 100)}% / {int(metadata.get('test_ratio', 0.3) * 100)}%",
            f"Cross-validation: {metadata.get('cross_validation_folds', 'N/A')}-fold",
            f"Runtime: Python {metadata.get('python_version', 'N/A')} / TensorFlow {metadata.get('tensorflow_version', 'off')}",
            "",
            "This report summarizes how each trained model behaved on the dataset using the saved benchmark artifacts.",
        ]
        ax.text(0.08, 0.92, "\n".join(lines), va="top", ha="left", fontsize=14)
        return fig

    def _holdout_page(self, holdout: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        display = holdout.copy()
        if "complexity_tier" not in display.columns:
            display["complexity_tier"] = "Derived in dashboard"
        cols = [c for c in ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "training_time_sec", "inference_ms_per_row", "complexity_tier"] if c in display.columns]
        table = ax.table(
            cellText=display[cols].round(4).astype(str).values,
            colLabels=cols,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.4)
        ax.set_title("Holdout Evaluation Summary", fontsize=14, pad=18)
        return fig

    def _cv_page(self, cv_results: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        means = cv_results[cv_results["fold"].astype(str) == "mean"].copy()
        cols = [c for c in ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"] if c in means.columns]
        table = ax.table(
            cellText=means[cols].round(4).astype(str).values,
            colLabels=cols,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax.set_title("Cross-Validation Mean Scores", fontsize=14, pad=18)
        return fig

    def _narrative_page(self, holdout: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        top = 0.97
        ax.text(0.06, top, "Model-by-model notes", fontsize=16, va="top")
        y = top - 0.05
        for row in holdout.itertuples(index=False):
            description = MODEL_DESCRIPTIONS.get(row.model, "Model description not available.")
            line = (
                f"{row.model}: {description} Holdout accuracy {row.accuracy:.4f}, "
                f"precision {row.precision:.4f}, recall {row.recall:.4f}, F1 {row.f1:.4f}, "
                f"ROC-AUC {row.roc_auc:.4f}, training time {row.training_time_sec:.2f}s, "
                f"inference {row.inference_ms_per_row:.4f} ms/row."
            )
            wrapped = fill(line, width=90)
            ax.text(0.06, y, wrapped, fontsize=9, va="top")
            y -= 0.08 + 0.02 * wrapped.count("\n")
            if y < 0.08:
                break
        return fig
