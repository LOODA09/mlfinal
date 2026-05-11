from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


MODEL_DESCRIPTIONS = {
    "ANN": "Feed-forward neural network used as a dense deep-learning classifier.",
    "LSTM": "Long short-term memory network trained on reshaped tabular sequences with TensorFlow.",
    "KNN": "Distance-based classifier that predicts from nearby training examples.",
    "Decision Tree": "Rule-based tree learner that splits bookings into cancellation patterns.",
    "Logistic Regression": "Linear probabilistic baseline using the logistic sigmoid for binary classification.",
    "Random Forest": "Bagged ensemble of decision trees with randomized feature selection.",
    "SVM": "RBF-kernel support vector machine trained on a bounded stratified sample for practicality.",
    "XGBoost": "Optimized gradient boosting tree method with regularized boosting.",
    "RNN": "Recurrent neural network trained on reshaped tabular sequences with TensorFlow.",
}


def _format_duration(seconds: object) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "N/A"
    if value < 60:
        return f"{value:.1f}s"
    minutes, remainder = divmod(int(round(value)), 60)
    if minutes < 60:
        return f"{minutes}m {remainder}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


class BenchmarkPdfBuilder:
    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.reports_dir = self.artifacts_dir / "reports"
        self.output_path = self.reports_dir / "model_evaluation_report.pdf"

    def build(self) -> Path:
        holdout = pd.read_csv(self.reports_dir / "holdout_summary.csv")
        cv_results = self._safe_read_csv(self.reports_dir / "cross_validation_results.csv")
        metadata = json.loads((self.reports_dir / "metadata.json").read_text(encoding="utf-8-sig"))

        with PdfPages(self.output_path) as pdf:
            pdf.savefig(self._cover_page(metadata))
            pdf.savefig(self._holdout_page(holdout))
            pdf.savefig(self._cv_page(cv_results))
            pdf.savefig(self._narrative_page(holdout))

        plt.close("all")
        return self.output_path

    @staticmethod
    def _safe_read_csv(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return pd.DataFrame()

    def _cover_page(self, metadata: dict) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = [
            "Hotel Cancellation Prediction",
            "",
            f"Best benchmark model: {metadata.get('best_model', 'N/A')}",
            f"Deployment artifact: {metadata.get('deployment_model', metadata.get('best_model', 'N/A'))}",
            f"Train / test split: {int(metadata.get('train_ratio', 0.7) * 100)}% / {int(metadata.get('test_ratio', 0.3) * 100)}%",
            f"Cross-validation: {metadata.get('cross_validation_folds', 'N/A')}-fold",
            f"Runtime: Python {metadata.get('python_version', 'N/A')} / TensorFlow {metadata.get('tensorflow_version', 'off')}",
            f"Pipeline wall clock: {_format_duration(metadata.get('total_pipeline_wall_clock_sec'))}",
            "",
            metadata.get(
                "pipeline_wall_clock_note",
                "This report summarizes how each trained model behaved on the dataset using the saved benchmark artifacts.",
            ),
        ]
        ax.text(0.08, 0.92, "\n".join(lines), va="top", ha="left", fontsize=14)
        return fig

    def _holdout_page(self, holdout: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        display = holdout.copy()
        if "complexity_tier" not in display.columns:
            display["complexity_tier"] = "Derived in dashboard"
        cols = [
            c
            for c in [
                "model",
                "train_accuracy",
                "accuracy",
                "train_precision",
                "precision",
                "train_recall",
                "recall",
                "train_f1",
                "f1",
                "train_roc_auc",
                "roc_auc",
                "benchmark_training_time_sec",
                "full_data_training_time_sec",
                "training_time_sec",
                "inference_ms_per_row",
                "complexity_tier",
            ]
            if c in display.columns
        ]
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
        if cv_results.empty or "fold" not in cv_results.columns:
            ax.text(
                0.08,
                0.88,
                "Cross-validation was skipped for this saved run, so no CV rows are available.",
                fontsize=13,
                va="top",
            )
            ax.set_title("Cross-Validation Mean Scores", fontsize=14, pad=18)
            return fig
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
                f"{row.model}: {description} Benchmark train accuracy {getattr(row, 'train_accuracy', float('nan')):.4f}, "
                f"holdout accuracy {row.accuracy:.4f}, precision {row.precision:.4f}, recall {row.recall:.4f}, "
                f"F1 {row.f1:.4f}, ROC-AUC {row.roc_auc:.4f}, benchmark fit {getattr(row, 'benchmark_training_time_sec', float('nan')):.2f}s, "
                f"full-data retrain {getattr(row, 'full_data_training_time_sec', float('nan')):.2f}s, total saved run cost {row.training_time_sec:.2f}s, "
                f"inference {row.inference_ms_per_row:.4f} ms/row."
            )
            wrapped = fill(line, width=90)
            ax.text(0.06, y, wrapped, fontsize=9, va="top")
            y -= 0.08 + 0.02 * wrapped.count("\n")
            if y < 0.08:
                break
        return fig
