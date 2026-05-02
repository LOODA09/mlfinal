from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


class EvaluationMetrics:
    metric_names = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "mcc",
    )

    @staticmethod
    def evaluate(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_score: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }
        if y_score is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score)
            except ValueError:
                metrics["roc_auc"] = np.nan
            try:
                metrics["average_precision"] = average_precision_score(y_true, y_score)
            except ValueError:
                metrics["average_precision"] = np.nan
            try:
                metrics["brier_score"] = brier_score_loss(y_true, y_score)
            except ValueError:
                metrics["brier_score"] = np.nan
            try:
                metrics["log_loss"] = log_loss(y_true, y_score)
            except ValueError:
                metrics["log_loss"] = np.nan
        else:
            metrics["roc_auc"] = np.nan
            metrics["average_precision"] = np.nan
            metrics["brier_score"] = np.nan
            metrics["log_loss"] = np.nan
        return metrics

    @staticmethod
    def report(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_score: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        return {
            "metrics": EvaluationMetrics.evaluate(y_true, y_pred, y_score),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
        }
