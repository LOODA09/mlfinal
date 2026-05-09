from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from .data import _positive_probabilities
from .metrics import EvaluationMetrics


class ModelTester:
    @staticmethod
    def _prefix_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}

    def test_model(
        self,
        name: str,
        model: Pipeline,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        train_pred = model.predict(x_train)
        train_score = _positive_probabilities(model, x_train)
        y_pred = model.predict(x_test)
        y_score = _positive_probabilities(model, x_test)
        result = EvaluationMetrics.report(y_test, y_pred, y_score)
        result["train_metrics"] = EvaluationMetrics.evaluate(y_train, train_pred, train_score)
        result["metrics"].update(self._prefix_metrics(result["train_metrics"], "train"))
        result["model"] = name
        return result

    def test_many(
        self,
        models: Dict[str, Pipeline],
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        details = {
            name: self.test_model(name, model, x_train, y_train, x_test, y_test)
            for name, model in models.items()
        }
        summary = pd.DataFrame([dict(model=name, **details[name]["metrics"]) for name in details]).sort_values(
            "f1", ascending=False
        )
        return summary, details
