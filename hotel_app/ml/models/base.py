from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


class BaseHotelModel:
    """Base interface for every project model class.

    Each concrete model class is responsible for returning a configured
    estimator or search object through ``get_estimator()``. The shared
    pipeline then prepends the project preprocessor so every model is
    trained on the same transformed feature space.
    """
    name = "Base Model"

    def get_estimator(self) -> Any:
        raise NotImplementedError

    def build_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        return Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", self.get_estimator())])


class BalancedClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Apply consistent balancing for models that need wrapper support.

    Some estimators support ``sample_weight`` natively, while others need
    simple oversampling to avoid the majority class dominating training.
    This wrapper keeps that logic in one place so the individual model
    classes stay small and readable.
    """
    def __init__(self, estimator: Any, strategy: str = "sample_weight", random_state: int = 42) -> None:
        self.estimator = estimator
        self.strategy = strategy
        self.random_state = random_state

    def fit(self, x_data: Any, y_data: Any) -> "BalancedClassifierWrapper":
        y_array = np.asarray(y_data)
        self.classes_ = np.unique(y_array)
        x_fit = x_data
        y_fit = y_array
        if self.strategy in {"oversample", "hybrid"}:
            x_fit, y_fit = self._oversample(x_data, y_array)
        fit_kwargs = {}
        supports_sample_weight = "sample_weight" in inspect.signature(self.estimator.fit).parameters
        if self.strategy in {"sample_weight", "hybrid"} and supports_sample_weight:
            fit_kwargs["sample_weight"] = compute_sample_weight(class_weight="balanced", y=np.asarray(y_fit))
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x_fit, y_fit, **fit_kwargs)
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def _oversample(self, x_data: Any, y_data: np.ndarray) -> tuple[Any, np.ndarray]:
        labels, counts = np.unique(y_data, return_counts=True)
        if len(labels) < 2 or counts.min() == counts.max():
            return x_data, y_data
        rng = np.random.default_rng(self.random_state)
        target = int(counts.max())
        sampled_indices = []
        for label in labels:
            label_indices = np.flatnonzero(y_data == label)
            extra = rng.choice(label_indices, size=target, replace=True)
            sampled_indices.append(extra)
        combined = np.concatenate(sampled_indices)
        rng.shuffle(combined)
        if isinstance(x_data, pd.DataFrame):
            return x_data.iloc[combined].reset_index(drop=True), y_data[combined]
        if isinstance(x_data, pd.Series):
            return x_data.iloc[combined].reset_index(drop=True), y_data[combined]
        array = np.asarray(x_data)
        return array[combined], y_data[combined]

    def predict(self, x_data: Any) -> Any:
        return self.estimator_.predict(x_data)

    def predict_proba(self, x_data: Any) -> Any:
        return self.estimator_.predict_proba(x_data)

    def decision_function(self, x_data: Any) -> Any:
        return self.estimator_.decision_function(x_data)
