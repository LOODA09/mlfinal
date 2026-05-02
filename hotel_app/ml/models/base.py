from __future__ import annotations

from typing import Any
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class BaseHotelModel:
    name = "Base Model"

    def get_estimator(self) -> Any:
        raise NotImplementedError

    def build_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        return Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", self.get_estimator())])
