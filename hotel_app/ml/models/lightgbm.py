from typing import Any
import importlib

from hotel_app.ml.models.base import BaseHotelModel


class LightGBMModel(BaseHotelModel):
    name = "LightGBM"

    def get_estimator(self) -> Any:
        try:
            lightgbm = importlib.import_module("lightgbm")
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required for LightGBMModel. Install it with `pip install lightgbm` "
                "or remove LightGBM from the selected models."
            ) from exc
        return lightgbm.LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            n_estimators=400,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            class_weight="balanced",
            verbosity=-1,
        )
