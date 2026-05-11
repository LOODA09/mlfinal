from typing import Any
import importlib
from hotel_app.ml.models.base import BaseHotelModel


class XGBoostModel(BaseHotelModel):
    """XGBoost classifier with default parameters.

    Doctor-facing notes:
    - estimator: ``XGBClassifier``
    - probability path: native ``predict_proba``
    - uses default hyperparameters matching the project notebook
    """
    name = "XGBoost"

    def get_estimator(self) -> Any:
        try:
            xgboost = importlib.import_module("xgboost")
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for XGBoostModel. Install it with `pip install xgboost` "
                "or remove XGBoost from the selected models."
            ) from exc
        return xgboost.XGBClassifier(
            random_state=42,
            n_jobs=1,
        )
