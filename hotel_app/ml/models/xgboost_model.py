from typing import Any
import importlib
from hotel_app.ml.models.base import BaseHotelModel


class XGBoostModel(BaseHotelModel):
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
            booster="gbtree",
            learning_rate=0.05,
            max_depth=6,
            n_estimators=320,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
