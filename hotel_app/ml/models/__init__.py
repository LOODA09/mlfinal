from typing import Dict, Type

from hotel_app.ml.models.base import BaseHotelModel
from hotel_app.ml.models.logistic import LogisticRegressionModel
from hotel_app.ml.models.knn import KNNModel
from hotel_app.ml.models.decision_tree import DecisionTreeModel
from hotel_app.ml.models.svm import SVMModel
from hotel_app.ml.models.random_forest import RandomForestModel
from hotel_app.ml.models.xgboost_model import XGBoostModel
from hotel_app.ml.models.ann import ANNModel
from hotel_app.ml.models.lstm import LSTMModel
from hotel_app.ml.models.rnn import RNNModel

MODEL_REGISTRY: Dict[str, Type[BaseHotelModel]] = {
    model.name: model
    for model in (
        LogisticRegressionModel,
        KNNModel,
        DecisionTreeModel,
        SVMModel,
        RandomForestModel,
        XGBoostModel,
        ANNModel,
        LSTMModel,
        RNNModel,
    )
}

__all__ = [
    "BaseHotelModel",
    "LogisticRegressionModel",
    "KNNModel",
    "DecisionTreeModel",
    "SVMModel",
    "RandomForestModel",
    "XGBoostModel",
    "ANNModel",
    "LSTMModel",
    "RNNModel",
    "MODEL_REGISTRY"
]
