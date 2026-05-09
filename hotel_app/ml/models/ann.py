from typing import Any
import importlib
from sklearn.neural_network import MLPClassifier
from hotel_app.ml.models.base import BaseHotelModel, BalancedClassifierWrapper
from hotel_app.ml.deep import KerasTabularClassifier


class ANNModel(BaseHotelModel):
    name = "ANN"

    def __init__(self, epochs: int = 120, batch_size: int = 192) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

    def get_estimator(self) -> Any:
        try:
            importlib.import_module("tensorflow")
            return KerasTabularClassifier(model_type="ann", epochs=self.epochs, batch_size=self.batch_size)
        except ImportError:
            return BalancedClassifierWrapper(
                MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="relu",
                    learning_rate_init=0.001,
                    batch_size=min(self.batch_size, 256),
                    max_iter=self.epochs,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                ),
                strategy="oversample",
                random_state=42,
            )
