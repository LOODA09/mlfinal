from hotel_app.ml.models.base import BaseHotelModel
from hotel_app.ml.deep import KerasTabularClassifier


class LSTMModel(BaseHotelModel):
    name = "LSTM"

    def __init__(self, epochs: int = 20, batch_size: int = 256) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

    def get_estimator(self) -> KerasTabularClassifier:
        return KerasTabularClassifier(model_type="lstm", epochs=self.epochs, batch_size=self.batch_size)
