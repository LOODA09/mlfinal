from hotel_app.ml.models.base import BaseHotelModel
from hotel_app.ml.deep import KerasTabularClassifier


class RNNModel(BaseHotelModel):
    """Simple recurrent neural network for reshaped tabular sequences.

    Doctor-facing notes:
    - estimator: TensorFlow ``KerasTabularClassifier`` in RNN mode
    - probability path: final dense layer uses ``sigmoid`` activation
    - balancing: class-weighted TensorFlow fit
    - tuning: epochs and batch size are exposed through the model class
    """
    name = "RNN"

    def __init__(self, epochs: int = 45, batch_size: int = 192) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

    def get_estimator(self) -> KerasTabularClassifier:
        return KerasTabularClassifier(model_type="rnn", epochs=self.epochs, batch_size=self.batch_size)
