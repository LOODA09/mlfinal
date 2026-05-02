from __future__ import annotations

from typing import Any, Sequence
import importlib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KerasTabularClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model_type: str = "ann",
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        random_state: int = 42,
        verbose: int = 0,
    ) -> None:
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self, n_features: int) -> Any:
        try:
            tf = importlib.import_module("tensorflow")
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for ANNModel and RNNModel. "
                "Install it with `pip install tensorflow` or remove deep models from the run."
            ) from exc
        tf.random.set_seed(self.random_state)
        model = tf.keras.Sequential()
        if self.model_type == "rnn":
            model.add(tf.keras.layers.Input(shape=(n_features, 1)))
            model.add(tf.keras.layers.SimpleRNN(64, activation="tanh"))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(32, activation="relu"))
        else:
            model.add(tf.keras.layers.Input(shape=(n_features,)))
            model.add(tf.keras.layers.Dense(75, activation="relu"))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(75, activation="relu"))
            model.add(tf.keras.layers.Dense(50, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def _reshape(self, x_data: Any) -> np.ndarray:
        array = np.asarray(x_data, dtype=np.float32)
        if self.model_type == "rnn":
            return array.reshape(array.shape[0], array.shape[1], 1)
        return array

    def fit(self, x_data: Any, y_data: Sequence[int]) -> "KerasTabularClassifier":
        if hasattr(x_data, "toarray"):
            x_data = x_data.toarray()
        array = np.asarray(x_data, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = array.shape[1]
        self.model_ = self._build_model(self.n_features_in_)
        self.history_ = self.model_.fit(
            self._reshape(array),
            np.asarray(y_data, dtype=np.float32),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=self.verbose,
        )
        return self

    def predict_proba(self, x_data: Any) -> np.ndarray:
        if hasattr(x_data, "toarray"):
            x_data = x_data.toarray()
        probabilities = self.model_.predict(self._reshape(x_data), verbose=0).ravel()
        return np.vstack([1 - probabilities, probabilities]).T

    def predict(self, x_data: Any) -> np.ndarray:
        return (self.predict_proba(x_data)[:, 1] >= 0.5).astype(int)
