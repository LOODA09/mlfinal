from __future__ import annotations

from typing import Any, Sequence
import importlib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight


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
                "TensorFlow is required for ANNModel, RNNModel, and LSTMModel. "
                "Install it with `pip install tensorflow` or remove deep models from the run."
            ) from exc
        tf.random.set_seed(self.random_state)
        regularizer = tf.keras.regularizers.l2(1e-4)
        model = tf.keras.Sequential()
        if self.model_type == "rnn":
            model.add(tf.keras.layers.Input(shape=(n_features, 1)))
            model.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.SimpleRNN(
                        40,
                        activation="tanh",
                        dropout=0.1,
                        recurrent_dropout=0.1,
                    )
                )
            )
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(0.25))
            model.add(tf.keras.layers.Dense(24, activation="relu", kernel_regularizer=regularizer))
        elif self.model_type == "lstm":
            model.add(tf.keras.layers.Input(shape=(n_features, 1)))
            model.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        32,
                        activation="tanh",
                        dropout=0.1,
                        recurrent_dropout=0.1,
                    )
                )
            )
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(0.25))
            model.add(tf.keras.layers.Dense(24, activation="relu", kernel_regularizer=regularizer))
        else:
            model.add(tf.keras.layers.Input(shape=(n_features,)))
            model.add(tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(0.25))
            model.add(tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(0.15))
            model.add(tf.keras.layers.Dense(24, activation="relu", kernel_regularizer=regularizer))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        return model

    def _reshape(self, x_data: Any) -> np.ndarray:
        array = np.asarray(x_data, dtype=np.float32)
        if self.model_type in {"rnn", "lstm"}:
            return array.reshape(array.shape[0], array.shape[1], 1)
        return array

    def fit(self, x_data: Any, y_data: Sequence[int]) -> "KerasTabularClassifier":
        if hasattr(x_data, "toarray"):
            x_data = x_data.toarray()
        array = np.asarray(x_data, dtype=np.float32)
        y_array = np.asarray(y_data, dtype=np.int32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = array.shape[1]
        self.model_ = self._build_model(self.n_features_in_)
        tf = importlib.import_module("tensorflow")
        present_classes = np.unique(y_array)
        class_weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_array)
        class_weight_map = {int(label): float(weight) for label, weight in zip(present_classes, class_weights)}
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=6,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
            ),
        ]
        self.history_ = self.model_.fit(
            self._reshape(array),
            y_array.astype(np.float32),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            class_weight=class_weight_map,
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
