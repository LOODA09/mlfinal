from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type
import importlib
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore", category=FutureWarning)


MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def _positive_probabilities(model: Any, x_data: pd.DataFrame) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_data)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_data)
        return 1 / (1 + np.exp(-scores))
    return None


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _count_model_complexity(estimator: Any) -> int:
    if hasattr(estimator, "tree_"):
        return int(getattr(estimator.tree_, "node_count", 0))
    if hasattr(estimator, "estimators_"):
        total = 0
        for inner in estimator.estimators_:
            if hasattr(inner, "tree_"):
                total += int(getattr(inner.tree_, "node_count", 0))
            else:
                total += 1
        return total
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        intercept = np.asarray(getattr(estimator, "intercept_", []))
        return int(coef.size + intercept.size)
    if hasattr(estimator, "coefs_"):
        total = sum(np.asarray(weights).size for weights in estimator.coefs_)
        total += sum(np.asarray(bias).size for bias in getattr(estimator, "intercepts_", []))
        return int(total)
    if hasattr(estimator, "support_vectors_"):
        return int(np.asarray(estimator.support_vectors_).shape[0])
    if hasattr(estimator, "n_features_in_"):
        return int(estimator.n_features_in_)
    return 0


@dataclass
class HotelDataProcessor:
    target_column: str = "is_canceled"
    dropped_low_signal_columns: Sequence[str] = field(
        default_factory=tuple
    )
    leakage_columns: Sequence[str] = field(
        default_factory=lambda: ("reservation_status", "reservation_status_date")
    )

    def load_data(self, path: str = "hotel_bookings.csv") -> pd.DataFrame:
        return pd.read_csv(path)

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        if "adr" in df.columns:
            df = df[df["adr"] >= 0]

        guest_columns = [col for col in ("children", "adults", "babies") if col in df.columns]
        if guest_columns:
            df = df[df[guest_columns].sum(axis=1) > 0]

        object_columns = df.select_dtypes(include=["object"]).columns
        numeric_columns = df.select_dtypes(exclude=["object"]).columns
        df[object_columns] = df[object_columns].fillna("Unknown")
        df[numeric_columns] = df[numeric_columns].fillna(0)
        return df.reset_index(drop=True)

    def build_features(
        self,
        data: pd.DataFrame,
        remove_leakage_features: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.clean_data(data)
        y = df[self.target_column].astype(int)
        x_data = self.add_engineered_features(
            self.build_raw_prediction_inputs(df, remove_leakage_features=remove_leakage_features)
        )
        return x_data, y

    def build_raw_prediction_inputs(
        self,
        data: pd.DataFrame,
        remove_leakage_features: bool = True,
    ) -> pd.DataFrame:
        df = self.clean_data(data)

        if not remove_leakage_features and "reservation_status_date" in df.columns:
            reservation_date = pd.to_datetime(df["reservation_status_date"], errors="coerce")
            df["reservation_status_year"] = reservation_date.dt.year.fillna(0).astype(int)
            df["reservation_status_month"] = reservation_date.dt.month.fillna(0).astype(int)
            df["reservation_status_day"] = reservation_date.dt.day.fillna(0).astype(int)
            df = df.drop(columns=["reservation_status_date"])

        drop_columns = list(self.dropped_low_signal_columns)
        if remove_leakage_features:
            drop_columns.extend(self.leakage_columns)
        drop_columns.append(self.target_column)
        return df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")

    def add_engineered_features(self, x_data: pd.DataFrame) -> pd.DataFrame:
        features = x_data.copy()

        if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(features.columns):
            features["total_nights"] = (
                features["stays_in_weekend_nights"] + features["stays_in_week_nights"]
            )

        if {"adults", "children", "babies"}.issubset(features.columns):
            features["total_guests"] = features["adults"] + features["children"] + features["babies"]

        if {"previous_cancellations", "previous_bookings_not_canceled"}.issubset(features.columns):
            previous_total = (
                features["previous_cancellations"] + features["previous_bookings_not_canceled"]
            )
            features["previous_cancel_rate"] = (
                features["previous_cancellations"] / previous_total.replace(0, 1)
            ).fillna(0)

        if {"booking_changes", "lead_time"}.issubset(features.columns):
            features["changes_per_lead_day"] = (
                features["booking_changes"] / features["lead_time"].replace(0, 1)
            ).fillna(0)

        if {"reserved_room_type", "assigned_room_type"}.issubset(features.columns):
            features["room_match"] = (
                features["reserved_room_type"].astype(str) == features["assigned_room_type"].astype(str)
            ).astype(int)

        if {"adults", "children", "babies"}.issubset(features.columns):
            features["family_booking"] = (
                (features["children"] + features["babies"]) > 0
            ).astype(int)

        if {"total_of_special_requests", "total_nights"}.issubset(features.columns):
            features["requests_per_night"] = (
                features["total_of_special_requests"] / features["total_nights"].replace(0, 1)
            ).fillna(0)

        if {"adr", "total_guests"}.issubset(features.columns):
            features["adr_per_guest"] = (
                features["adr"] / features["total_guests"].replace(0, 1)
            ).fillna(0)

        if "lead_time" in features.columns:
            features["lead_time_log"] = np.log1p(features["lead_time"])

        return features

    def build_preprocessor(self, x_data: pd.DataFrame) -> ColumnTransformer:
        categorical_columns = list(x_data.select_dtypes(include=["object", "category"]).columns)
        numeric_columns = [col for col in x_data.columns if col not in categorical_columns]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _one_hot_encoder()),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )


class NotebookEDAAnalyzer:
    def __init__(self, data: pd.DataFrame, processor: Optional[HotelDataProcessor] = None) -> None:
        self.raw_data = data.copy()
        self.processor = processor or HotelDataProcessor()
        self.clean_data = self.processor.clean_data(data)

    def preview(self, rows: int = 5) -> pd.DataFrame:
        return self.raw_data.head(rows)

    def summary_statistics(self) -> pd.DataFrame:
        return self.raw_data.describe(include="all").T

    def info_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dtype": self.raw_data.dtypes.astype(str),
                "non_null": self.raw_data.notna().sum(),
                "null_values": self.raw_data.isna().sum(),
                "null_percentage": self.raw_data.isna().mean() * 100,
                "unique_values": self.raw_data.nunique(dropna=True),
            }
        )

    def null_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Null Values": self.raw_data.isna().sum(),
                "Percentage Null Values": self.raw_data.isna().mean() * 100,
            }
        ).sort_values("Null Values", ascending=False)

    def negative_adr_rows(self) -> pd.DataFrame:
        if "adr" not in self.raw_data.columns:
            return pd.DataFrame()
        return self.raw_data[self.raw_data["adr"] < 0]

    def empty_guest_bookings(self) -> pd.DataFrame:
        guest_columns = [col for col in ("children", "adults", "babies") if col in self.raw_data.columns]
        if not guest_columns:
            return pd.DataFrame()
        return self.raw_data[self.raw_data[guest_columns].fillna(0).sum(axis=1) == 0]

    def country_wise_guests(self) -> pd.DataFrame:
        if not {"is_canceled", "country"}.issubset(self.clean_data.columns):
            return pd.DataFrame()
        country_data = (
            self.clean_data[self.clean_data["is_canceled"] == 0]["country"]
            .value_counts()
            .reset_index()
        )
        country_data.columns = ["country", "No of guests"]
        return country_data

    def country_choropleth(self) -> Any:
        import plotly.express as px

        country_data = self.country_wise_guests()
        return px.choropleth(
            country_data,
            locations="country",
            color="No of guests",
            hover_name="country",
            color_continuous_scale="Viridis",
        )

    def room_price_by_month(self) -> pd.DataFrame:
        required = {"is_canceled", "hotel", "arrival_date_month", "adr"}
        if not required.issubset(self.clean_data.columns):
            return pd.DataFrame()

        not_canceled = self.clean_data[self.clean_data["is_canceled"] == 0]
        prices = (
            not_canceled.groupby(["arrival_date_month", "hotel"])["adr"]
            .mean()
            .reset_index()
            .pivot(index="arrival_date_month", columns="hotel", values="adr")
            .reset_index()
            .rename(columns={"arrival_date_month": "month"})
        )
        return self._sort_months(prices, "month")

    def room_price_figure(self) -> Any:
        import plotly.express as px

        prices = self.room_price_by_month()
        value_columns = [col for col in prices.columns if col != "month"]
        return px.line(
            prices,
            x="month",
            y=value_columns,
            title="Room price per night over the months",
            labels={"value": "Price", "month": "Month"},
        )

    def monthly_guest_counts(self) -> pd.DataFrame:
        required = {"is_canceled", "hotel", "arrival_date_month"}
        if not required.issubset(self.clean_data.columns):
            return pd.DataFrame()

        not_canceled = self.clean_data[self.clean_data["is_canceled"] == 0]
        guests = (
            not_canceled.groupby(["arrival_date_month", "hotel"])
            .size()
            .reset_index(name="Number of guests")
            .pivot(index="arrival_date_month", columns="hotel", values="Number of guests")
            .reset_index()
            .rename(columns={"arrival_date_month": "month"})
        )
        return self._sort_months(guests.fillna(0), "month")

    def monthly_guest_figure(self) -> Any:
        import plotly.express as px

        guests = self.monthly_guest_counts()
        value_columns = [col for col in guests.columns if col != "month"]
        return px.line(
            guests,
            x="month",
            y=value_columns,
            title="Total number of guests per month",
            labels={"value": "Number of Guests", "month": "Month"},
        )

    def price_vs_guests_figure(self) -> Any:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        prices = self.room_price_by_month()
        guests = self.monthly_guest_counts()
        merged = prices.merge(guests, on="month", suffixes=("_price", "_guests"))

        figure = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"secondary_y": True}, {"secondary_y": True}]],
            subplot_titles=("Resort: Price vs Guests", "City: Price vs Guests"),
        )
        hotel_pairs = [
            ("Resort Hotel", 1),
            ("City Hotel", 2),
        ]

        for hotel_name, column_index in hotel_pairs:
            price_column = f"{hotel_name}_price"
            guest_column = f"{hotel_name}_guests"
            if price_column not in merged.columns or guest_column not in merged.columns:
                continue
            figure.add_trace(
                go.Scatter(x=merged["month"], y=merged[price_column], mode="lines", name=f"{hotel_name} Price"),
                row=1,
                col=column_index,
                secondary_y=False,
            )
            figure.add_trace(
                go.Scatter(x=merged["month"], y=merged[guest_column], mode="lines", name=f"{hotel_name} Guests"),
                row=1,
                col=column_index,
                secondary_y=True,
            )

        figure.update_layout(title="Price vs Guests")
        return figure

    def stay_distribution(self) -> pd.DataFrame:
        required = {
            "is_canceled",
            "hotel",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
        }
        if not required.issubset(self.clean_data.columns):
            return pd.DataFrame()

        not_canceled = self.clean_data[self.clean_data["is_canceled"] == 0].copy()
        not_canceled["total_nights"] = (
            not_canceled["stays_in_weekend_nights"] + not_canceled["stays_in_week_nights"]
        )
        return (
            not_canceled.groupby(["total_nights", "hotel"])
            .size()
            .reset_index(name="Number of stays")
            .sort_values(["total_nights", "hotel"])
        )

    def stay_distribution_figure(self) -> Any:
        import plotly.express as px

        stays = self.stay_distribution()
        return px.bar(
            stays,
            x="total_nights",
            y="Number of stays",
            color="hotel",
            barmode="group",
            title="Length of stay by hotel type",
        )

    def numeric_correlation(self) -> pd.DataFrame:
        df = self.clean_data.copy()
        if "hotel" in df.columns:
            df["Hotel_Type"] = np.where(df["hotel"] == "City Hotel", 1, 0)
        numeric_data = df.select_dtypes(exclude=["object"])
        return numeric_data.corr()

    def target_correlation(self) -> pd.Series:
        correlation = self.numeric_correlation()
        if self.processor.target_column not in correlation.columns:
            return pd.Series(dtype=float)
        return correlation[self.processor.target_column].abs().sort_values(ascending=False)

    def correlation_heatmap(self) -> Any:
        import matplotlib.pyplot as plt
        import seaborn as sns

        figure, axis = plt.subplots(figsize=(18, 9))
        sns.heatmap(self.numeric_correlation(), cmap="viridis", linewidths=0.5, ax=axis)
        axis.set_title("Numeric Feature Correlation")
        figure.tight_layout()
        return figure

    @staticmethod
    def _sort_months(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
        sorted_data = data.copy()
        sorted_data[column_name] = pd.Categorical(
            sorted_data[column_name],
            categories=MONTH_ORDER,
            ordered=True,
        )
        return sorted_data.sort_values(column_name).reset_index(drop=True)


class EvaluationMetrics:
    metric_names = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "mcc",
    )

    @staticmethod
    def evaluate(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_score: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }

        if y_score is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score)
            except ValueError:
                metrics["roc_auc"] = np.nan
            try:
                metrics["average_precision"] = average_precision_score(y_true, y_score)
            except ValueError:
                metrics["average_precision"] = np.nan
            try:
                metrics["brier_score"] = brier_score_loss(y_true, y_score)
            except ValueError:
                metrics["brier_score"] = np.nan
            try:
                metrics["log_loss"] = log_loss(y_true, y_score)
            except ValueError:
                metrics["log_loss"] = np.nan
        else:
            metrics["roc_auc"] = np.nan
            metrics["average_precision"] = np.nan
            metrics["brier_score"] = np.nan
            metrics["log_loss"] = np.nan

        return metrics

    @staticmethod
    def report(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_score: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        return {
            "metrics": EvaluationMetrics.evaluate(y_true, y_pred, y_score),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
        }


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


class BaseHotelModel:
    name = "Base Model"

    def get_estimator(self) -> Any:
        raise NotImplementedError

    def build_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", self.get_estimator()),
            ]
        )


class LogisticRegressionModel(BaseHotelModel):
    name = "Logistic Regression"

    def get_estimator(self) -> LogisticRegression:
        return LogisticRegression(max_iter=1000, random_state=42)


class KNNModel(BaseHotelModel):
    name = "KNN"

    def get_estimator(self) -> KNeighborsClassifier:
        return KNeighborsClassifier()


class DecisionTreeModel(BaseHotelModel):
    name = "Decision Tree"

    def get_estimator(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(random_state=42)


class NaiveBayesModel(BaseHotelModel):
    name = "Naive Bayes"

    def get_estimator(self) -> GaussianNB:
        return GaussianNB()


class SVMModel(BaseHotelModel):
    name = "SVM"

    def get_estimator(self) -> CalibratedClassifierCV:
        return CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=42, dual="auto"),
            cv=3,
        )


class RandomForestModel(BaseHotelModel):
    name = "Random Forest"

    def get_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


class AdaBoostModel(BaseHotelModel):
    name = "AdaBoost"

    def get_estimator(self) -> AdaBoostClassifier:
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        try:
            return AdaBoostClassifier(estimator=tree, random_state=42)
        except TypeError:
            return AdaBoostClassifier(base_estimator=tree, random_state=42)


class GradientBoostingModel(BaseHotelModel):
    name = "Gradient Boosting"

    def get_estimator(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(random_state=42)


class ExtraTreesModel(BaseHotelModel):
    name = "Extra Trees"

    def get_estimator(self) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)


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
            learning_rate=0.1,
            max_depth=5,
            n_estimators=120,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )


class VotingEnsembleModel(BaseHotelModel):
    name = "Voting Ensemble"

    def get_estimator(self) -> VotingClassifier:
        estimators = [
            ("logistic", LogisticRegression(max_iter=1000, random_state=42)),
            ("decision_tree", DecisionTreeClassifier(random_state=42)),
            ("random_forest", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
            ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
            ("extra_trees", ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
        ]
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


class ANNModel(BaseHotelModel):
    name = "ANN"

    def __init__(self, epochs: int = 250, batch_size: int = 256) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

    def get_estimator(self) -> Any:
        try:
            importlib.import_module("tensorflow")
            return KerasTabularClassifier(model_type="ann", epochs=self.epochs, batch_size=self.batch_size)
        except ImportError:
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                learning_rate_init=0.001,
                batch_size=min(self.batch_size, 256),
                max_iter=self.epochs,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )


class RNNModel(BaseHotelModel):
    name = "RNN"

    def __init__(self, epochs: int = 20, batch_size: int = 256) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

    def get_estimator(self) -> KerasTabularClassifier:
        return KerasTabularClassifier(model_type="rnn", epochs=self.epochs, batch_size=self.batch_size)


MODEL_REGISTRY: Dict[str, Type[BaseHotelModel]] = {
    model.name: model
    for model in (
        LogisticRegressionModel,
        KNNModel,
        DecisionTreeModel,
        NaiveBayesModel,
        SVMModel,
        RandomForestModel,
        AdaBoostModel,
        GradientBoostingModel,
        ExtraTreesModel,
        XGBoostModel,
        VotingEnsembleModel,
        ANNModel,
        RNNModel,
    )
}


class ModelTrainer:
    def __init__(
        self,
        processor: Optional[HotelDataProcessor] = None,
        random_state: int = 42,
        test_size: float = 0.3,
    ) -> None:
        self.processor = processor or HotelDataProcessor()
        self.random_state = random_state
        self.test_size = test_size

    def prepare_data(
        self,
        data_path: str = "hotel_bookings.csv",
        sample_size: Optional[int] = None,
        remove_leakage_features: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        data = self.processor.load_data(data_path)
        if sample_size and sample_size < len(data):
            data = data.sample(sample_size, random_state=self.random_state)
        return self.processor.build_features(data, remove_leakage_features=remove_leakage_features)

    def split_data(
        self,
        x_data: pd.DataFrame,
        y_data: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(
            x_data,
            y_data,
            test_size=self.test_size,
            stratify=y_data,
            random_state=self.random_state,
        )

    def train_model(
        self,
        model_spec: BaseHotelModel,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Pipeline:
        preprocessor = self.processor.build_preprocessor(x_train)
        pipeline = model_spec.build_pipeline(preprocessor)
        return pipeline.fit(x_train, y_train)

    def train_many(
        self,
        model_specs: Iterable[BaseHotelModel],
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, Pipeline]:
        return {model.name: self.train_model(model, x_train, y_train) for model in model_specs}

    def k_fold_cross_validate(
        self,
        model_specs: Iterable[BaseHotelModel],
        x_data: pd.DataFrame,
        y_data: pd.Series,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for model_spec in model_specs:
            fold_metrics: List[Dict[str, float]] = []
            for fold, (train_index, valid_index) in enumerate(splitter.split(x_data, y_data), start=1):
                x_train, x_valid = x_data.iloc[train_index], x_data.iloc[valid_index]
                y_train, y_valid = y_data.iloc[train_index], y_data.iloc[valid_index]
                fitted_model = self.train_model(model_spec, x_train, y_train)
                y_pred = fitted_model.predict(x_valid)
                y_score = _positive_probabilities(fitted_model, x_valid)
                metrics = EvaluationMetrics.evaluate(y_valid, y_pred, y_score)
                metrics.update({"model": model_spec.name, "fold": fold})
                fold_metrics.append(metrics)
                rows.append(metrics)

            average_metrics = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
            average_metrics.update({"model": model_spec.name, "fold": "mean"})
            rows.append(average_metrics)

        return pd.DataFrame(rows)


class ModelTester:
    def test_model(
        self,
        name: str,
        model: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        y_pred = model.predict(x_test)
        y_score = _positive_probabilities(model, x_test)
        result = EvaluationMetrics.report(y_test, y_pred, y_score)
        result["model"] = name
        return result

    def test_many(
        self,
        models: Dict[str, Pipeline],
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        details = {
            name: self.test_model(name, model, x_test, y_test)
            for name, model in models.items()
        }
        summary = pd.DataFrame(
            [dict(model=name, **details[name]["metrics"]) for name in details]
        ).sort_values("f1", ascending=False)
        return summary, details


class SHAPAnalyzer:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def explain(
        self,
        model: Pipeline,
        x_background: pd.DataFrame,
        x_explain: pd.DataFrame,
        max_background: int = 100,
    ) -> Any:
        import shap
        from scipy import sparse

        background = x_background.sample(
            min(max_background, len(x_background)),
            random_state=self.random_state,
        )
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]

        background_values = preprocessor.transform(background)
        explain_values = preprocessor.transform(x_explain)
        if sparse.issparse(background_values):
            background_values = background_values.toarray()
        if sparse.issparse(explain_values):
            explain_values = explain_values.toarray()

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{index}" for index in range(background_values.shape[1])]

        background_frame = pd.DataFrame(background_values, columns=feature_names)
        explain_frame = pd.DataFrame(explain_values, columns=feature_names)

        def predict_fn(values: np.ndarray) -> np.ndarray:
            return estimator.predict_proba(np.asarray(values, dtype=np.float32))[:, 1]

        explainer = shap.Explainer(predict_fn, background_frame)
        return explainer(explain_frame)

    def summary_plot(self, shap_values: Any, max_display: int = 15) -> Any:
        import matplotlib.pyplot as plt
        import shap

        shap.summary_plot(shap_values, show=False, max_display=max_display)
        figure = plt.gcf()
        plt.tight_layout()
        return figure


class KMeansSegmenter:
    def __init__(self, random_state: int = 42, n_clusters: int = 4) -> None:
        self.random_state = random_state
        self.n_clusters = n_clusters

    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        numeric_features = [
            column
            for column in ("lead_time", "adr", "total_nights", "total_guests", "previous_cancellations")
            if column in data.columns
        ]
        working = data[numeric_features].copy()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(working)
        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=20)
        labels = model.fit_predict(scaled)

        enriched = working.copy()
        enriched["segment"] = labels

        pca = PCA(n_components=2, random_state=self.random_state)
        projection = pca.fit_transform(scaled)
        projection_frame = pd.DataFrame(
            {
                "pc1": projection[:, 0],
                "pc2": projection[:, 1],
                "segment": labels,
            }
        )
        segment_summary = enriched.groupby("segment").mean(numeric_only=True).round(2).reset_index()
        return {
            "model": model,
            "summary": segment_summary,
            "projection": projection_frame,
            "feature_columns": numeric_features,
        }


class TrainingArtifacts:
    def __init__(self, output_dir: str | Path) -> None:
        self.root = Path(output_dir)
        self.models_dir = self.root / "models"
        self.plots_dir = self.root / "plots"
        self.reports_dir = self.root / "reports"
        for directory in (self.root, self.models_dir, self.plots_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def save_model(self, name: str, model: Pipeline) -> Path:
        path = self.models_dir / f"{_slugify(name)}.joblib"
        joblib.dump(model, path)
        return path

    def save_dataframe(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.reports_dir / name
        if path.suffix.lower() == ".json":
            frame.to_json(path, orient="records", indent=2)
        else:
            frame.to_csv(path, index=False)
        return path

    def save_json(self, name: str, payload: Dict[str, Any]) -> Path:
        path = self.reports_dir / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=self._json_default)
        return path

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value


class TerminalTrainingRunner:
    def __init__(
        self,
        trainer: Optional[ModelTrainer] = None,
        tester: Optional[ModelTester] = None,
        random_state: int = 42,
    ) -> None:
        self.trainer = trainer or ModelTrainer(random_state=random_state, test_size=0.3)
        self.tester = tester or ModelTester()
        self.random_state = random_state
        self.processor = self.trainer.processor

    def default_models(self, ann_epochs: int = 250, rnn_epochs: int = 10) -> List[BaseHotelModel]:
        models: List[BaseHotelModel] = [
            ANNModel(epochs=ann_epochs),
            KNNModel(),
            DecisionTreeModel(),
            RandomForestModel(),
            NaiveBayesModel(),
            SVMModel(),
            GradientBoostingModel(),
            ExtraTreesModel(),
            XGBoostModel(),
            VotingEnsembleModel(),
        ]
        try:
            importlib.import_module("tensorflow")
            models.append(RNNModel(epochs=rnn_epochs))
        except ImportError:
            pass
        return models

    def run(
        self,
        data_path: str,
        output_dir: str = "artifacts",
        cv_folds: int = 5,
        ann_epochs: int = 250,
        rnn_epochs: int = 10,
        shap_rows: int = 250,
    ) -> Dict[str, Any]:
        artifacts = TrainingArtifacts(output_dir)
        raw_data = self.processor.load_data(data_path)
        prediction_inputs = self.processor.build_raw_prediction_inputs(raw_data, remove_leakage_features=True)
        x_data, y_data = self.processor.build_features(raw_data, remove_leakage_features=True)
        x_train, x_test, y_train, y_test = self.trainer.split_data(x_data, y_data)
        models = self.default_models(ann_epochs=ann_epochs, rnn_epochs=rnn_epochs)

        benchmark_rows: List[Dict[str, Any]] = []
        details: Dict[str, Dict[str, Any]] = {}
        trained_models: Dict[str, Pipeline] = {}
        skipped_models: Dict[str, str] = {}

        try:
            importlib.import_module("tensorflow")
        except ImportError:
            skipped_models["RNN"] = "TensorFlow is not installed in this Python 3.14 environment."

        for model_spec in models:
            try:
                training_start = time.perf_counter()
                trained_model = self.trainer.train_model(model_spec, x_train, y_train)
                training_time = time.perf_counter() - training_start

                inference_start = time.perf_counter()
                detail = self.tester.test_model(model_spec.name, trained_model, x_test, y_test)
                inference_time = time.perf_counter() - inference_start

                model_path = artifacts.save_model(model_spec.name, trained_model)
                model_size_mb = model_path.stat().st_size / (1024 * 1024)
                estimator = trained_model.named_steps["model"]
                transformed_feature_count = int(
                    trained_model.named_steps["preprocessor"].transform(x_train.iloc[:1]).shape[1]
                )

                metrics = detail["metrics"].copy()
                metrics.update(
                    {
                        "model": model_spec.name,
                        "training_time_sec": training_time,
                        "inference_time_sec": inference_time,
                        "inference_ms_per_row": (inference_time / max(len(x_test), 1)) * 1000,
                        "complexity_score": _count_model_complexity(estimator),
                        "transformed_feature_count": transformed_feature_count,
                        "model_size_mb": model_size_mb,
                    }
                )
                benchmark_rows.append(metrics)
                details[model_spec.name] = detail
                trained_models[model_spec.name] = trained_model
            except Exception as exc:
                skipped_models[model_spec.name] = str(exc)

        holdout_summary = pd.DataFrame(benchmark_rows).sort_values(
            ["f1", "accuracy", "roc_auc"],
            ascending=[False, False, False],
        )
        artifacts.save_dataframe("holdout_summary.csv", holdout_summary)
        cv_results = self.trainer.k_fold_cross_validate(
            [model for model in models if model.name in trained_models],
            x_data,
            y_data,
            n_splits=cv_folds,
        )
        artifacts.save_dataframe("cross_validation_results.csv", cv_results)

        best_model_name = holdout_summary.iloc[0]["model"] if not holdout_summary.empty else None
        shap_explanations: List[Dict[str, Any]] = []
        if best_model_name:
            try:
                shap_explanations = self._save_shap_artifacts(
                    artifacts,
                    best_model_name,
                    trained_models[best_model_name],
                    x_train,
                    x_test,
                    rows=min(shap_rows, len(x_test)),
                )
            except Exception:
                shap_explanations = []

        self._save_confusion_matrices(artifacts, details)
        self._save_metric_plots(artifacts, holdout_summary, cv_results)
        segmentation = self._save_segmentation_artifacts(artifacts, x_data)
        prediction_schema = self._build_prediction_schema(prediction_inputs)
        artifacts.save_json("prediction_schema.json", prediction_schema)
        prediction_inputs.head(500).to_csv(artifacts.reports_dir / "prediction_examples.csv", index=False)

        deployment_model_name = self._select_deployment_model(holdout_summary)

        metadata = {
            "data_path": str(Path(data_path).resolve()),
            "python_version": sys.version.split()[0],
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "train_ratio": 0.7,
            "test_ratio": 0.3,
            "cross_validation_folds": cv_folds,
            "best_model": best_model_name,
            "deployment_model": deployment_model_name,
            "trained_models": list(trained_models.keys()),
            "skipped_models": skipped_models,
            "shap_explanations": shap_explanations,
            "segmentation_summary_rows": segmentation["summary"].to_dict(orient="records"),
        }
        try:
            metadata["tensorflow_version"] = importlib.import_module("tensorflow").__version__
        except ImportError:
            metadata["tensorflow_version"] = None
        artifacts.save_json("metadata.json", metadata)

        if best_model_name:
            best_path = artifacts.models_dir / "best_model.joblib"
            joblib.dump(trained_models[best_model_name], best_path)
        if deployment_model_name:
            deployment_path = artifacts.models_dir / "deployment_model.joblib"
            joblib.dump(trained_models[deployment_model_name], deployment_path)

        return {
            "holdout_summary": holdout_summary,
            "cross_validation_results": cv_results,
            "metadata": metadata,
            "details": details,
        }

    @staticmethod
    def _select_deployment_model(holdout_summary: pd.DataFrame) -> Optional[str]:
        if holdout_summary.empty:
            return None
        deployable = holdout_summary[holdout_summary["model_size_mb"] <= 100].copy()
        if deployable.empty:
            return str(holdout_summary.iloc[0]["model"])
        deployable = deployable.sort_values(
            ["f1", "accuracy", "roc_auc"],
            ascending=[False, False, False],
        )
        return str(deployable.iloc[0]["model"])

    def _build_prediction_schema(self, prediction_inputs: pd.DataFrame) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"columns": []}
        categorical_columns = list(prediction_inputs.select_dtypes(include=["object", "category"]).columns)
        numeric_columns = [column for column in prediction_inputs.columns if column not in categorical_columns]

        for column in categorical_columns:
            series = prediction_inputs[column].astype(str).fillna("Unknown")
            mode = series.mode(dropna=True)
            schema["columns"].append(
                {
                    "name": column,
                    "type": "categorical",
                    "default": str(mode.iloc[0]) if not mode.empty else str(series.iloc[0]),
                    "options": sorted(series.unique().tolist()),
                }
            )

        for column in numeric_columns:
            series = pd.to_numeric(prediction_inputs[column], errors="coerce").fillna(0)
            schema["columns"].append(
                {
                    "name": column,
                    "type": "numeric",
                    "default": _safe_float(series.median()),
                    "min": _safe_float(series.min()),
                    "max": _safe_float(series.max()),
                    "step": 1.0 if pd.api.types.is_integer_dtype(series) else 0.1,
                }
            )
        return schema

    def _save_confusion_matrices(
        self,
        artifacts: TrainingArtifacts,
        details: Dict[str, Dict[str, Any]],
    ) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        for model_name, detail in details.items():
            figure, axis = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                detail["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="YlGnBu",
                cbar=False,
                ax=axis,
            )
            axis.set_title(f"{model_name} Confusion Matrix")
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            figure.tight_layout()
            figure.savefig(artifacts.plots_dir / f"{_slugify(model_name)}_confusion_matrix.png", dpi=180)
            plt.close(figure)

    def _save_metric_plots(
        self,
        artifacts: TrainingArtifacts,
        holdout_summary: pd.DataFrame,
        cv_results: pd.DataFrame,
    ) -> None:
        import matplotlib.pyplot as plt

        if holdout_summary.empty:
            return

        metrics = holdout_summary.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
        figure, axis = plt.subplots(figsize=(12, 6))
        metrics.plot(kind="bar", ax=axis, colormap="viridis")
        axis.set_ylim(0, 1.05)
        axis.set_title("Holdout Metrics by Model")
        axis.set_ylabel("Score")
        figure.tight_layout()
        figure.savefig(artifacts.plots_dir / "holdout_metrics.png", dpi=180)
        plt.close(figure)

        timing = holdout_summary.set_index("model")[["training_time_sec", "inference_ms_per_row"]]
        figure, axis = plt.subplots(figsize=(12, 6))
        timing.plot(kind="bar", ax=axis, colormap="cividis")
        axis.set_title("Observed Training and Inference Cost")
        axis.set_ylabel("Time")
        figure.tight_layout()
        figure.savefig(artifacts.plots_dir / "timing_metrics.png", dpi=180)
        plt.close(figure)

        cv_means = cv_results[cv_results["fold"].astype(str) == "mean"]
        if not cv_means.empty:
            figure, axis = plt.subplots(figsize=(12, 6))
            axis.bar(cv_means["model"], cv_means["f1"], color="#1f6f8b")
            axis.set_ylim(0, 1.05)
            axis.set_title("5-Fold Cross-Validation Mean F1")
            axis.set_ylabel("F1")
            axis.tick_params(axis="x", rotation=35)
            figure.tight_layout()
            figure.savefig(artifacts.plots_dir / "cross_validation_f1.png", dpi=180)
            plt.close(figure)

    def _save_shap_artifacts(
        self,
        artifacts: TrainingArtifacts,
        model_name: str,
        model: Pipeline,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        rows: int,
    ) -> List[Dict[str, Any]]:
        import matplotlib.pyplot as plt

        sample = x_test.sample(rows, random_state=self.random_state)
        analyzer = SHAPAnalyzer(random_state=self.random_state)
        shap_values = analyzer.explain(model, x_train, sample)
        summary_figure = analyzer.summary_plot(shap_values)
        summary_figure.savefig(artifacts.plots_dir / f"{_slugify(model_name)}_shap_summary.png", dpi=180)
        plt.close(summary_figure)

        values = np.asarray(shap_values.values)
        feature_names = list(shap_values.feature_names)
        mean_strength = np.abs(values).mean(axis=0)
        top_indices = np.argsort(mean_strength)[::-1][:3]
        explanations: List[Dict[str, Any]] = []

        for index in top_indices:
            feature_name = feature_names[index]
            feature_values = np.asarray(shap_values.data[:, index], dtype=float)
            shap_column = values[:, index]
            correlation = float(np.corrcoef(feature_values, shap_column)[0, 1]) if len(feature_values) > 1 else 0.0
            direction = "Higher values tend to increase cancellation risk." if correlation >= 0 else "Higher values tend to reduce cancellation risk."

            figure, axis = plt.subplots(figsize=(7, 5))
            axis.scatter(feature_values, shap_column, alpha=0.55, color="#0f766e")
            axis.axhline(0, color="#9ca3af", linestyle="--", linewidth=1)
            axis.set_title(f"SHAP Dependence: {feature_name}")
            axis.set_xlabel(feature_name)
            axis.set_ylabel("Impact on cancellation risk")
            figure.tight_layout()
            figure.savefig(
                artifacts.plots_dir / f"{_slugify(model_name)}_shap_{_slugify(feature_name)}.png",
                dpi=180,
            )
            plt.close(figure)

            explanations.append(
                {
                    "feature": feature_name,
                    "mean_abs_shap": float(mean_strength[index]),
                    "correlation_with_risk": correlation,
                    "explanation": direction,
                }
            )

        return explanations

    def _save_segmentation_artifacts(
        self,
        artifacts: TrainingArtifacts,
        x_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        import matplotlib.pyplot as plt

        segmenter = KMeansSegmenter(random_state=self.random_state, n_clusters=4)
        segmentation = segmenter.fit(x_data)
        segmentation["summary"].to_csv(artifacts.reports_dir / "guest_segments.csv", index=False)

        projection = segmentation["projection"]
        figure, axis = plt.subplots(figsize=(8, 6))
        scatter = axis.scatter(
            projection["pc1"],
            projection["pc2"],
            c=projection["segment"],
            cmap="viridis",
            alpha=0.65,
            s=20,
        )
        axis.set_title("Guest Segmentation with K-Means")
        axis.set_xlabel("Principal Component 1")
        axis.set_ylabel("Principal Component 2")
        figure.colorbar(scatter, ax=axis, label="Segment")
        figure.tight_layout()
        figure.savefig(artifacts.plots_dir / "guest_segmentation.png", dpi=180)
        plt.close(figure)
        return segmentation
