from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type
import importlib
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
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


@dataclass
class HotelDataProcessor:
    target_column: str = "is_canceled"
    dropped_low_signal_columns: Sequence[str] = field(
        default_factory=lambda: (
            "agent",
            "adr",
            "babies",
            "stays_in_week_nights",
            "arrival_date_year",
            "arrival_date_week_number",
            "arrival_date_day_of_month",
            "children",
            "stays_in_weekend_nights",
        )
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

        if not remove_leakage_features and "reservation_status_date" in df.columns:
            reservation_date = pd.to_datetime(df["reservation_status_date"], errors="coerce")
            df["reservation_status_year"] = reservation_date.dt.year.fillna(0).astype(int)
            df["reservation_status_month"] = reservation_date.dt.month.fillna(0).astype(int)
            df["reservation_status_day"] = reservation_date.dt.day.fillna(0).astype(int)
            df = df.drop(columns=["reservation_status_date"])

        drop_columns = list(self.dropped_low_signal_columns)
        if remove_leakage_features:
            drop_columns.extend(self.leakage_columns)
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])

        y = df[self.target_column].astype(int)
        x_data = df.drop(columns=[self.target_column])
        return x_data, y

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
                metrics["log_loss"] = log_loss(y_true, y_score)
            except ValueError:
                metrics["log_loss"] = np.nan
        else:
            metrics["roc_auc"] = np.nan
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
            n_estimators=180,
            eval_metric="logloss",
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

    def __init__(self, epochs: int = 20, batch_size: int = 256) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

    def get_estimator(self) -> KerasTabularClassifier:
        return KerasTabularClassifier(model_type="ann", epochs=self.epochs, batch_size=self.batch_size)


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
