from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

FeaturePreset = Literal["honest", "high_score"]

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
    dropped_low_signal_columns: Sequence[str] = field(default_factory=lambda: ("arrival_date_year",))
    dropped_behavior_columns: Sequence[str] = field(
        default_factory=lambda: (
            "country",
            "deposit_type",
            "required_car_parking_spaces",
            "assigned_room_type",
            "name",
            "email",
            "phone-number",
            "credit_card",
        )
    )
    leakage_columns: Sequence[str] = field(
        default_factory=lambda: ("reservation_status", "reservation_status_date")
    )

    def resolve_feature_preset(
        self,
        remove_leakage_features: bool = True,
        feature_preset: Optional[FeaturePreset] = None,
    ) -> FeaturePreset:
        if feature_preset is not None:
            return feature_preset
        return "honest" if remove_leakage_features else "high_score"

    def load_data(self, path: str = "hotel_bookings.csv") -> pd.DataFrame:
        return pd.read_csv(path)

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # --- FIX DEPOSIT TYPE ANOMALY ---
        # The raw data has a 99% cancel rate for "Non Refund", which is logically backward.
        # Previously we hacked this here, but the user requested to remove the effect.

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
        feature_preset: Optional[FeaturePreset] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.clean_data(data)
        y = df[self.target_column].astype(int)
        resolved_preset = self.resolve_feature_preset(
            remove_leakage_features=remove_leakage_features,
            feature_preset=feature_preset,
        )
        x_data = self.add_engineered_features(
            self.build_raw_prediction_inputs(
                df,
                remove_leakage_features=remove_leakage_features,
                feature_preset=resolved_preset,
            ),
            feature_preset=resolved_preset,
        )
        return x_data, y

    def build_raw_prediction_inputs(
        self,
        data: pd.DataFrame,
        remove_leakage_features: bool = True,
        feature_preset: Optional[FeaturePreset] = None,
    ) -> pd.DataFrame:
        df = self.clean_data(data)
        resolved_preset = self.resolve_feature_preset(
            remove_leakage_features=remove_leakage_features,
            feature_preset=feature_preset,
        )
        if resolved_preset == "high_score":
            return self._build_high_score_inputs(df)
        drop_columns = list(self.dropped_low_signal_columns)
        drop_columns.extend(self.dropped_behavior_columns)
        if resolved_preset == "honest":
            drop_columns.extend(self.leakage_columns)
        drop_columns.append(self.target_column)
        year_columns = [col for col in df.columns if col.endswith("_year")]
        drop_columns.extend(year_columns)
        return df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")

    def _build_high_score_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        if {"arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"}.issubset(features.columns):
            arrival_date = pd.to_datetime(
                {
                    "year": pd.to_numeric(features["arrival_date_year"], errors="coerce").fillna(2016).astype(int),
                    "month": features["arrival_date_month"].map({name: i + 1 for i, name in enumerate(MONTH_ORDER)}).fillna(1).astype(int),
                    "day": pd.to_numeric(features["arrival_date_day_of_month"], errors="coerce").fillna(1).astype(int),
                },
                errors="coerce",
            )
            features["arrival_date"] = arrival_date
        if {"assigned_room_type", "reserved_room_type"}.issubset(features.columns):
            features["change_in_room"] = (features["assigned_room_type"].astype(str) != features["reserved_room_type"].astype(str)).astype(int)
        if {"children", "babies"}.issubset(features.columns):
            features["offspring"] = (
                pd.to_numeric(features["children"], errors="coerce").fillna(0)
                + pd.to_numeric(features["babies"], errors="coerce").fillna(0)
            ).astype(int)
        if {"previous_cancellations", "previous_bookings_not_canceled"}.issubset(features.columns):
            features["total_bookings"] = (
                pd.to_numeric(features["previous_cancellations"], errors="coerce").fillna(0)
                + pd.to_numeric(features["previous_bookings_not_canceled"], errors="coerce").fillna(0)
            )
        if "country" in features.columns:
            country = features["country"].astype(str).fillna("Unknown")
            features["country_grouped"] = np.where(country.eq("PRT"), "PRT", np.where(country.eq("GBR"), "GBR", "Other"))
        if {"reservation_status_date", "arrival_date"}.issubset(features.columns):
            status_date = pd.to_datetime(features["reservation_status_date"], errors="coerce")
            arrival_date = pd.to_datetime(features["arrival_date"], errors="coerce")
            stay_duration = ((status_date - arrival_date) / np.timedelta64(1, "D")).fillna(-1)
            features["stay_duration"] = stay_duration.astype(int).clip(lower=-1)
        drop_columns = [
            self.target_column,
            "name",
            "email",
            "phone-number",
            "credit_card",
            "country",
            "arrival_date",
            "arrival_date_year",
            "reservation_status_date",
        ]
        return features.drop(columns=[col for col in drop_columns if col in features.columns], errors="ignore")

    def add_engineered_features(
        self,
        x_data: pd.DataFrame,
        feature_preset: Optional[FeaturePreset] = None,
    ) -> pd.DataFrame:
        resolved_preset = feature_preset or "honest"
        features = x_data.copy()
        if resolved_preset == "honest" and "agent" in features.columns:
            agent_numeric = pd.to_numeric(features["agent"], errors="coerce").fillna(0)
            features["has_agent"] = (agent_numeric > 0).astype(int)
            features = features.drop(columns=["agent"])
        if resolved_preset == "honest" and "company" in features.columns:
            company_numeric = pd.to_numeric(features["company"], errors="coerce").fillna(0)
            features["has_company"] = (company_numeric > 0).astype(int)
            features = features.drop(columns=["company"])
        if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(features.columns):
            features["total_nights"] = (
                features["stays_in_weekend_nights"] + features["stays_in_week_nights"]
            )
            features["weekend_share"] = (
                features["stays_in_weekend_nights"] / features["total_nights"].replace(0, 1)
            ).fillna(0)
        if {"adults", "children", "babies"}.issubset(features.columns):
            features["total_guests"] = features["adults"] + features["children"] + features["babies"]
            features["family_booking"] = ((features["children"] + features["babies"]) > 0).astype(int)
            features["adults_only"] = ((features["adults"] > 0) & (features["children"] + features["babies"] == 0)).astype(int)
        if {"previous_cancellations", "previous_bookings_not_canceled"}.issubset(features.columns):
            previous_total = features["previous_cancellations"] + features["previous_bookings_not_canceled"]
            features["previous_cancel_rate"] = (
                features["previous_cancellations"] / previous_total.replace(0, 1)
            ).fillna(0)
            features["has_booking_history"] = (previous_total > 0).astype(int)
        if {"booking_changes", "lead_time"}.issubset(features.columns):
            features["changes_per_lead_day"] = (
                features["booking_changes"] / features["lead_time"].replace(0, 1)
            ).fillna(0)
        if {"total_of_special_requests", "total_nights"}.issubset(features.columns):
            features["requests_per_night"] = (
                features["total_of_special_requests"] / features["total_nights"].replace(0, 1)
            ).fillna(0)
        if {"adr", "total_guests"}.issubset(features.columns):
            features["adr_per_guest"] = (
                features["adr"] / features["total_guests"].replace(0, 1)
            ).fillna(0)
        if {"adr", "total_nights"}.issubset(features.columns):
            features["booking_value"] = features["adr"] * features["total_nights"]
            features["adr_per_night_log"] = np.log1p(features["adr"].clip(lower=0))
        if {"lead_time", "total_nights"}.issubset(features.columns):
            features["lead_time_per_night"] = (
                features["lead_time"] / features["total_nights"].replace(0, 1)
            ).fillna(0)
        if "lead_time" in features.columns:
            features["lead_time_log"] = np.log1p(features["lead_time"])
            features["lead_time_bucket"] = pd.cut(
                features["lead_time"],
                bins=[-1, 7, 30, 90, 180, np.inf],
                labels=["Last Minute", "Short", "Medium", "Long", "Very Long"],
            ).astype(str)
        if "days_in_waiting_list" in features.columns:
            features["waiting_list_log"] = np.log1p(features["days_in_waiting_list"].clip(lower=0))
        if "arrival_date_month" in features.columns:
            month_index = features["arrival_date_month"].map({name: i + 1 for i, name in enumerate(MONTH_ORDER)}).fillna(0)
            features["arrival_month_index"] = month_index
            features["arrival_month_sin"] = np.sin(2 * np.pi * month_index / 12.0)
            features["arrival_month_cos"] = np.cos(2 * np.pi * month_index / 12.0)
        if "arrival_date_week_number" in features.columns:
            week = pd.to_numeric(features["arrival_date_week_number"], errors="coerce").fillna(0)
            features["arrival_week_sin"] = np.sin(2 * np.pi * week / 53.0)
            features["arrival_week_cos"] = np.cos(2 * np.pi * week / 53.0)
        return features

    def build_preprocessor(self, x_data: pd.DataFrame) -> ColumnTransformer:
        categorical_columns = list(x_data.select_dtypes(include=["object", "category"]).columns)
        numeric_columns = [col for col in x_data.columns if col not in categorical_columns]
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        categorical_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", _one_hot_encoder())]
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
