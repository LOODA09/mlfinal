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


def _one_hot_encoder(
    drop_first: bool = False,
    categories: Optional[Sequence[Sequence[Any]]] = None,
) -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            categories=categories,
            drop="first" if drop_first else None,
            sparse_output=False,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            categories=categories,
            drop="first" if drop_first else None,
            sparse=False,
            dtype=np.float32,
        )


def _positive_probabilities(model: Any, x_data: pd.DataFrame) -> Optional[np.ndarray]:
    """Return positive-class probabilities for any fitted classifier.

    Most classifiers in this project expose ``predict_proba`` directly.
    For score-based estimators that only expose ``decision_function``,
    such as a linear SVM before calibration, we convert scores with the
    logistic sigmoid function ``1 / (1 + exp(-score))`` so downstream
    metrics and dashboards can work with probability-style outputs.
    """
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
    if hasattr(estimator, "best_estimator_"):
        return _count_model_complexity(estimator.best_estimator_)
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
    if hasattr(estimator, "estimator_"):
        return _count_model_complexity(estimator.estimator_)
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
    reservation_target_column: str = "booking_status"
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
    reservation_guest_count_categories: Sequence[str] = field(
        default_factory=lambda: tuple(str(value) for value in range(1, 13))
    )

    def resolve_feature_preset(
        self,
        remove_leakage_features: bool = True,
        feature_preset: Optional[FeaturePreset] = None,
    ) -> FeaturePreset:
        if feature_preset is not None:
            return feature_preset
        return "honest" if remove_leakage_features else "high_score"

    @staticmethod
    def _snake_case(value: str) -> str:
        value = value.strip().lower()
        value = re.sub(r"[^a-z0-9]+", "_", value)
        return re.sub(r"_+", "_", value).strip("_")

    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.columns = [self._snake_case(str(column)) for column in df.columns]
        return df

    def detect_dataset(self, data: pd.DataFrame) -> str:
        columns = set(data.columns)
        if self.reservation_target_column in columns:
            return "reservation"
        if self.target_column in columns:
            return "hotel"
        raise KeyError("Could not find a supported target column in the dataset.")

    def resolve_target_column(self, data: pd.DataFrame) -> str:
        if self.target_column in data.columns:
            return self.target_column
        if self.reservation_target_column in data.columns:
            return self.reservation_target_column
        raise KeyError("Could not resolve the target column for the dataset.")

    def load_data(self, path: str = "hotel_bookings.csv") -> pd.DataFrame:
        return self._standardize_columns(pd.read_csv(path))

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self._standardize_columns(data)

        if "date_of_reservation" in df.columns:
            df["date_of_reservation"] = pd.to_datetime(
                df["date_of_reservation"],
                format="%m/%d/%Y",
                errors="coerce",
            )
            df = df.dropna(subset=["date_of_reservation"]).copy()

        for price_column in ("adr", "average_price"):
            if price_column in df.columns:
                prices = pd.to_numeric(df[price_column], errors="coerce").fillna(0)
                prices = prices.clip(lower=0)
                if price_column == "average_price" and self.reservation_target_column in df.columns and len(prices) > 10:
                    q1 = float(prices.quantile(0.25))
                    q3 = float(prices.quantile(0.75))
                    iqr = q3 - q1
                    lower_bound = max(0.0, q1 - 1.5 * iqr)
                    upper_bound = q3 + 1.5 * iqr
                    median_price = float(prices.median())
                    prices = prices.mask((prices < lower_bound) | (prices > upper_bound), median_price)
                elif len(prices) > 10:
                    lower_quantile = float(prices.quantile(0.01))
                    upper_quantile = float(prices.quantile(0.99))
                    prices = prices.clip(lower=lower_quantile, upper=upper_quantile)
                df[price_column] = prices
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
        target_column = self.resolve_target_column(df)
        if target_column == self.reservation_target_column:
            y = df[target_column].astype(str).str.lower().eq("canceled").astype(int)
        else:
            y = pd.to_numeric(df[target_column], errors="coerce").fillna(0).astype(int)
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
        dataset_kind = self.detect_dataset(df)
        resolved_preset = self.resolve_feature_preset(
            remove_leakage_features=remove_leakage_features,
            feature_preset=feature_preset,
        )
        if dataset_kind == "reservation":
            return self._build_reservation_inputs(df)
        if resolved_preset == "high_score":
            return self._build_high_score_inputs(df)
        if resolved_preset == "honest":
            df = df.copy()
            if "country" in df.columns:
                country = df["country"].astype(str).fillna("Unknown")
                df["country_grouped"] = np.where(
                    country.eq("PRT"),
                    "PRT",
                    np.where(country.eq("GBR"), "GBR", "Other"),
                )
            if "required_car_parking_spaces" in df.columns:
                parking = pd.to_numeric(df["required_car_parking_spaces"], errors="coerce").fillna(0)
                df["needs_parking"] = (parking > 0).astype(int)
        drop_columns = list(self.dropped_low_signal_columns)
        drop_columns.extend(self.dropped_behavior_columns)
        if resolved_preset == "honest":
            drop_columns.extend(self.leakage_columns)
        drop_columns.append(self.target_column)
        year_columns = [col for col in df.columns if col.endswith("_year")]
        drop_columns.extend(year_columns)
        return df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")

    def _build_reservation_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        lead_time_source = pd.to_numeric(features.get("lead_time"), errors="coerce").fillna(0)
        if {"number_of_adults", "number_of_children"}.issubset(features.columns):
            total_people = (
                pd.to_numeric(features["number_of_adults"], errors="coerce").fillna(0)
                + pd.to_numeric(features["number_of_children"], errors="coerce").fillna(0)
            ).astype(int)
            features["number_of_children_and_adults"] = total_people
        if {"number_of_weekend_nights", "number_of_week_nights"}.issubset(features.columns):
            total_nights = (
                pd.to_numeric(features["number_of_weekend_nights"], errors="coerce").fillna(0)
                + pd.to_numeric(features["number_of_week_nights"], errors="coerce").fillna(0)
            ).astype(int)
            features["number_of_total_nights"] = pd.cut(
                total_nights,
                bins=[-1, 0, 3, 7, 14, np.inf],
                labels=[0, 1, 2, 3, 4],
            ).astype("Int64").fillna(0).astype(int)
        if "lead_time" in features.columns:
            features["lead_time"] = pd.cut(
                lead_time_source,
                bins=[-1, 1, 7, 30, 365, np.inf],
                labels=[0, 1, 2, 3, 4],
            ).astype("Int64").fillna(0).astype(int)
        if {"p_c", "p_not_c", "repeated"}.issubset(features.columns):
            history_total = (
                pd.to_numeric(features["p_c"], errors="coerce").fillna(0)
                + pd.to_numeric(features["p_not_c"], errors="coerce").fillna(0)
            )
            repeated = pd.to_numeric(features["repeated"], errors="coerce").fillna(0).clip(lower=0, upper=1)
            features["cancellation_ratio"] = np.where(
                repeated > 0,
                pd.to_numeric(features["p_c"], errors="coerce").fillna(0) / history_total.replace(0, 1),
                0,
            )
            features["cancellation_ratio"] = features["cancellation_ratio"].round(2)
            features["first_time_visitor"] = 1 - repeated
        if "date_of_reservation" in features.columns:
            date_series = pd.to_datetime(features["date_of_reservation"], errors="coerce")
            features["day_name"] = date_series.dt.dayofweek.fillna(0).astype(int)
            features["month"] = date_series.dt.month.fillna(0).astype(int)
            features["year"] = date_series.dt.year.fillna(0).astype(int)

        drop_columns = [
            self.reservation_target_column,
            "booking_id",
            "date_of_reservation",
            "number_of_adults",
            "number_of_children",
            "number_of_weekend_nights",
            "number_of_week_nights",
            "repeated",
            "p_c",
            "p_not_c",
        ]
        return features.drop(columns=[col for col in drop_columns if col in features.columns], errors="ignore")

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
        if self._is_reservation_feature_frame(features):
            if "lead_time" in features.columns:
                features["lead_time"] = pd.to_numeric(features["lead_time"], errors="coerce").fillna(0).astype(int)
            if "number_of_total_nights" in features.columns:
                features["number_of_total_nights"] = (
                    pd.to_numeric(features["number_of_total_nights"], errors="coerce").fillna(0).astype(int)
                )
            if "day_name" in features.columns:
                features["day_name"] = pd.to_numeric(features["day_name"], errors="coerce").fillna(0).astype(int)
            if "number_of_children_and_adults" in features.columns:
                features["number_of_children_and_adults"] = (
                    pd.to_numeric(features["number_of_children_and_adults"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
            if "first_time_visitor" in features.columns:
                features["first_time_visitor"] = (
                    pd.to_numeric(features["first_time_visitor"], errors="coerce").fillna(0).clip(0, 1).astype(int)
                )
            if "cancellation_ratio" in features.columns:
                features["cancellation_ratio"] = (
                    pd.to_numeric(features["cancellation_ratio"], errors="coerce").fillna(0).clip(0, 1).round(2)
                )
            return features
        if {"average_price", "number_of_children_and_adults"}.issubset(features.columns):
            features["average_price_per_guest"] = (
                pd.to_numeric(features["average_price"], errors="coerce").fillna(0)
                / pd.to_numeric(features["number_of_children_and_adults"], errors="coerce").fillna(0).replace(0, 1)
            ).fillna(0)
            total_people = pd.to_numeric(features["number_of_children_and_adults"], errors="coerce").fillna(0)
            features["guest_party_type"] = np.select(
                [total_people <= 1, total_people == 2, total_people <= 4],
                ["Solo", "Couple", "Small Group"],
                default="Large Group",
            )
        if {"average_price", "number_of_total_nights"}.issubset(features.columns):
            features["total_stay_value"] = (
                pd.to_numeric(features["average_price"], errors="coerce").fillna(0)
                * pd.to_numeric(features["number_of_total_nights"], errors="coerce").fillna(0)
            )
        if "lead_time" in features.columns and "lead_time_band" not in features.columns:
            lead_time_value = pd.to_numeric(features.get("lead_time_raw", features["lead_time"]), errors="coerce").fillna(0)
            features["lead_time_band"] = lead_time_value.map(
                lambda value: "Same Day" if value <= 1 else "Short Notice" if value <= 7 else "Medium Term" if value <= 30 else "Long Term" if value <= 365 else "Very Long Term"
            )
            features["lead_time_bucket_code"] = lead_time_value.map(
                lambda value: 0 if value <= 1 else 1 if value <= 7 else 2 if value <= 30 else 3 if value <= 365 else 4
            )
        if "number_of_total_nights" in features.columns and "stay_length_bucket" not in features.columns:
            total_nights_bucket = pd.to_numeric(features["number_of_total_nights"], errors="coerce").fillna(0)
            features["stay_length_bucket"] = total_nights_bucket.map(
                lambda value: "Day Use" if value == 0 else "Short Stay" if value <= 3 else "Week Stay" if value <= 7 else "Two Weeks Stay" if value <= 14 else "Long Stay"
            )
            features["stay_length_bucket_code"] = total_nights_bucket.map(
                lambda value: 0 if value == 0 else 1 if value <= 3 else 2 if value <= 7 else 3 if value <= 14 else 4
            )
        if {"special_requests", "number_of_total_nights"}.issubset(features.columns):
            features["special_requests_per_night"] = (
                pd.to_numeric(features["special_requests"], errors="coerce").fillna(0)
                / pd.to_numeric(features["number_of_total_nights"], errors="coerce").fillna(0).replace(0, 1)
            ).fillna(0)
        if {"special_requests", "number_of_children_and_adults"}.issubset(features.columns):
            features["special_requests_per_guest"] = (
                pd.to_numeric(features["special_requests"], errors="coerce").fillna(0)
                / pd.to_numeric(features["number_of_children_and_adults"], errors="coerce").fillna(0).replace(0, 1)
            ).fillna(0)
        if "agent" in features.columns:
            agent_numeric = pd.to_numeric(features["agent"], errors="coerce").fillna(0)
            features["has_agent"] = (agent_numeric > 0).astype(int)
            if resolved_preset == "honest":
                features = features.drop(columns=["agent"])
        if "company" in features.columns:
            company_numeric = pd.to_numeric(features["company"], errors="coerce").fillna(0)
            features["has_company"] = (company_numeric > 0).astype(int)
            if resolved_preset == "honest":
                features = features.drop(columns=["company"])
        if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(features.columns):
            features["total_nights"] = (
                features["stays_in_weekend_nights"] + features["stays_in_week_nights"]
            )
            features["weekend_share"] = (
                features["stays_in_weekend_nights"] / features["total_nights"].replace(0, 1)
            ).fillna(0)
            features["stay_length_bucket"] = pd.cut(
                features["total_nights"],
                bins=[-1, 0, 3, 7, 14, np.inf],
                labels=["Day Use", "Short Stay", "Week Stay", "Two Weeks Stay", "Long Stay"],
            ).astype(str)
        if {"adults", "children", "babies"}.issubset(features.columns):
            features["total_guests"] = features["adults"] + features["children"] + features["babies"]
            features["family_booking"] = ((features["children"] + features["babies"]) > 0).astype(int)
            features["adults_only"] = ((features["adults"] > 0) & (features["children"] + features["babies"] == 0)).astype(int)
            features["guest_party_type"] = np.select(
                [
                    features["total_guests"] <= 1,
                    features["total_guests"] == 2,
                    features["total_guests"] <= 4,
                ],
                ["Solo", "Couple", "Small Group"],
                default="Large Group",
            )
        if "is_repeated_guest" in features.columns:
            repeated_guest = pd.to_numeric(features["is_repeated_guest"], errors="coerce").fillna(0).clip(lower=0, upper=1)
            features["first_time_visitor"] = 1 - repeated_guest
        if {"previous_cancellations", "previous_bookings_not_canceled"}.issubset(features.columns):
            previous_total = features["previous_cancellations"] + features["previous_bookings_not_canceled"]
            features["total_bookings"] = previous_total
            features["previous_cancel_rate"] = (
                features["previous_cancellations"] / previous_total.replace(0, 1)
            ).fillna(0)
            features["has_booking_history"] = (previous_total > 0).astype(int)
        if {"booking_changes", "lead_time"}.issubset(features.columns):
            features["changes_per_lead_day"] = (
                features["booking_changes"] / features["lead_time"].replace(0, 1)
            ).fillna(0)
        if {"booking_changes", "total_nights"}.issubset(features.columns):
            features["changes_per_night"] = (
                features["booking_changes"] / features["total_nights"].replace(0, 1)
            ).fillna(0)
        if {"total_of_special_requests", "total_nights"}.issubset(features.columns):
            features["requests_per_night"] = (
                features["total_of_special_requests"] / features["total_nights"].replace(0, 1)
            ).fillna(0)
        if {"total_of_special_requests", "total_guests"}.issubset(features.columns):
            features["requests_per_guest"] = (
                features["total_of_special_requests"] / features["total_guests"].replace(0, 1)
            ).fillna(0)
        if {"adr", "total_guests"}.issubset(features.columns):
            features["adr_per_guest"] = (
                features["adr"] / features["total_guests"].replace(0, 1)
            ).fillna(0)
        if {"adr", "total_nights"}.issubset(features.columns):
            features["booking_value"] = features["adr"] * features["total_nights"]
            features["adr_per_night_log"] = np.log1p(features["adr"].clip(lower=0))
        if {"booking_value", "total_guests", "total_nights"}.issubset(features.columns):
            features["value_per_guest_night"] = (
                features["booking_value"]
                / (features["total_guests"].replace(0, 1) * features["total_nights"].replace(0, 1))
            ).fillna(0)
        if {"total_guests", "total_nights"}.issubset(features.columns):
            features["guests_per_night"] = (
                features["total_guests"] / features["total_nights"].replace(0, 1)
            ).fillna(0)
        if {"lead_time", "total_guests"}.issubset(features.columns):
            features["lead_time_per_guest"] = (
                features["lead_time"] / features["total_guests"].replace(0, 1)
            ).fillna(0)
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
            features["lead_time_band"] = pd.cut(
                features["lead_time"],
                bins=[-1, 1, 7, 30, 365, np.inf],
                labels=["Same Day", "Short Notice", "Medium Term", "Long Term", "Very Long Term"],
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
        if self._is_reservation_feature_frame(x_data):
            categorical_columns = list(x_data.select_dtypes(include=["object", "category"]).columns)
            forced_categorical = [column for column in ("number_of_children_and_adults",) if column in x_data.columns]
            categorical_columns = list(dict.fromkeys([*categorical_columns, *forced_categorical]))
            numeric_columns = [column for column in x_data.columns if column not in categorical_columns]
            scaled_numeric_columns = [column for column in ("average_price",) if column in numeric_columns]
            passthrough_numeric_columns = [
                column for column in numeric_columns if column not in scaled_numeric_columns
            ]
            transformers = []
            if scaled_numeric_columns:
                transformers.append(
                    (
                        "scaled_numeric",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        scaled_numeric_columns,
                    )
                )
            if passthrough_numeric_columns:
                transformers.append(
                    (
                        "numeric",
                        Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                        passthrough_numeric_columns,
                    )
                )
            if categorical_columns:
                reservation_categories = {
                    "type_of_meal": ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"],
                    "room_type": [f"Room_Type {index}" for index in range(1, 8)],
                    "market_segment_type": ["Aviation", "Complementary", "Corporate", "Offline", "Online"],
                    "number_of_children_and_adults": list(range(1, 13)),
                }
                category_lists = [
                    reservation_categories.get(
                        column,
                        sorted(pd.Series(x_data[column]).dropna().unique().tolist()),
                    )
                    for column in categorical_columns
                ]
                transformers.append(
                    (
                        "categorical",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", _one_hot_encoder(drop_first=True, categories=category_lists)),
                            ]
                        ),
                        categorical_columns,
                    )
                )
            return ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                verbose_feature_names_out=False,
            )
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

    @staticmethod
    def _is_reservation_feature_frame(x_data: pd.DataFrame) -> bool:
        reservation_markers = {
            "type_of_meal",
            "car_parking_space",
            "room_type",
            "market_segment_type",
            "average_price",
            "special_requests",
        }
        return len(reservation_markers.intersection(x_data.columns)) >= 4


class NotebookEDAAnalyzer:
    def __init__(self, data: pd.DataFrame, processor: Optional[HotelDataProcessor] = None) -> None:
        self.raw_data = data.copy()
        self.processor = processor or HotelDataProcessor()
        self.clean_data = self.processor.clean_data(data)

    def preview(self, rows: int = 5) -> pd.DataFrame:
        return self.raw_data.head(rows)
