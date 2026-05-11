from __future__ import annotations

from dataclasses import dataclass
import json
import importlib
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import HotelDataProcessor, _count_model_complexity, _positive_probabilities, _safe_float, _slugify
from .explainability import SHAPAnalyzer
from .metrics import EvaluationMetrics
from .models import (
    ANNModel,
    BaseHotelModel,
    DecisionTreeModel,
    KNNModel,
    LSTMModel,
    LogisticRegressionModel,
    RandomForestModel,
    RNNModel,
    SVMModel,
    XGBoostModel,
)


class ModelTrainer:
    def __init__(
        self,
        processor: Optional[HotelDataProcessor] = None,
        random_state: int = 42,
        test_size: float = 0.2,
    ) -> None:
        self.processor = processor or HotelDataProcessor()
        self.random_state = random_state
        self.test_size = test_size

    def prepare_data(
        self,
        data_path: str = "hotel_bookings.csv",
        sample_size: Optional[int] = None,
        remove_leakage_features: bool = True,
        feature_preset: Optional[str] = None,
    ):
        data = self.processor.load_data(data_path)
        if sample_size and sample_size < len(data):
            data = data.sample(sample_size, random_state=self.random_state)
        return self.processor.build_features(
            data,
            remove_leakage_features=remove_leakage_features,
            feature_preset=feature_preset,
        )

    def split_data(self, x_data: pd.DataFrame, y_data: pd.Series):
        stratify_target = None if self.processor._is_reservation_feature_frame(x_data) else y_data
        return train_test_split(
            x_data,
            y_data,
            test_size=self.test_size,
            stratify=stratify_target,
            random_state=self.random_state,
        )

    def train_model(self, model_spec: BaseHotelModel, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        preprocessor = self.processor.build_preprocessor(x_train)
        pipeline = model_spec.build_pipeline(preprocessor)
        return pipeline.fit(x_train, y_train)

    def retrain_from_benchmark(self, benchmark_model: Pipeline, x_data: pd.DataFrame, y_data: pd.Series) -> Pipeline:
        preprocessor = clone(benchmark_model.named_steps["preprocessor"])
        benchmark_estimator = benchmark_model.named_steps["model"]
        if hasattr(benchmark_estimator, "best_params_") and hasattr(benchmark_estimator, "estimator"):
            final_estimator = clone(benchmark_estimator.estimator)
            final_estimator.set_params(**benchmark_estimator.best_params_)
        elif hasattr(benchmark_estimator, "best_estimator_"):
            final_estimator = clone(benchmark_estimator.best_estimator_)
        else:
            final_estimator = clone(benchmark_estimator)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", final_estimator)])
        return pipeline.fit(x_data, y_data)

    def train_many(self, model_specs: Iterable[BaseHotelModel], x_train: pd.DataFrame, y_train: pd.Series):
        return {model.name: self.train_model(model, x_train, y_train) for model in model_specs}

    def resample_training_data(self, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        try:
            from imblearn.over_sampling import SMOTE, SMOTENC
        except ImportError:
            return x_train, y_train

        categorical_columns = list(x_train.select_dtypes(include=["object", "category"]).columns)
        if categorical_columns:
            sampler = SMOTENC(
                categorical_features=[x_train.columns.get_loc(column) for column in categorical_columns],
                random_state=self.random_state,
            )
        else:
            sampler = SMOTE(random_state=self.random_state)

        x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
        if not isinstance(x_resampled, pd.DataFrame):
            x_resampled = pd.DataFrame(x_resampled, columns=x_train.columns)
        for column in x_train.columns:
            if pd.api.types.is_numeric_dtype(x_train[column]):
                replacement = pd.to_numeric(x_train[column], errors="coerce").median()
                x_resampled[column] = pd.to_numeric(x_resampled[column], errors="coerce").fillna(replacement)
            else:
                x_resampled[column] = x_resampled[column].astype(str)
        return x_resampled.reset_index(drop=True), pd.Series(y_resampled, name=y_train.name)

    def k_fold_cross_validate(
        self,
        model_specs: Iterable[BaseHotelModel],
        x_data: pd.DataFrame,
        y_data: pd.Series,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        if n_splits < 2:
            return pd.DataFrame()
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


class KMeansSegmenter:
    def __init__(self, random_state: int = 42, n_clusters: int = 4) -> None:
        self.random_state = random_state
        self.n_clusters = n_clusters

    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        numeric_features = [
            column
            for column in (
                "lead_time",
                "adr",
                "average_price",
                "total_nights",
                "number_of_total_nights",
                "total_guests",
                "number_of_children_and_adults",
                "special_requests",
                "previous_cancellations",
                "cancellation_ratio",
            )
            if column in data.columns
        ]
        if not numeric_features:
            projection_frame = pd.DataFrame({"pc1": [], "pc2": [], "segment": []})
            return {
                "model": None,
                "summary": pd.DataFrame(),
                "projection": projection_frame,
                "feature_columns": [],
            }
        working = data[numeric_features].copy()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(working)
        cluster_count = min(self.n_clusters, max(len(working), 1))
        model = KMeans(n_clusters=cluster_count, random_state=self.random_state, n_init=20)
        labels = model.fit_predict(scaled)
        enriched = working.copy()
        enriched["segment"] = labels
        if scaled.shape[1] >= 2:
            pca = PCA(n_components=2, random_state=self.random_state)
            projection = pca.fit_transform(scaled)
            projection_frame = pd.DataFrame({"pc1": projection[:, 0], "pc2": projection[:, 1], "segment": labels})
        else:
            projection_frame = pd.DataFrame({"pc1": scaled[:, 0], "pc2": np.zeros(len(scaled)), "segment": labels})
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
        joblib.dump(model, path, compress=3)
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
    def __init__(self, trainer: Optional[ModelTrainer] = None, random_state: int = 42) -> None:
        self.trainer = trainer or ModelTrainer(random_state=random_state, test_size=0.2)
        self.random_state = random_state
        self.processor = self.trainer.processor

    def default_models(
        self,
        ann_epochs: int = 250,
        rnn_epochs: int = 10,
        lstm_epochs: int = 10,
        selected_models: Optional[Sequence[str]] = None,
    ) -> List[BaseHotelModel]:
        default_order = [
            "Logistic Regression",
            "KNN",
            "Decision Tree",
            "Random Forest",
            "SVM",
            "XGBoost",
            "ANN",
            "LSTM",
            "RNN",
        ]
        model_names = list(selected_models) if selected_models else default_order
        models: List[BaseHotelModel] = []
        tensorflow_available = False
        try:
            importlib.import_module("tensorflow")
            tensorflow_available = True
        except ImportError:
            tensorflow_available = False

        for model_name in model_names:
            if model_name == "ANN":
                models.append(ANNModel(epochs=ann_epochs))
            elif model_name == "Logistic Regression":
                models.append(LogisticRegressionModel())
            elif model_name == "KNN":
                models.append(KNNModel())
            elif model_name == "Decision Tree":
                models.append(DecisionTreeModel())
            elif model_name == "Random Forest":
                models.append(RandomForestModel())
            elif model_name == "SVM":
                models.append(SVMModel())
            elif model_name == "XGBoost":
                models.append(XGBoostModel())
            elif model_name == "LSTM":
                if tensorflow_available:
                    models.append(LSTMModel(epochs=lstm_epochs))
            elif model_name == "RNN":
                if tensorflow_available:
                    models.append(RNNModel(epochs=rnn_epochs))
            else:
                raise ValueError(f"Unknown model requested: {model_name}")
        return models

    def run(
        self,
        data_path: str,
        output_dir: str = "artifacts",
        cv_folds: int = 3,
        ann_epochs: int = 250,
        rnn_epochs: int = 10,
        lstm_epochs: int = 10,
        shap_rows: int = 250,
        selected_models: Optional[Sequence[str]] = None,
        remove_leakage_features: bool = True,
        feature_preset: Optional[str] = None,
    ) -> Dict[str, Any]:
        from .testing import ModelTester

        pipeline_start = time.perf_counter()
        tester = ModelTester()
        artifacts = TrainingArtifacts(output_dir)
        raw_data = self.processor.load_data(data_path)
        dataset_kind = self.processor.detect_dataset(raw_data)
        resolved_preset = self.processor.resolve_feature_preset(
            remove_leakage_features=remove_leakage_features,
            feature_preset=feature_preset,
        )
        prediction_inputs = self.processor.build_raw_prediction_inputs(
            raw_data,
            remove_leakage_features=remove_leakage_features,
            feature_preset=resolved_preset,
        )
        x_data, y_data = self.processor.build_features(
            raw_data,
            remove_leakage_features=remove_leakage_features,
            feature_preset=resolved_preset,
        )
        x_train, x_test, y_train, y_test = self.trainer.split_data(x_data, y_data)
        x_fit, y_fit = x_train, y_train
        balance_strategy = "none"
        if dataset_kind == "reservation":
            try:
                x_fit, y_fit = self.trainer.resample_training_data(x_train, y_train)
                if len(x_fit) > len(x_train):
                    balance_strategy = "smote"
            except Exception as exc:
                balance_strategy = f"smote_failed: {exc}"
        models = self.default_models(
            ann_epochs=ann_epochs,
            rnn_epochs=rnn_epochs,
            lstm_epochs=lstm_epochs,
            selected_models=selected_models,
        )

        benchmark_rows: List[Dict[str, Any]] = []
        benchmark_rows_by_name: Dict[str, Dict[str, Any]] = {}  # keyed for later timing update
        details: Dict[str, Dict[str, Any]] = {}
        trained_models: Dict[str, Pipeline] = {}
        skipped_models: Dict[str, str] = {}
        confusion_payload: Dict[str, Dict[str, Any]] = {}
        benchmark_phase_start = time.perf_counter()

        for model_spec in models:
            try:
                training_start = time.perf_counter()
                trained_model = self.trainer.train_model(model_spec, x_fit, y_fit)
                training_time = time.perf_counter() - training_start
                inference_start = time.perf_counter()
                detail = tester.test_model(model_spec.name, trained_model, x_train, y_train, x_test, y_test)
                inference_time = time.perf_counter() - inference_start
                metrics = detail["metrics"].copy()
                metrics.update(
                    {
                        "model": model_spec.name,
                        "training_time_sec": training_time,      # updated after full-data retrain
                        "benchmark_training_time_sec": training_time,  # 70% split only, preserved
                        "inference_time_sec": inference_time,
                        "inference_ms_per_row": (inference_time / max(len(x_test), 1)) * 1000,
                        "complexity_score": _count_model_complexity(trained_model.named_steps["model"]),
                        "transformed_feature_count": int(
                            trained_model.named_steps["preprocessor"].transform(x_train.iloc[:1]).shape[1]
                        ),
                    }
                )
                benchmark_rows.append(metrics)
                benchmark_rows_by_name[model_spec.name] = metrics
                details[model_spec.name] = detail
                trained_models[model_spec.name] = trained_model
                confusion_payload[model_spec.name] = {
                    "matrix": np.asarray(detail["confusion_matrix"]).tolist(),
                    "labels": ["Actual 0", "Actual 1"],
                    "predicted": ["Predicted 0", "Predicted 1"],
                }
                artifacts.save_model(model_spec.name, trained_model)
            except Exception as exc:
                skipped_models[model_spec.name] = str(exc)

        if not benchmark_rows:
            raise RuntimeError(f"All requested models failed to train. Reasons: {skipped_models}")

        holdout_summary = pd.DataFrame(benchmark_rows).sort_values(
            ["f1", "accuracy", "roc_auc"], ascending=[False, False, False]
        )
        cv_results = self.trainer.k_fold_cross_validate(
            [model for model in models if model.name in trained_models], x_data, y_data, n_splits=cv_folds
        )
        artifacts.save_dataframe("cross_validation_results.csv", cv_results)
        artifacts.save_json("confusion_matrices.json", confusion_payload)
        benchmark_phase_wall_clock_sec = time.perf_counter() - benchmark_phase_start

        best_model_name = holdout_summary.iloc[0]["model"] if not holdout_summary.empty else None
        shap_explanations: List[Dict[str, Any]] = []
        shap_phase_start = time.perf_counter()
        if best_model_name:
            try:
                shap_explanations = self._save_shap_artifacts(
                    artifacts, best_model_name, trained_models[best_model_name], x_train, x_test, rows=min(shap_rows, len(x_test))
                )
            except Exception:
                shap_explanations = []
        shap_phase_wall_clock_sec = time.perf_counter() - shap_phase_start

        # ── Retrain every benchmarked model on the FULL dataset and save ──────
        print("\nRetraining the best model on full dataset for deployment...")
        full_data_models: Dict[str, Pipeline] = {}
        full_retrain_phase_start = time.perf_counter()
        if best_model_name and best_model_name in trained_models:
            try:
                full_start = time.perf_counter()
                full_model = self.trainer.retrain_from_benchmark(trained_models[best_model_name], x_data, y_data)
                full_time = time.perf_counter() - full_start
                full_data_models[best_model_name] = full_model
                if best_model_name in benchmark_rows_by_name:
                    row = benchmark_rows_by_name[best_model_name]
                    row["full_data_training_time_sec"] = full_time
                    row["training_time_sec"] = row["benchmark_training_time_sec"] + full_time
                print(f"  [{best_model_name}] full-data train: {full_time:.1f}s")
            except Exception as exc:
                print(f"  WARNING: Could not retrain {best_model_name} on full data: {exc}")
        full_retrain_phase_wall_clock_sec = time.perf_counter() - full_retrain_phase_start

        # Save the best-performing model as the deployment model
        deployment_model_name = best_model_name
        artifact_finalize_start = time.perf_counter()
        if deployment_model_name and deployment_model_name in full_data_models:
            deployment_path = artifacts.models_dir / "deployment_model.joblib"
            joblib.dump(full_data_models[deployment_model_name], deployment_path, compress=("xz", 3))
            print(f"  Deployment model saved: {deployment_model_name} -> deployment_model.joblib")
        elif trained_models:
            # Fallback: use the best available full-data model (or benchmark model if full-data failed)
            fallback_name = next(iter(full_data_models or trained_models))
            deployment_model_name = fallback_name
            fallback_model = (full_data_models or trained_models)[fallback_name]
            deployment_path = artifacts.models_dir / "deployment_model.joblib"
            joblib.dump(fallback_model, deployment_path, compress=("xz", 3))
            print(f"  Deployment model saved (fallback): {fallback_name} -> deployment_model.joblib")

        segmentation = self._save_segmentation_artifacts(artifacts, x_data)
        holdout_summary = pd.DataFrame(benchmark_rows).sort_values(
            ["f1", "accuracy", "roc_auc"], ascending=[False, False, False]
        )
        artifacts.save_dataframe("holdout_summary.csv", holdout_summary)
        artifact_finalize_wall_clock_sec = time.perf_counter() - artifact_finalize_start
        metadata = {
            "data_path": str(Path(data_path).resolve()),
            "dataset_kind": dataset_kind,
            "remove_leakage_features": remove_leakage_features,
            "feature_preset": resolved_preset,
            "balance_strategy": balance_strategy,
            "evaluation_mode": "Honest Prediction" if resolved_preset == "honest" else "High-Score Benchmark",
            "python_version": sys.version.split()[0],
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "total_rows": int(len(x_data)),
            "train_ratio": round(1.0 - self.trainer.test_size, 4),
            "test_ratio": round(self.trainer.test_size, 4),
            "cross_validation_folds": cv_folds,
            "best_model": best_model_name,
            "deployment_model": deployment_model_name,
            "trained_models": list(trained_models.keys()),
            "full_data_models": list(full_data_models.keys()),
            "skipped_models": skipped_models,
            "shap_explanations": shap_explanations,
            "segmentation_summary_rows": segmentation["summary"].to_dict(orient="records"),
            "tensorflow_version": None,
            "benchmark_phase_wall_clock_sec": benchmark_phase_wall_clock_sec,
            "shap_phase_wall_clock_sec": shap_phase_wall_clock_sec,
            "full_retrain_phase_wall_clock_sec": full_retrain_phase_wall_clock_sec,
            "artifact_finalize_wall_clock_sec": artifact_finalize_wall_clock_sec,
            "total_pipeline_wall_clock_sec": time.perf_counter() - pipeline_start,
        }
        metadata["pipeline_wall_clock_note"] = (
            "Total pipeline wall clock includes the benchmark holdout fits, cross-validation, "
            "SHAP generation, best-model full-data retraining for the deployment artifact, and report creation. "
            "Per-model training_time_sec is the sum of benchmark_training_time_sec and "
            "full_data_training_time_sec when a full-data retrain was performed."
        )
        try:
            metadata["tensorflow_version"] = importlib.import_module("tensorflow").__version__
        except ImportError:
            pass
        artifacts.save_json("metadata.json", metadata)
        artifacts.save_json(
            "prediction_schema.json",
            self._build_prediction_schema(prediction_inputs),
        )
        prediction_inputs.head(500).to_csv(artifacts.reports_dir / "prediction_examples.csv", index=False)
        return {"holdout_summary": holdout_summary, "cross_validation_results": cv_results, "metadata": metadata}

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

    def _save_shap_artifacts(self, artifacts: TrainingArtifacts, model_name: str, model: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame, rows: int):
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
            explanations.append(
                {
                    "feature": feature_name,
                    "mean_abs_shap": float(mean_strength[index]),
                    "correlation_with_risk": correlation,
                    "explanation": "Higher values tend to increase cancellation risk." if correlation >= 0 else "Higher values tend to reduce cancellation risk.",
                }
            )
        return explanations

    @staticmethod
    def _segment_display_names(summary: pd.DataFrame) -> dict[int, str]:
        if summary.empty or "segment" not in summary.columns:
            return {}
        working = summary.copy()
        for column in [
            "lead_time",
            "average_price",
            "number_of_total_nights",
            "number_of_children_and_adults",
            "special_requests",
            "cancellation_ratio",
        ]:
            if column in working.columns:
                working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0)

        names: dict[int, str] = {}
        if "cancellation_ratio" in working.columns:
            risk_segment = int(working.sort_values("cancellation_ratio", ascending=False).iloc[0]["segment"])
            names[risk_segment] = "High-Risk Returners"
        if "lead_time" in working.columns:
            planner_segment = int(working.sort_values("lead_time", ascending=False).iloc[0]["segment"])
            names.setdefault(planner_segment, "Advance Planners")
        premium_features = [column for column in ("average_price", "special_requests") if column in working.columns]
        if premium_features:
            premium_scores = working[premium_features].mean(axis=1)
            premium_segment = int(working.iloc[int(premium_scores.idxmax())]["segment"])
            names.setdefault(premium_segment, "Premium Experience Guests")

        for row in working.itertuples(index=False):
            segment_id = int(getattr(row, "segment"))
            if segment_id in names:
                continue
            lead_time_value = float(getattr(row, "lead_time", 0.0))
            booking_value = float(getattr(row, "average_price", 0.0))
            if lead_time_value <= working["lead_time"].median() and booking_value <= working["average_price"].median():
                names[segment_id] = "Quick-Book Value Guests"
            else:
                names[segment_id] = "Steady Leisure Guests"
        return names

    def _save_segmentation_artifacts(self, artifacts: TrainingArtifacts, x_data: pd.DataFrame) -> Dict[str, Any]:
        import matplotlib.pyplot as plt

        segmenter = KMeansSegmenter(random_state=self.random_state, n_clusters=4)
        segmentation = segmenter.fit(x_data)
        segment_names = self._segment_display_names(segmentation["summary"])
        if not segmentation["summary"].empty:
            segmentation["summary"] = segmentation["summary"].copy()
            segmentation["summary"]["segment_name"] = segmentation["summary"]["segment"].map(segment_names)
        if not segmentation["projection"].empty:
            segmentation["projection"] = segmentation["projection"].copy()
            segmentation["projection"]["segment_name"] = segmentation["projection"]["segment"].map(segment_names)
        segmentation["summary"].to_csv(artifacts.reports_dir / "guest_segments.csv", index=False)
        projection = segmentation["projection"]
        figure, axis = plt.subplots(figsize=(8, 6))
        scatter = axis.scatter(
            projection["pc1"], projection["pc2"], c=projection["segment"], cmap="viridis", alpha=0.65, s=20
        )
        axis.set_title("Guest Segmentation with K-Means")
        axis.set_xlabel("Principal Component 1")
        axis.set_ylabel("Principal Component 2")
        figure.colorbar(scatter, ax=axis, label="Segment")
        figure.tight_layout()
        figure.savefig(artifacts.plots_dir / "guest_segmentation.png", dpi=180)
        plt.close(figure)
        return segmentation
