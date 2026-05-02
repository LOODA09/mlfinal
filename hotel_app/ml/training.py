from __future__ import annotations

from dataclasses import dataclass
import json
import importlib
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
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
    ExtraTreesModel,
    GradientBoostingModel,
    KNNModel,
    NaiveBayesModel,
    RandomForestModel,
    RNNModel,
    SVMModel,
    StackingEnsembleModel,
    VotingEnsembleModel,
    XGBoostModel,
)


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
    ):
        data = self.processor.load_data(data_path)
        if sample_size and sample_size < len(data):
            data = data.sample(sample_size, random_state=self.random_state)
        return self.processor.build_features(data, remove_leakage_features=remove_leakage_features)

    def split_data(self, x_data: pd.DataFrame, y_data: pd.Series):
        return train_test_split(
            x_data,
            y_data,
            test_size=self.test_size,
            stratify=y_data,
            random_state=self.random_state,
        )

    def train_model(self, model_spec: BaseHotelModel, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        preprocessor = self.processor.build_preprocessor(x_train)
        pipeline = model_spec.build_pipeline(preprocessor)
        return pipeline.fit(x_train, y_train)

    def train_many(self, model_specs: Iterable[BaseHotelModel], x_train: pd.DataFrame, y_train: pd.Series):
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
        projection_frame = pd.DataFrame({"pc1": projection[:, 0], "pc2": projection[:, 1], "segment": labels})
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
    def __init__(self, trainer: Optional[ModelTrainer] = None, random_state: int = 42) -> None:
        self.trainer = trainer or ModelTrainer(random_state=random_state, test_size=0.3)
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
            StackingEnsembleModel(),
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
        from .testing import ModelTester

        tester = ModelTester()
        artifacts = TrainingArtifacts(output_dir)
        raw_data = self.processor.load_data(data_path)
        prediction_inputs = self.processor.build_raw_prediction_inputs(raw_data, remove_leakage_features=True)
        x_data, y_data = self.processor.build_features(raw_data, remove_leakage_features=True)
        x_train, x_test, y_train, y_test = self.trainer.split_data(x_data, y_data)
        models = self.default_models(ann_epochs=ann_epochs, rnn_epochs=rnn_epochs)

        benchmark_rows: List[Dict[str, Any]] = []
        benchmark_rows_by_name: Dict[str, Dict[str, Any]] = {}  # keyed for later timing update
        details: Dict[str, Dict[str, Any]] = {}
        trained_models: Dict[str, Pipeline] = {}
        skipped_models: Dict[str, str] = {}

        for model_spec in models:
            try:
                training_start = time.perf_counter()
                trained_model = self.trainer.train_model(model_spec, x_train, y_train)
                training_time = time.perf_counter() - training_start
                inference_start = time.perf_counter()
                detail = tester.test_model(model_spec.name, trained_model, x_test, y_test)
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
            except Exception as exc:
                skipped_models[model_spec.name] = str(exc)

        holdout_summary = pd.DataFrame(benchmark_rows).sort_values(
            ["f1", "accuracy", "roc_auc"], ascending=[False, False, False]
        )
        artifacts.save_dataframe("holdout_summary.csv", holdout_summary)
        cv_results = self.trainer.k_fold_cross_validate(
            [model for model in models if model.name in trained_models], x_data, y_data, n_splits=cv_folds
        )
        artifacts.save_dataframe("cross_validation_results.csv", cv_results)

        best_model_name = holdout_summary.iloc[0]["model"] if not holdout_summary.empty else None
        shap_explanations: List[Dict[str, Any]] = []
        if best_model_name:
            try:
                shap_explanations = self._save_shap_artifacts(
                    artifacts, best_model_name, trained_models[best_model_name], x_train, x_test, rows=min(shap_rows, len(x_test))
                )
            except Exception:
                shap_explanations = []

        # ── Retrain every benchmarked model on the FULL dataset and save ──────
        print("\nRetraining all models on full dataset for deployment...")
        full_data_models: Dict[str, Pipeline] = {}
        for model_spec in models:
            if model_spec.name not in trained_models:
                continue  # skip models that failed during benchmarking
            try:
                full_start = time.perf_counter()
                full_model = self.trainer.train_model(model_spec, x_data, y_data)
                full_time = time.perf_counter() - full_start
                artifacts.save_model(model_spec.name, full_model)
                full_data_models[model_spec.name] = full_model
                # Update training_time_sec = benchmark time + full-data retrain time (true total cost)
                if model_spec.name in benchmark_rows_by_name:
                    row = benchmark_rows_by_name[model_spec.name]
                    row["full_data_training_time_sec"] = full_time
                    row["training_time_sec"] = row["benchmark_training_time_sec"] + full_time
                print(f"  [{model_spec.name}] full-data train: {full_time:.1f}s  total: {row['training_time_sec']:.1f}s")
            except Exception as exc:
                print(f"  WARNING: Could not retrain {model_spec.name} on full data: {exc}")

        # Save the best-performing model as the deployment model
        deployment_model_name = best_model_name
        if deployment_model_name and deployment_model_name in full_data_models:
            deployment_path = artifacts.models_dir / "deployment_model.joblib"
            joblib.dump(full_data_models[deployment_model_name], deployment_path)
            print(f"  Deployment model saved: {deployment_model_name} → deployment_model.joblib")
        elif trained_models:
            # Fallback: use the best available full-data model (or benchmark model if full-data failed)
            fallback_name = next(iter(full_data_models or trained_models))
            deployment_model_name = fallback_name
            fallback_model = (full_data_models or trained_models)[fallback_name]
            deployment_path = artifacts.models_dir / "deployment_model.joblib"
            joblib.dump(fallback_model, deployment_path)
            print(f"  Deployment model saved (fallback): {fallback_name} → deployment_model.joblib")

        segmentation = self._save_segmentation_artifacts(artifacts, x_data)
        metadata = {
            "data_path": str(Path(data_path).resolve()),
            "python_version": sys.version.split()[0],
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "total_rows": int(len(x_data)),
            "train_ratio": 0.7,
            "test_ratio": 0.3,
            "cross_validation_folds": cv_folds,
            "best_model": best_model_name,
            "deployment_model": deployment_model_name,
            "trained_models": list(trained_models.keys()),
            "full_data_models": list(full_data_models.keys()),
            "skipped_models": skipped_models,
            "shap_explanations": shap_explanations,
            "segmentation_summary_rows": segmentation["summary"].to_dict(orient="records"),
            "tensorflow_version": None,
        }
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

    def _save_segmentation_artifacts(self, artifacts: TrainingArtifacts, x_data: pd.DataFrame) -> Dict[str, Any]:
        import matplotlib.pyplot as plt

        segmenter = KMeansSegmenter(random_state=self.random_state, n_clusters=4)
        segmentation = segmenter.fit(x_data)
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
