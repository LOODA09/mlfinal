from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hotel_app.ml.testing import ModelTester
from hotel_app.ml.training import ModelTrainer, TerminalTrainingRunner, TrainingArtifacts
from hotel_app.ml.data import _count_model_complexity


MODEL_NAMES = [
    "Logistic Regression",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "SVM",
    "ANN",
    "RNN",
    "LSTM",
]


def main() -> None:
    output_dir = Path("artifacts")
    artifacts = TrainingArtifacts(output_dir)

    for old_model in artifacts.models_dir.glob("*.joblib"):
        old_model.unlink()

    trainer = ModelTrainer(random_state=42, test_size=0.2)
    runner = TerminalTrainingRunner(trainer=trainer, random_state=42)
    processor = runner.processor
    tester = ModelTester()

    raw_data = processor.load_data("hotel reservation data set .csv")
    prediction_inputs = processor.build_raw_prediction_inputs(
        raw_data,
        remove_leakage_features=False,
        feature_preset="high_score",
    )
    x_data, y_data = processor.build_features(
        raw_data,
        remove_leakage_features=False,
        feature_preset="high_score",
    )
    x_train, x_test, y_train, y_test = trainer.split_data(x_data, y_data)
    x_fit, y_fit = trainer.resample_training_data(x_train, y_train)

    models = runner.default_models(
        ann_epochs=120,
        rnn_epochs=45,
        lstm_epochs=55,
        selected_models=MODEL_NAMES,
    )

    benchmark_rows: list[dict[str, object]] = []
    confusion_payload: dict[str, dict[str, object]] = {}
    trained_models: dict[str, object] = {}

    pipeline_start = time.perf_counter()
    benchmark_phase_start = time.perf_counter()
    for model_spec in models:
        print(f"TRAIN {model_spec.name}", flush=True)
        train_start = time.perf_counter()
        model = trainer.train_model(model_spec, x_fit, y_fit)
        benchmark_training_time = time.perf_counter() - train_start

        inference_start = time.perf_counter()
        detail = tester.test_model(model_spec.name, model, x_train, y_train, x_test, y_test)
        inference_time = time.perf_counter() - inference_start

        artifacts.save_model(model_spec.name, model)
        trained_models[model_spec.name] = model

        row = detail["metrics"].copy()
        row.update(
            {
                "model": model_spec.name,
                "training_time_sec": benchmark_training_time,
                "benchmark_training_time_sec": benchmark_training_time,
                "full_data_training_time_sec": np.nan,
                "inference_time_sec": inference_time,
                "inference_ms_per_row": (inference_time / max(len(x_test), 1)) * 1000,
                "complexity_score": _count_model_complexity(model.named_steps["model"]),
                "transformed_feature_count": int(
                    model.named_steps["preprocessor"].transform(x_train.iloc[:1]).shape[1]
                ),
            }
        )
        benchmark_rows.append(row)
        confusion_payload[model_spec.name] = {
            "matrix": np.asarray(detail["confusion_matrix"]).tolist(),
            "labels": ["Actual 0", "Actual 1"],
            "predicted": ["Predicted 0", "Predicted 1"],
        }

    benchmark_phase_wall_clock_sec = time.perf_counter() - benchmark_phase_start

    holdout_summary = pd.DataFrame(benchmark_rows).sort_values(
        ["accuracy", "f1", "roc_auc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best_model_name = str(holdout_summary.iloc[0]["model"])

    shap_phase_start = time.perf_counter()
    try:
        shap_explanations = runner._save_shap_artifacts(
            artifacts,
            best_model_name,
            trained_models[best_model_name],
            x_train,
            x_test,
            rows=min(80, len(x_test)),
        )
    except Exception:
        shap_explanations = []
    shap_phase_wall_clock_sec = time.perf_counter() - shap_phase_start

    print(f"RETRAIN {best_model_name}", flush=True)
    full_retrain_start = time.perf_counter()
    deployment_model = trainer.retrain_from_benchmark(trained_models[best_model_name], x_data, y_data)
    full_retrain_phase_wall_clock_sec = time.perf_counter() - full_retrain_start
    joblib.dump(
        deployment_model,
        artifacts.models_dir / "deployment_model.joblib",
        compress=("xz", 3),
    )
    mask = holdout_summary["model"] == best_model_name
    holdout_summary.loc[mask, "full_data_training_time_sec"] = full_retrain_phase_wall_clock_sec
    holdout_summary.loc[mask, "training_time_sec"] = (
        holdout_summary.loc[mask, "benchmark_training_time_sec"] + full_retrain_phase_wall_clock_sec
    )

    artifact_finalize_start = time.perf_counter()
    segmentation = runner._save_segmentation_artifacts(artifacts, x_data)
    artifacts.save_dataframe("holdout_summary.csv", holdout_summary)
    artifacts.save_dataframe("cross_validation_results.csv", pd.DataFrame())
    artifacts.save_json("confusion_matrices.json", confusion_payload)
    artifacts.save_json("prediction_schema.json", runner._build_prediction_schema(prediction_inputs))
    prediction_inputs.head(500).to_csv(artifacts.reports_dir / "prediction_examples.csv", index=False)
    artifact_finalize_wall_clock_sec = time.perf_counter() - artifact_finalize_start

    metadata = {
        "data_path": str(Path("hotel reservation data set .csv").resolve()),
        "dataset_kind": "reservation",
        "remove_leakage_features": False,
        "feature_preset": "high_score",
        "balance_strategy": "smote" if len(x_fit) > len(x_train) else "none",
        "evaluation_mode": "Notebook-Aligned Reservation Benchmark",
        "python_version": sys.version.split()[0],
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "total_rows": int(len(x_data)),
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "cross_validation_folds": 0,
        "best_model": best_model_name,
        "deployment_model": best_model_name,
        "trained_models": MODEL_NAMES,
        "full_data_models": [best_model_name],
        "skipped_models": {},
        "shap_explanations": shap_explanations,
        "segmentation_summary_rows": segmentation["summary"].to_dict(orient="records"),
        "tensorflow_version": None,
        "benchmark_phase_wall_clock_sec": benchmark_phase_wall_clock_sec,
        "shap_phase_wall_clock_sec": shap_phase_wall_clock_sec,
        "full_retrain_phase_wall_clock_sec": full_retrain_phase_wall_clock_sec,
        "artifact_finalize_wall_clock_sec": artifact_finalize_wall_clock_sec,
        "total_pipeline_wall_clock_sec": time.perf_counter() - pipeline_start,
        "pipeline_wall_clock_note": (
            "Total pipeline wall clock includes notebook-aligned reservation feature generation, "
            "SMOTE balancing, benchmark fits, SHAP generation, best-model retraining, "
            "and artifact generation."
        ),
    }
    try:
        metadata["tensorflow_version"] = importlib.import_module("tensorflow").__version__
    except Exception:
        pass
    artifacts.save_json("metadata.json", metadata)

    print(holdout_summary[["model", "accuracy", "f1", "roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
