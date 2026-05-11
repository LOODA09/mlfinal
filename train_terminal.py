from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hotel_app.ml import TerminalTrainingRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train hotel cancellation models from the terminal and save artifacts."
    )
    parser.add_argument("--data", default="hotel_bookings.csv", help="Path to the dataset CSV file.")
    parser.add_argument("--output", default="artifacts", help="Directory for saved models and reports.")
    parser.add_argument("--cv-folds", type=int, default=3, help="Number of stratified CV folds. Use 0 or 1 to skip CV.")
    parser.add_argument("--ann-epochs", type=int, default=250, help="ANN max iterations.")
    parser.add_argument("--rnn-epochs", type=int, default=10, help="RNN epochs when TensorFlow is available.")
    parser.add_argument("--lstm-epochs", type=int, default=10, help="LSTM epochs when TensorFlow is available.")
    parser.add_argument("--shap-rows", type=int, default=250, help="Rows to use for SHAP plots.")
    parser.add_argument(
        "--feature-preset",
        choices=["honest", "high_score"],
        default=None,
        help="Feature preset to use for training artifacts.",
    )
    parser.add_argument(
        "--keep-leakage-features",
        action="store_true",
        help="Keep leakage features such as reservation_status in training and prediction artifacts.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional list of model names to train, for example --models \"Logistic Regression\" KNN \"Random Forest\" XGBoost ANN",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runner = TerminalTrainingRunner()
    results = runner.run(
        data_path=args.data,
        output_dir=args.output,
        cv_folds=args.cv_folds,
        ann_epochs=args.ann_epochs,
        rnn_epochs=args.rnn_epochs,
        lstm_epochs=args.lstm_epochs,
        shap_rows=args.shap_rows,
        selected_models=args.models,
        remove_leakage_features=not args.keep_leakage_features,
        feature_preset=args.feature_preset,
    )

    holdout = results["holdout_summary"].copy()
    cross_validation = results["cross_validation_results"].copy()
    metadata = results["metadata"]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\nTraining complete.\n")
    test_ratio = float(metadata.get("test_ratio", 0.2))
    print(f"Holdout summary ({int(round(test_ratio * 100))}% test split plus benchmark-train metrics):")
    print(holdout.round(4).to_string(index=False))

    fold_count = int(metadata.get("cross_validation_folds", args.cv_folds))
    print(f"\n{fold_count}-fold cross-validation means:")
    if not cross_validation.empty and "fold" in cross_validation.columns:
        cv_means = cross_validation[cross_validation["fold"].astype(str) == "mean"].copy()
    else:
        cv_means = pd.DataFrame()
    if not cv_means.empty:
        print(cv_means.round(4).to_string(index=False))
    else:
        print("No cross-validation rows were produced.")

    print("\nSaved artifacts:")
    print(f"- Output directory: {Path(args.output).resolve()}")
    print(f"- Best model: {metadata.get('best_model')}")
    if metadata.get("skipped_models"):
        print("- Skipped models:")
        for name, reason in metadata["skipped_models"].items():
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
