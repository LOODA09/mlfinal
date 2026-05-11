from __future__ import annotations

import json
from pathlib import Path


FILES = [
    "build_pdf_report.py",
    "build_word_report.py",
    "train_terminal.py",
    "streamlit_app.py",
    "hotel_cancellation_oop.py",
    "hotel_app/__init__.py",
    "hotel_app/ui.py",
    "hotel_app/services.py",
    "hotel_app/reporting.py",
    "hotel_app/ml/__init__.py",
    "hotel_app/ml/data.py",
    "hotel_app/ml/deep.py",
    "hotel_app/ml/explainability.py",
    "hotel_app/ml/metrics.py",
    "hotel_app/ml/testing.py",
    "hotel_app/ml/training.py",
    "hotel_app/ml/validation.py",
    "hotel_app/ml/models/__init__.py",
    "hotel_app/ml/models/base.py",
    "hotel_app/ml/models/logistic.py",
    "hotel_app/ml/models/knn.py",
    "hotel_app/ml/models/decision_tree.py",
    "hotel_app/ml/models/random_forest.py",
    "hotel_app/ml/models/svm.py",
    "hotel_app/ml/models/ann.py",
    "hotel_app/ml/models/rnn.py",
    "hotel_app/ml/models/lstm.py",
    "hotel_app/ml/models/xgboost_model.py",
]

MODEL_REFERENCE = [
    {
        "name": "Logistic Regression",
        "path": "hotel_app/ml/models/logistic.py",
        "estimator": "LogisticRegression inside GridSearchCV",
        "balancing": "class_weight='balanced'",
        "probability": "native logistic sigmoid via predict_proba",
        "training_metrics": "saved in holdout_summary.csv as train_* columns",
    },
    {
        "name": "KNN",
        "path": "hotel_app/ml/models/knn.py",
        "estimator": "KNeighborsClassifier inside GridSearchCV",
        "balancing": "BalancedClassifierWrapper with oversampling",
        "probability": "native neighbor vote probabilities",
        "training_metrics": "saved in holdout_summary.csv as train_* columns",
    },
    {
        "name": "Decision Tree",
        "path": "hotel_app/ml/models/decision_tree.py",
        "estimator": "DecisionTreeClassifier inside GridSearchCV",
        "balancing": "class_weight='balanced'",
        "probability": "native tree leaf probabilities",
        "training_metrics": "saved in holdout_summary.csv as train_* columns",
    },
    {
        "name": "Random Forest",
        "path": "hotel_app/ml/models/random_forest.py",
        "estimator": "RandomForestClassifier inside RandomizedSearchCV",
        "balancing": "class_weight='balanced_subsample'",
        "probability": "average positive-class probability over trees",
        "training_metrics": "saved in holdout_summary.csv as train_* columns",
    },
    {
        "name": "XGBoost",
        "path": "hotel_app/ml/models/xgboost_model.py",
        "estimator": "XGBClassifier with tuned tree parameters",
        "balancing": "boosted tree regularization and weighted loss behavior",
        "probability": "native predict_proba from XGBoost",
        "training_metrics": "saved in holdout_summary.csv as train_* columns",
    },
    {
        "name": "SVM",
        "path": "hotel_app/ml/models/svm.py",
        "estimator": "RBF-kernel SVC inside a stratified subsampling wrapper and GridSearchCV",
        "balancing": "class_weight='balanced'",
        "probability": "native SVC predict_proba",
        "training_metrics": "saved in holdout_summary.csv as train_* columns",
    },
    {
        "name": "ANN",
        "path": "hotel_app/ml/models/ann.py",
        "estimator": "TensorFlow KerasTabularClassifier or MLP fallback",
        "balancing": "class weights in TensorFlow, oversampling in fallback",
        "probability": "final sigmoid output layer in hotel_app/ml/deep.py",
        "training_metrics": "saved in holdout_summary.csv as train_* columns",
    },
    {
        "name": "RNN",
        "path": "hotel_app/ml/models/rnn.py",
        "estimator": "TensorFlow KerasTabularClassifier in rnn mode",
        "balancing": "class-weighted TensorFlow fit",
        "probability": "final sigmoid output layer in hotel_app/ml/deep.py",
        "training_metrics": "saved in holdout_summary.csv as train_* columns",
    },
    {
        "name": "LSTM",
        "path": "hotel_app/ml/models/lstm.py",
        "estimator": "TensorFlow KerasTabularClassifier in lstm mode",
        "balancing": "class-weighted TensorFlow fit",
        "probability": "final sigmoid output layer in hotel_app/ml/deep.py",
        "training_metrics": "saved in holdout_summary.csv as train_* columns",
    },
]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook(root: Path) -> dict:
    reference_lines = [
        "| Model | Class File | Estimator / Search | Balancing | Probability / Sigmoid Path | Metrics Saved |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in MODEL_REFERENCE:
        reference_lines.append(
            f"| {item['name']} | `{item['path']}` | {item['estimator']} | {item['balancing']} | {item['probability']} | {item['training_metrics']} |"
        )

    cells = [
        markdown_cell(
            "# Hotel Booking Cancellation Project: All Code in One Notebook\n\n"
            "This notebook was generated from the current project source so the whole "
            "implementation is available in one place.\n\n"
            "Preprocessing notes:\n"
            "- numeric features are normalized with `StandardScaler`\n"
            "- categorical features are one-hot encoded after imputation\n"
            "- Random Forest now uses `RandomizedSearchCV` to search for stronger hyperparameters such as `n_estimators`, `max_depth`, and `max_features`\n"
            "- class balancing is applied across the model set through native class weights, balanced sample weights, oversampling wrappers, and class-weighted deep learning fits\n"
            "- engineered ratio features such as `requests_per_night`, `requests_per_guest`, `adr_per_guest`, `value_per_guest_night`, and `guests_per_night` are added to expose stronger booking patterns\n"
            "- the project supports both a notebook-matched high-score benchmark mode and an honest future-booking mode\n\n"
            "Where key answers live:\n"
            "- core model classes: `hotel_app/ml/models/`\n"
            "- sigmoid / probability helper for non-probability estimators: `hotel_app/ml/data.py::_positive_probabilities`\n"
            "- TensorFlow sigmoid output for ANN/RNN/LSTM: `hotel_app/ml/deep.py`\n"
            "- train/test metric generation: `hotel_app/ml/testing.py` and `hotel_app/ml/training.py`\n"
            "- saved metric tables: `artifacts*/reports/holdout_summary.csv`\n\n"
            "Contents:\n"
            "- reporting and terminal scripts\n"
            "- Streamlit dashboard\n"
            "- service layer\n"
            "- machine learning pipeline\n"
            "- all model classes\n\n"
            "Each section below shows the source of one project file.\n"
        ),
        markdown_cell(
            "## Model Reference Map\n\n"
            "This table is meant to answer the common doctor / examiner questions about where each model lives, "
            "how it is balanced, where its probability output comes from, and where the training metrics are saved.\n\n"
            + "\n".join(reference_lines)
        ),
    ]

    for relative_path in FILES:
        path = root / relative_path
        if not path.exists():
            continue
        cells.append(markdown_cell(f"## `{relative_path}`\n"))
        cells.append(code_cell(path.read_text(encoding="utf-8")))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    notebook = build_notebook(root)
    output_path = root / "Project_All_Code_Classes.ipynb"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, ensure_ascii=False, indent=1)
    print(output_path)
    print(f"cells={len(notebook['cells'])}")


if __name__ == "__main__":
    main()
