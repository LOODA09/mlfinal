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
    "hotel_app/ml/models/naive_bayes.py",
    "hotel_app/ml/models/svm.py",
    "hotel_app/ml/models/ann.py",
    "hotel_app/ml/models/rnn.py",
    "hotel_app/ml/models/lstm.py",
    "hotel_app/ml/models/lightgbm.py",
    "hotel_app/ml/models/xgboost_model.py",
    "hotel_app/ml/models/adaboost.py",
    "hotel_app/ml/models/gradient_boosting.py",
    "hotel_app/ml/models/extra_trees.py",
    "hotel_app/ml/models/voting.py",
    "hotel_app/ml/models/stacking.py",
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
    cells = [
        markdown_cell(
            "# Hotel Booking Cancellation Project: All Code in One Notebook\n\n"
            "This notebook was generated from the current project source so the whole "
            "implementation is available in one place.\n\n"
            "Preprocessing notes:\n"
            "- numeric features are normalized with `StandardScaler`\n"
            "- categorical features are one-hot encoded after imputation\n"
            "- class balancing is applied across the model set through native class weights, balanced sample weights, oversampling wrappers, and class-weighted deep learning fits\n"
            "- the project supports both a notebook-matched high-score benchmark mode and an honest future-booking mode\n\n"
            "Contents:\n"
            "- reporting and terminal scripts\n"
            "- Streamlit dashboard\n"
            "- service layer\n"
            "- machine learning pipeline\n"
            "- all model classes\n\n"
            "Each section below shows the source of one project file.\n"
        )
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
