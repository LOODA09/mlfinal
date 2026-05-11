from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts" / "reports"
HONEST_DIR = ROOT / "artifacts_honest" / "reports"
OUTPUT_PATH = ARTIFACTS_DIR / "model_parameter_reference.pdf"


MODEL_DETAILS = {
    "Logistic Regression": {
        "class_file": "hotel_app/ml/models/logistic.py",
        "parameters": (
            "Base estimator: LogisticRegression(max_iter=3000, class_weight='balanced', solver='lbfgs', random_state=42). "
            "Search space: C in [0.35, 0.75, 1.25, 2.0] through GridSearchCV(cv=3, scoring='accuracy')."
        ),
        "how_it_works": (
            "Logistic regression learns a weighted linear score and passes it through the sigmoid function to produce a binary probability. "
            "It is a strong baseline for sparse one-hot encoded hotel features because it is stable, fast, and naturally probabilistic."
        ),
        "why_picked": (
            "The class weight keeps the cancellation class from being under-penalized. A moderate C search was chosen instead of an extreme grid because the feature space "
            "is already standardized and wide, so the main question is how much regularization to allow rather than whether the model can represent non-linearity."
        ),
    },
    "KNN": {
        "class_file": "hotel_app/ml/models/knn.py",
        "parameters": (
            "Base estimator: KNeighborsClassifier() inside BalancedClassifierWrapper(strategy='oversample'). "
            "Search space: n_neighbors [15, 21, 27], weights ['distance', 'uniform'], p [1, 2], leaf_size [20, 30] via GridSearchCV(cv=3, scoring='accuracy')."
        ),
        "how_it_works": (
            "KNN predicts from nearby training examples in scaled feature space. It does not build an explicit formula, so the model depends strongly on normalization and the local geometry of bookings."
        ),
        "why_picked": (
            "Oversampling is used because KNN has no native class_weight parameter. The search focuses on neighbor count and weighting because those choices control the noise-vs-smoothness tradeoff. "
            "Higher neighbor counts were chosen over tiny ones because one-hot hotel data can be noisy at the individual-row level."
        ),
    },
    "Decision Tree": {
        "class_file": "hotel_app/ml/models/decision_tree.py",
        "parameters": (
            "Base estimator: DecisionTreeClassifier(class_weight='balanced', random_state=42). "
            "Search space: criterion ['gini', 'entropy'], max_depth [8, 12, 16], min_samples_split [2, 8], min_samples_leaf [4, 8, 12] via GridSearchCV(cv=3, scoring='accuracy')."
        ),
        "how_it_works": (
            "A decision tree creates if-then rules that split the hotel bookings into cleaner cancellation groups. The leaf class proportions become the prediction probabilities."
        ),
        "why_picked": (
            "The depth and minimum leaf constraints intentionally keep the tree from growing into a brittle memorization structure. This project has many categorical expansions, "
            "so a shallow-to-medium depth tree is a better bias-variance balance than an unrestricted one."
        ),
    },
    "Random Forest": {
        "class_file": "hotel_app/ml/models/random_forest.py",
        "parameters": (
            "Base estimator: RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=-1). "
            "RandomizedSearchCV explores n_estimators [120, 180, 240, 320], max_depth [12, 18, 24, None], min_samples_split [2, 4, 8], "
            "min_samples_leaf [1, 2, 4], max_features [0.3, 0.35, 0.45, 'sqrt'], bootstrap [True, False] with n_iter=8, cv=3, scoring='accuracy'."
        ),
        "how_it_works": (
            "Random forest averages many randomized decision trees. Each tree sees a different bootstrap sample and different feature subsets, so the final model captures non-linear interactions while reducing overfitting."
        ),
        "why_picked": (
            "This is the strongest honest model because hotel cancellation is a classic tabular problem with threshold effects, mixed feature types, and interactions among lead time, segment, history, and price. "
            "The search space was centered on medium-to-strong forests instead of tiny ones because the project benefits from ensemble stability more than from extreme simplicity."
        ),
    },
    "XGBoost": {
        "class_file": "hotel_app/ml/models/xgboost_model.py",
        "parameters": (
            "Estimator: XGBClassifier(booster='gbtree', learning_rate=0.05, max_depth=6, n_estimators=320, min_child_weight=2, "
            "subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5, eval_metric='logloss', tree_method='hist', n_jobs=1, random_state=42)."
        ),
        "how_it_works": (
            "XGBoost builds trees sequentially and each new tree concentrates on the residual errors left by the previous trees. "
            "It is one of the strongest off-the-shelf algorithms for structured business tables."
        ),
        "why_picked": (
            "The settings balance predictive power with runtime. The moderate learning rate and 320 trees give the booster enough capacity, "
            "while max_depth, child weight, subsampling, and regularization keep the model from becoming unnecessarily unstable."
        ),
    },
    "SVM": {
        "class_file": "hotel_app/ml/models/svm.py",
        "parameters": (
            "Base estimator: SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42) wrapped in "
            "SubsampledEstimatorWrapper(max_samples=20000, random_state=42). Search space: estimator__C [1.0, 3.0] and estimator__gamma ['scale', 0.01] "
            "through GridSearchCV(cv=3, scoring='accuracy', n_jobs=1)."
        ),
        "how_it_works": (
            "The RBF SVM creates a non-linear decision boundary by measuring similarity to support vectors in an implicit high-dimensional kernel space."
        ),
        "why_picked": (
            "An RBF kernel was chosen because you explicitly asked for the non-linear SVM version. The stratified subsampling wrapper is necessary because a full RBF SVM "
            "on the entire hotel-booking table would be impractically expensive. The compact C/gamma grid keeps the search realistic while still testing a stronger non-linear margin boundary."
        ),
    },
    "ANN": {
        "class_file": "hotel_app/ml/models/ann.py and hotel_app/ml/deep.py",
        "parameters": (
            "Wrapper defaults: KerasTabularClassifier(model_type='ann', epochs=120, batch_size=192, learning_rate=0.001, random_state=42). "
            "Architecture: Dense(128, relu, l2=1e-4) -> BatchNorm -> Dropout(0.25) -> Dense(64, relu, l2=1e-4) -> BatchNorm -> Dropout(0.15) "
            "-> Dense(24, relu, l2=1e-4) -> Dense(1, sigmoid). Optimizer: Adam(learning_rate=0.001, clipnorm=1.0). "
            "Training controls: validation_split=0.1, EarlyStopping(monitor='val_auc', patience=6), ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5), class_weight='balanced'."
        ),
        "how_it_works": (
            "The ANN stacks dense non-linear transformations on the tabular feature vector and ends with a sigmoid neuron that outputs cancellation probability."
        ),
        "why_picked": (
            "The hidden sizes are large enough to learn interactions but small enough to stay practical on this dataset. Batch normalization and dropout stabilize training, "
            "while early stopping on validation AUC keeps the network from running too long after it stops improving."
        ),
    },
    "RNN": {
        "class_file": "hotel_app/ml/models/rnn.py and hotel_app/ml/deep.py",
        "parameters": (
            "Wrapper defaults: KerasTabularClassifier(model_type='rnn', epochs=45, batch_size=192, learning_rate=0.001, random_state=42). "
            "Architecture: Bidirectional(SimpleRNN(40, activation='tanh', dropout=0.1, recurrent_dropout=0.1)) -> BatchNorm -> Dropout(0.25) "
            "-> Dense(24, relu, l2=1e-4) -> Dense(1, sigmoid). Optimizer and callbacks match the ANN path."
        ),
        "how_it_works": (
            "The RNN processes the feature vector as a pseudo-sequence and carries state through the sequence before making a final sigmoid-based decision."
        ),
        "why_picked": (
            "A modest hidden width of 40 was chosen because the input is not a true natural sequence. The goal was to test whether recurrent modeling adds value without creating an oversized unstable network."
        ),
    },
    "LSTM": {
        "class_file": "hotel_app/ml/models/lstm.py and hotel_app/ml/deep.py",
        "parameters": (
            "Wrapper defaults: KerasTabularClassifier(model_type='lstm', epochs=55, batch_size=192, learning_rate=0.001, random_state=42). "
            "Architecture: Bidirectional(LSTM(32, activation='tanh', dropout=0.1, recurrent_dropout=0.1)) -> BatchNorm -> Dropout(0.25) "
            "-> Dense(24, relu, l2=1e-4) -> Dense(1, sigmoid). Optimizer and callbacks match the ANN path."
        ),
        "how_it_works": (
            "LSTM is a gated recurrent model that can decide what information to keep or forget as it moves through the input sequence, then emits a final sigmoid probability."
        ),
        "why_picked": (
            "The LSTM is intentionally smaller than the ANN because recurrent layers are more parameter-heavy. Since the project data is tabular rather than naturally sequential, "
            "the settings were chosen to test the idea without overcommitting to a very large recurrent model."
        ),
    },
}


def wrap(text: str, width: int = 100) -> str:
    return fill(text, width=width)


def load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    high = pd.read_csv(ARTIFACTS_DIR / "holdout_summary.csv")
    honest = pd.read_csv(HONEST_DIR / "holdout_summary.csv")
    return high, honest


def metric_lookup(frame: pd.DataFrame, model_name: str) -> dict:
    if "model" not in frame.columns:
        return {}
    matched = frame[frame["model"] == model_name]
    if matched.empty:
        return {}
    return matched.iloc[0].to_dict()


def add_block(ax: plt.Axes, y: float, title: str, body: str, width: int = 100) -> float:
    ax.text(0.05, y, title, fontsize=12, fontweight="bold", va="top")
    y -= 0.03
    wrapped = wrap(body, width=width)
    ax.text(0.05, y, wrapped, fontsize=9.5, va="top")
    return y - (wrapped.count("\n") + 1) * 0.026 - 0.028


def format_metric(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def cover_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    y = 0.95
    ax.text(0.05, y, "Model Parameter Reference", fontsize=22, fontweight="bold", va="top")
    y -= 0.06
    ax.text(0.05, y, "Hotel Booking Cancellation Prediction Project", fontsize=15, va="top")
    y -= 0.07
    y = add_block(
        ax,
        y,
        "What this report covers",
        "This PDF explains the exact model settings used in the current project code, how each model works, "
        "and why those settings were chosen for hotel-booking cancellation prediction. It only documents models that are already in the project.",
    )
    y = add_block(
        ax,
        y,
        "How to read the parameter choices",
        "Some models use fixed parameters, while others use GridSearchCV or RandomizedSearchCV. "
        "For search-based models, this report explains both the base estimator and the search space because the search space itself is part of the design decision.",
    )
    y = add_block(
        ax,
        y,
        "Notebook ideas used selectively",
        "The project borrows only high-signal ideas from the final notebook, such as lead-time categories, stay-length buckets, guest-group buckets, "
        "first-time-visitor logic, and more robust price handling. It does not copy notebook code wholesale and does not add extra models that are not already part of the project.",
    )
    pdf.savefig(fig)
    plt.close(fig)


def summary_table_page(pdf: PdfPages, high: pd.DataFrame, honest: pd.DataFrame) -> None:
    rows = []
    for model_name in MODEL_DETAILS:
        high_row = metric_lookup(high, model_name)
        honest_row = metric_lookup(honest, model_name)
        rows.append(
            {
                "Model": model_name,
                "High-Score Acc": format_metric(high_row.get("accuracy")),
                "Honest Acc": format_metric(honest_row.get("accuracy")),
                "High-Score F1": format_metric(high_row.get("f1")),
                "Honest F1": format_metric(honest_row.get("f1")),
            }
        )
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    table = ax.table(
        cellText=frame.values,
        colLabels=frame.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    ax.set_title("Current Saved Metrics for the Documented Models", fontsize=16, pad=18)
    pdf.savefig(fig)
    plt.close(fig)


def model_page(pdf: PdfPages, model_name: str, high: pd.DataFrame, honest: pd.DataFrame) -> None:
    details = MODEL_DETAILS[model_name]
    high_row = metric_lookup(high, model_name)
    honest_row = metric_lookup(honest, model_name)

    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    y = 0.96
    ax.text(0.05, y, model_name, fontsize=19, fontweight="bold", va="top")
    y -= 0.04
    ax.text(0.05, y, f"Class file: {details['class_file']}", fontsize=10, family="monospace", va="top")
    y -= 0.05

    y = add_block(ax, y, "Model parameters / search space", details["parameters"])
    y = add_block(ax, y, "How this model works", details["how_it_works"])
    y = add_block(ax, y, "Why these parameters were chosen", details["why_picked"])

    metrics_text = (
        f"High-score benchmark: accuracy {format_metric(high_row.get('accuracy'))}, F1 {format_metric(high_row.get('f1'))}, "
        f"ROC-AUC {format_metric(high_row.get('roc_auc'))}."
    )
    if honest_row:
        metrics_text += (
            f" Honest mode: accuracy {format_metric(honest_row.get('accuracy'))}, F1 {format_metric(honest_row.get('f1'))}, "
            f"ROC-AUC {format_metric(honest_row.get('roc_auc'))}."
        )
    else:
        metrics_text += " This model does not currently have a published row in the latest honest artifact table."
    y = add_block(ax, y, "Current saved metrics", metrics_text)

    pdf.savefig(fig)
    plt.close(fig)


def conclusion_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    y = 0.95
    ax.text(0.05, y, "Parameter Selection Conclusion", fontsize=18, fontweight="bold", va="top")
    y -= 0.05
    y = add_block(
        ax,
        y,
        "Overall design logic",
        "The project uses a mixed strategy: search-based tuning for the classical models that are most sensitive to hyperparameter choice, "
        "and stable hand-picked architectures for the deep models where the design space is broader and more expensive to search."
    )
    y = add_block(
        ax,
        y,
        "Why Random Forest remains the best honest choice",
        "Random Forest receives the richest search space because it is the most natural fit for leakage-free hotel-booking tabular prediction. "
        "Its settings were chosen to balance depth, tree diversity, and stability rather than chase a tiny overfit benchmark gain."
    )
    y = add_block(
        ax,
        y,
        "Why the app keeps multiple model types",
        "Keeping linear, distance-based, tree-based, and deep-learning models is useful academically because it shows how the same hotel problem behaves under very different learning assumptions. "
        "The report therefore explains not only which model scores best, but also why the parameter choices differ across families."
    )
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf() -> Path:
    high, honest = load_metrics()
    with PdfPages(OUTPUT_PATH) as pdf:
        cover_page(pdf)
        summary_table_page(pdf, high, honest)
        for model_name in MODEL_DETAILS:
            model_page(pdf, model_name, high, honest)
        conclusion_page(pdf)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_pdf()
    print(path)
