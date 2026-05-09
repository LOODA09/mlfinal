from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


ROOT = Path(__file__).resolve().parent
HIGH_SCORE_DIR = ROOT / "artifacts" / "reports"
HONEST_DIR = ROOT / "artifacts_honest" / "reports"
OUTPUT_PATH = HIGH_SCORE_DIR / "detailed_model_explanation_report.pdf"


MODEL_NOTES = {
    "Logistic Regression": {
        "class_file": "hotel_app/ml/models/logistic.py",
        "training_setup": (
            "Uses LogisticRegression with class_weight='balanced', solver='lbfgs', "
            "max_iter=3000, and GridSearchCV over C values [0.35, 0.75, 1.25, 2.0]. "
            "Its probability output is native because logistic regression is built on the sigmoid function."
        ),
        "how_it_works": (
            "This model learns one weighted linear equation across all transformed features. "
            "The weighted sum is then passed through the sigmoid function so the output becomes a probability "
            "between 0 and 1. In plain language, it asks whether the booking sits on the cancel or not-cancel "
            "side of a learned decision boundary."
        ),
        "why_fit": (
            "It is a strong baseline for this project because the preprocessed data contains many one-hot-encoded "
            "categorical indicators, and logistic regression handles that style of feature space efficiently."
        ),
        "why_metrics": (
            "Its benchmark score becomes perfect in the high-score mode because leakage features such as "
            "reservation_status make the classes almost directly separable with a simple linear boundary. "
            "That does not mean the model is the most realistic deployment choice; it means the benchmark task "
            "becomes easy once future-known signals are present."
        ),
    },
    "Decision Tree": {
        "class_file": "hotel_app/ml/models/decision_tree.py",
        "training_setup": (
            "Uses DecisionTreeClassifier with class_weight='balanced' and GridSearchCV over criterion, max_depth, "
            "min_samples_split, and min_samples_leaf. The saved tree is intentionally tuned rather than left at defaults."
        ),
        "how_it_works": (
            "A decision tree repeatedly splits the data using if-then rules. Each split chooses a feature and threshold "
            "that reduces class impurity the most. The final leaf stores class proportions, which become probabilities."
        ),
        "why_fit": (
            "It matches the hotel cancellation problem well when there are strong threshold effects such as long lead time, "
            "prior cancellation history, or specific categorical patterns."
        ),
        "why_metrics": (
            "Its perfect high-score benchmark is expected because one tree can isolate leakage-heavy variables very quickly. "
            "In a realistic honest setting, a single tree usually becomes less stable than ensemble methods because it can "
            "memorize narrow patterns."
        ),
    },
    "Naive Bayes": {
        "class_file": "hotel_app/ml/models/naive_bayes.py",
        "training_setup": (
            "Uses GaussianNB inside a balancing wrapper that applies balanced sample weights. "
            "This keeps the minority cancellation class from being ignored."
        ),
        "how_it_works": (
            "Naive Bayes estimates how likely each feature value is under each class and then combines them under a "
            "conditional-independence assumption. Even when the assumption is not exactly true, it can still work well "
            "if a few features are extremely informative."
        ),
        "why_fit": (
            "It is useful in this project as a lightweight probabilistic baseline and as a contrast to more complex models."
        ),
        "why_metrics": (
            "Its perfect benchmark score happens because leakage features are so informative that even a simple probabilistic "
            "model can separate the classes almost completely. That performance should therefore be interpreted as a property "
            "of the benchmark mode, not as proof that Naive Bayes is the best real-world model."
        ),
    },
    "SVM": {
        "class_file": "hotel_app/ml/models/svm.py",
        "training_setup": (
            "Uses a LinearSVC with class_weight='balanced', wrapped by CalibratedClassifierCV, then tuned with GridSearchCV "
            "over C values [0.35, 0.75, 1.25, 2.0]. The calibration layer produces probability-style outputs."
        ),
        "how_it_works": (
            "A support vector machine looks for the separating hyperplane with the largest margin between classes. "
            "The calibrated wrapper then maps its scores into probabilities so the project can compute ROC-AUC, log loss, "
            "and probability-driven explanations."
        ),
        "why_fit": (
            "Because the project uses scaling and high-dimensional encoded features, a linear margin model is a sensible option."
        ),
        "why_metrics": (
            "In high-score mode the classes are nearly linearly separable once leakage variables are included, so SVM reaches "
            "perfect scores. In an honest mode, its performance typically falls because the real task is noisier and less "
            "cleanly separable."
        ),
    },
    "Random Forest": {
        "class_file": "hotel_app/ml/models/random_forest.py",
        "training_setup": (
            "Uses RandomForestClassifier with class_weight='balanced_subsample' and RandomizedSearchCV. The search explores "
            "n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, and bootstrap."
        ),
        "how_it_works": (
            "Random forest trains many decision trees on resampled data and randomized feature subsets, then averages their "
            "predictions. This reduces the variance of a single decision tree and captures non-linear interactions without "
            "requiring hand-crafted equations."
        ),
        "why_fit": (
            "This is the strongest honest model for the project because hotel cancellation is a tabular classification problem "
            "with mixed numeric and categorical signals, non-linear interactions, and threshold-like behavior. Random forests "
            "are especially strong under those conditions."
        ),
        "why_metrics": (
            "It reaches perfect benchmark scores in the high-score mode for the same leakage reason as the other models. "
            "More importantly, it remains the best honest model with accuracy 0.8802 and ROC-AUC 0.9492 because it can model "
            "complex interactions among lead time, booking history, market segment, grouped country, and other pre-arrival signals."
        ),
    },
    "LightGBM": {
        "class_file": "hotel_app/ml/models/lightgbm.py",
        "training_setup": (
            "Uses LGBMClassifier with objective='binary', learning_rate=0.05, n_estimators=400, num_leaves=31, "
            "min_child_samples=40, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, and class_weight='balanced'."
        ),
        "how_it_works": (
            "LightGBM is a gradient boosting tree method. Instead of averaging many independent trees, it builds trees "
            "sequentially so each new tree focuses on the errors left by the previous ones."
        ),
        "why_fit": (
            "Boosted trees are usually excellent for structured business datasets like hotel bookings because they capture "
            "non-linear relationships with strong predictive efficiency."
        ),
        "why_metrics": (
            "Its perfect benchmark score reflects how aggressively boosting can exploit leakage features. In a leakage-free "
            "setting it would usually be competitive, but the latest honest artifact bundle does not currently publish a "
            "fresh LightGBM row, so the report should not overclaim a realistic score for it."
        ),
    },
    "ANN": {
        "class_file": "hotel_app/ml/models/ann.py",
        "training_setup": (
            "Uses TensorFlow KerasTabularClassifier in ANN mode when TensorFlow is available. The network uses dense layers, "
            "batch normalization, dropout, Adam optimization, early stopping, learning-rate reduction, and class-weighted training. "
            "The fallback path is an MLPClassifier wrapped with oversampling."
        ),
        "how_it_works": (
            "A feed-forward neural network learns many weighted non-linear transformations of the input. Each hidden layer builds "
            "a richer representation, and the final neuron uses a sigmoid activation to produce a cancellation probability."
        ),
        "why_fit": (
            "ANN is useful here because it can absorb many engineered features and encoded categories without manual interaction terms, "
            "though tabular tree ensembles often remain stronger."
        ),
        "why_metrics": (
            "In the high-score mode, ANN is essentially given a nearly solved problem because leakage features dominate, so it scores "
            "almost perfectly. In the honest mode it remains competitive at accuracy 0.8533 and ROC-AUC 0.9399, which is strong but "
            "still below Random Forest because dense networks do not exploit tabular threshold structure as naturally as tree ensembles."
        ),
    },
    "RNN": {
        "class_file": "hotel_app/ml/models/rnn.py",
        "training_setup": (
            "Uses TensorFlow KerasTabularClassifier in RNN mode with bidirectional SimpleRNN layers, batch normalization, dropout, "
            "Adam optimization, early stopping, learning-rate reduction, and class-weighted training."
        ),
        "how_it_works": (
            "An RNN processes inputs as a sequence and updates an internal state as it moves through the feature order. "
            "In this project the tabular feature vector is reshaped into a pseudo-sequence so the recurrent layer can be tested."
        ),
        "why_fit": (
            "It is included for deep-learning comparison and to show that sequence-style models can be evaluated, but hotel booking "
            "rows are not true time series, so this is more of an experimental model than the natural first choice."
        ),
        "why_metrics": (
            "RNN reaches perfect high-score results only because the benchmark mode is leakage-heavy. In the honest mode it drops to "
            "accuracy 0.8073 and ROC-AUC 0.8935 because the recurrent memory mechanism does not bring a strong advantage on non-sequential tabular data."
        ),
    },
    "LSTM": {
        "class_file": "hotel_app/ml/models/lstm.py",
        "training_setup": (
            "Uses TensorFlow KerasTabularClassifier in LSTM mode with bidirectional LSTM layers, batch normalization, dropout, "
            "Adam optimization, early stopping, learning-rate reduction, and class-weighted training."
        ),
        "how_it_works": (
            "LSTM is a more advanced recurrent network with input, forget, and output gates. These gates help the network decide "
            "what information to keep or discard over long sequences."
        ),
        "why_fit": (
            "It is valuable for comparison because many examiners expect to see a deep sequential model, but this dataset does not "
            "naturally contain ordered temporal sequences at the row level."
        ),
        "why_metrics": (
            "Its benchmark result is near-perfect because leakage makes the task easy. In honest mode it reaches accuracy 0.8271 and "
            "ROC-AUC 0.9048, which is respectable but still below ANN and Random Forest. The reason is not that LSTM is weak in general; "
            "it is that the project data is tabular and not truly sequential, so the extra recurrent gating does not translate into a real advantage."
        ),
    },
    "KNN": {
        "class_file": "hotel_app/ml/models/knn.py",
        "training_setup": (
            "Uses KNeighborsClassifier inside an oversampling wrapper, then GridSearchCV over n_neighbors, distance weighting, Minkowski p, "
            "and leaf_size. It relies heavily on StandardScaler in preprocessing."
        ),
        "how_it_works": (
            "KNN does not learn explicit coefficients or trees. It stores the training data and classifies a new booking by looking at the "
            "closest examples in feature space."
        ),
        "why_fit": (
            "It is a useful comparison model because it tests whether cancellations cluster locally among similar bookings."
        ),
        "why_metrics": (
            "It is the weakest model in the high-score table at 0.9573 accuracy even though that still looks strong. That happens because "
            "distance-based methods struggle more in high-dimensional one-hot encoded spaces. Even with scaling and oversampling, nearest-neighbor "
            "methods are more sensitive to sparse categorical expansion than tree ensembles or linear models."
        ),
    },
}


def wrapped_lines(text: str, width: int = 112) -> str:
    return fill(text, width=width)


def add_text_block(ax: plt.Axes, y: float, title: str, body: str, width: int = 112, title_size: int = 12, body_size: int = 9.5) -> float:
    ax.text(0.05, y, title, fontsize=title_size, fontweight="bold", va="top")
    y -= 0.03
    wrapped = wrapped_lines(body, width=width)
    ax.text(0.05, y, wrapped, fontsize=body_size, va="top")
    line_count = wrapped.count("\n") + 1
    return y - (line_count * 0.027) - 0.03


def format_metric(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def load_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    high = pd.read_csv(HIGH_SCORE_DIR / "holdout_summary.csv")
    honest = pd.read_csv(HONEST_DIR / "holdout_summary.csv")
    high_meta = json.loads((HIGH_SCORE_DIR / "metadata.json").read_text(encoding="utf-8-sig"))
    honest_meta = json.loads((HONEST_DIR / "metadata.json").read_text(encoding="utf-8-sig"))
    return high, honest, high_meta, honest_meta


def build_summary_table(high: pd.DataFrame, honest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    honest_by_model = honest.set_index("model") if not honest.empty else pd.DataFrame()
    for _, row in high.iterrows():
        honest_row = honest_by_model.loc[row["model"]] if row["model"] in honest_by_model.index else None
        rows.append(
            {
                "model": row["model"],
                "benchmark_accuracy": row["accuracy"],
                "benchmark_f1": row["f1"],
                "benchmark_roc_auc": row["roc_auc"],
                "honest_accuracy": honest_row["accuracy"] if honest_row is not None else None,
                "honest_f1": honest_row["f1"] if honest_row is not None else None,
                "honest_roc_auc": honest_row["roc_auc"] if honest_row is not None else None,
            }
        )
    return pd.DataFrame(rows)


def make_cover(pdf: PdfPages, high_meta: dict, honest_meta: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    y = 0.95
    ax.text(0.05, y, "Detailed Model Explanation Report", fontsize=22, fontweight="bold", va="top")
    y -= 0.06
    ax.text(0.05, y, "Hotel Booking Cancellation Prediction Project", fontsize=15, va="top")
    y -= 0.06
    y = add_text_block(
        ax,
        y,
        "What this PDF explains",
        "This report explains how each model in the project works, how it was trained, why it behaves the way it does on this dataset, "
        "and why some models score almost perfectly in the high-score benchmark while the honest mode remains lower and more realistic.",
    )
    y = add_text_block(
        ax,
        y,
        "Important interpretation",
        "The project intentionally keeps two modes. High-Score Benchmark keeps leakage-heavy features so the visible benchmark reaches very high accuracy. "
        "Honest Prediction removes future-known fields and reflects the realistic pre-arrival problem. Therefore, perfect benchmark scores are not proof "
        "that the same models would achieve perfect real-world prediction.",
    )
    y = add_text_block(
        ax,
        y,
        "Current saved winners",
        f"High-score benchmark winner: {high_meta.get('best_model', 'N/A')}. Honest-mode winner: {honest_meta.get('best_model', 'N/A')}. "
        f"The honest-mode winner is the most important one for real deployment because it predicts from information available before the guest outcome is known.",
    )
    pdf.savefig(fig)
    plt.close(fig)


def make_methodology(pdf: PdfPages, high_meta: dict, honest_meta: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    y = 0.96
    ax.text(0.05, y, "Training and Evaluation Methodology", fontsize=18, fontweight="bold", va="top")
    y -= 0.05
    y = add_text_block(
        ax,
        y,
        "Shared preprocessing",
        "All models use the same preprocessing pipeline. Numeric variables are standardized with StandardScaler, categorical variables are one-hot encoded after imputation, "
        "and the project adds engineered behavioral features such as total nights, booking history ratios, guest composition, and pricing-style ratios where appropriate.",
    )
    y = add_text_block(
        ax,
        y,
        "Balancing strategy",
        "Class balancing is handled through native class weights, balanced sample weights, oversampling wrappers, or class-weighted TensorFlow training depending on the estimator. "
        "This prevents the majority class from dominating the fit.",
    )
    y = add_text_block(
        ax,
        y,
        "Probability and sigmoid path",
        "Models with predict_proba expose positive-class probabilities directly. Logistic Regression is inherently based on the sigmoid function. "
        "ANN, RNN, and LSTM use a final sigmoid output layer. Score-based models can also be mapped to probability-style outputs through the shared sigmoid helper in hotel_app/ml/data.py.",
    )
    y = add_text_block(
        ax,
        y,
        "Benchmark split and runtime context",
        f"Both modes use a 70/30 train-test split on {honest_meta.get('total_rows', 'N/A')} rows. "
        f"The saved runtime environment is Python {honest_meta.get('python_version', 'N/A')} and TensorFlow {honest_meta.get('tensorflow_version', 'off')}. "
        f"High-score mode saves {high_meta.get('cross_validation_folds', 'N/A')}-fold validation metadata, while the honest mode currently saves {honest_meta.get('cross_validation_folds', 'N/A')}-fold metadata.",
    )
    y = add_text_block(
        ax,
        y,
        "Why the metrics differ by mode",
        "High-score mode includes reservation-status-derived information and similar notebook-matched benchmark features, so the target is almost directly recoverable. "
        "Honest mode removes those leakage-heavy columns. That makes the task genuinely predictive instead of retrospective, so the metrics fall to a more believable level.",
    )
    pdf.savefig(fig)
    plt.close(fig)


def make_summary_table(pdf: PdfPages, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    display = summary.copy()
    numeric_cols = [c for c in display.columns if c != "model"]
    for col in numeric_cols:
        display[col] = display[col].apply(format_metric)
    table = ax.table(
        cellText=display.values,
        colLabels=[
            "Model",
            "Benchmark Acc",
            "Benchmark F1",
            "Benchmark ROC-AUC",
            "Honest Acc",
            "Honest F1",
            "Honest ROC-AUC",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.65)
    ax.set_title("Saved Metrics by Model and Mode", fontsize=16, pad=18)
    pdf.savefig(fig)
    plt.close(fig)


def make_model_page(pdf: PdfPages, model_name: str, high: pd.DataFrame, honest: pd.DataFrame) -> None:
    high_row = high.set_index("model").loc[model_name]
    honest_by_model = honest.set_index("model") if not honest.empty else pd.DataFrame()
    honest_row = honest_by_model.loc[model_name] if model_name in honest_by_model.index else None
    notes = MODEL_NOTES[model_name]

    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    y = 0.96
    ax.text(0.05, y, model_name, fontsize=20, fontweight="bold", va="top")
    y -= 0.045
    ax.text(0.05, y, f"Class file: {notes['class_file']}", fontsize=10, family="monospace", va="top")
    y -= 0.05

    y = add_text_block(ax, y, "How this model is trained in the project", notes["training_setup"])
    y = add_text_block(ax, y, "How the algorithm works", notes["how_it_works"])
    y = add_text_block(ax, y, "Why it is or is not a strong fit for this project", notes["why_fit"])

    metrics_text = (
        f"High-score benchmark metrics: accuracy {format_metric(high_row['accuracy'])}, precision {format_metric(high_row['precision'])}, "
        f"recall {format_metric(high_row['recall'])}, F1 {format_metric(high_row['f1'])}, ROC-AUC {format_metric(high_row['roc_auc'])}, "
        f"training time {format_metric(high_row['training_time_sec'])} seconds, inference {format_metric(high_row['inference_ms_per_row'])} ms per row."
    )
    if honest_row is not None:
        metrics_text += (
            f" Honest-mode metrics: accuracy {format_metric(honest_row['accuracy'])}, precision {format_metric(honest_row['precision'])}, "
            f"recall {format_metric(honest_row['recall'])}, F1 {format_metric(honest_row['f1'])}, ROC-AUC {format_metric(honest_row['roc_auc'])}, "
            f"training time {format_metric(honest_row['training_time_sec'])} seconds, inference {format_metric(honest_row['inference_ms_per_row'])} ms per row."
        )
    else:
        metrics_text += " This model does not currently have a published row in the latest honest artifact table."
    y = add_text_block(ax, y, "What the saved metrics show", metrics_text)
    y = add_text_block(ax, y, "Why the model got these metrics", notes["why_metrics"])

    pdf.savefig(fig)
    plt.close(fig)


def make_conclusion(pdf: PdfPages, honest: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    y = 0.96
    ax.text(0.05, y, "Conclusion", fontsize=18, fontweight="bold", va="top")
    y -= 0.05
    top_honest = honest.sort_values(["f1", "accuracy", "roc_auc"], ascending=[False, False, False]).iloc[0]
    y = add_text_block(
        ax,
        y,
        "Best realistic model",
        f"Random Forest is the strongest realistic model in the current honest artifact set. It achieves accuracy {format_metric(top_honest['accuracy'])}, "
        f"F1 {format_metric(top_honest['f1'])}, and ROC-AUC {format_metric(top_honest['roc_auc'])}. It is best because hotel cancellation prediction is fundamentally a "
        f"tabular, interaction-heavy problem, and random forests model that structure very well without needing a true temporal sequence.",
    )
    y = add_text_block(
        ax,
        y,
        "Why deep learning is not automatically best here",
        "ANN, RNN, and LSTM are valuable for experimentation and comparison, but the dataset is not a natural image, audio, or real time-series problem. "
        "Because of that, deep networks do not automatically dominate classical tabular models. ANN remains competitive, while RNN and LSTM mainly serve as deep-learning baselines.",
    )
    y = add_text_block(
        ax,
        y,
        "How to explain the near-perfect benchmark to an examiner",
        "The correct explanation is that the benchmark mode intentionally reproduces a notebook-style high-score setup with leakage-heavy signals. "
        "Those metrics are useful for showing model capacity and benchmark reproducibility, but the honest mode is the better representation of deployment realism.",
    )
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf() -> Path:
    high, honest, high_meta, honest_meta = load_rows()
    summary = build_summary_table(high, honest)
    with PdfPages(OUTPUT_PATH) as pdf:
        make_cover(pdf, high_meta, honest_meta)
        make_methodology(pdf, high_meta, honest_meta)
        make_summary_table(pdf, summary)
        for model_name in summary["model"].tolist():
            if model_name in MODEL_NOTES:
                make_model_page(pdf, model_name, high, honest)
        make_conclusion(pdf, honest)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_pdf()
    print(path)
