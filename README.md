# Hotel-Booking-Cancellation-Prediction

This project now uses a terminal-first ML workflow:

- Training, testing, 70/30 splitting, 5-fold cross-validation, model benchmarking, SHAP analysis, confusion matrices, and K-Means guest segmentation run in the terminal.
- The Streamlit app is prediction-only and loads saved training artifacts for a cleaner, more production-style UI.

## Models included

- ANN (`MLPClassifier`)
- KNN
- Decision Tree
- Random Forest
- Naive Bayes
- SVM
- Gradient Boosting
- Extra Trees
- XGBoost
- Voting Ensemble
- K-Means guest segmentation
- RNN is still supported only when TensorFlow is available in the environment

## Install

```bash
pip install -r requirements.txt
```

For TensorFlow-based ANN, RNN, and LSTM, use Python `3.11`.

This repo now pins Python `3.11.9` in both:

- `.python-version` for local tooling
- `runtime.txt` for Streamlit Community Cloud deployment

If Streamlit Cloud was previously running on Python `3.14`, redeploy or reboot the app so it rebuilds on `3.11.9` and installs TensorFlow.

## Train from terminal

```bash
python train_terminal.py --data hotel_bookings.csv --output artifacts --cv-folds 5
```

Artifacts are saved in:

- `artifacts/models/`
- `artifacts/reports/`
- `artifacts/plots/`

Saved outputs include:

- trained model files
- holdout metrics from the 30% test split
- 5-fold cross-validation results
- training time and inference time metrics
- model complexity proxies
- confusion matrices
- SHAP summary and dependence plots
- guest segmentation report and plot

## Run prediction UI

After training:

```bash
streamlit run streamlit_app.py
```

If artifacts are missing, the UI will ask you to train first.

## Latest local run on this machine

The latest full-dataset run used:

- Train split: 70%
- Test split: 30%
- Cross-validation: 5-fold
- Best holdout model: `Random Forest`
- Cloud deployment model: `ANN`

Best honest holdout scores from the saved run:

- Accuracy: `0.8941`
- Precision: `0.8962`
- Recall: `0.8078`
- F1: `0.8497`
- ROC-AUC: `0.9584`

These numbers come directly from the real training run saved in `artifacts/reports/holdout_summary.csv`. No dummy metrics are used.
