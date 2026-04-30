# Hotel-Booking-Cancellation-Prediction
This notebook contains an exploratory data analysis of a hotel booking dataset, followed by feature engineering and then the building and evaluation of a suite of machine learning classifiers to predict which bookings will result in cancellations. Achieving a 100% accuracy in predicting booking cancellations using ensemble based machine learning classifiers.

The ML Models built and evaluated in this project are the following: Decision Tree Classifier, Random Forest Classifier, Ada Boost Classifier, Gradient Boosting Classifier, XgBoost, Cat Boost, LGBM, Voting Classifier, Logistic Regression, KNN, ANN

## Class-based application

The project now includes a real object-oriented implementation:

- `hotel_cancellation_oop.py` contains classes for the original notebook EDA, data cleaning, feature engineering, every model, evaluation metrics, training, testing, k-fold cross validation, ANN/RNN, and SHAP helpers.
- `streamlit_app.py` contains a new standalone class-based Streamlit dashboard with animated HTML/CSS UI components.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

Use these deployment settings:

- Repository: `HamzaMawazKhan/Hotel-Booking-Cancellation-Prediction`
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python version: choose Python 3.12 in Advanced settings

The repository includes `requirements.txt` and `.streamlit/config.toml`, which Streamlit Community Cloud reads during deployment.

TensorFlow, XGBoost, CatBoost, and LightGBM are optional because Streamlit Cloud may use a Python version where their wheels are unavailable. The app deploys with the sklearn models by default. To use the optional models locally, install them manually:

```bash
pip install tensorflow xgboost catboost lightgbm
```
