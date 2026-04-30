# Hotel-Booking-Cancellation-Prediction
This notebook contains an exploratory data analysis of a hotel booking dataset, followed by feature engineering and then the building and evaluation of a suite of machine learning classifiers to predict which bookings will result in cancellations. Achieving a 100% accuracy in predicting booking cancellations using ensemble based machine learning classifiers.

The ML Models built and evaluated in this project are the following: Decision Tree Classifier, Random Forest Classifier, Ada Boost Classifier, Gradient Boosting Classifier, XgBoost, Cat Boost, LGBM, Voting Classifier, Logistic Regression, KNN, ANN

## Class-based application

The project now includes a real object-oriented implementation:

- `hotel_cancellation_oop.py` contains classes for the original notebook EDA, data cleaning, feature engineering, supported models, evaluation metrics, training, testing, k-fold cross validation, ANN/RNN, and SHAP helpers.
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

TensorFlow needs a compatible Python version. This project was verified locally with Python 3.11:

```bash
py -3.11 -m venv .venv311
.\.venv311\Scripts\activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

`requirements.txt` installs `xgboost==3.2.0` everywhere and installs `tensorflow==2.21.0` only on Python versions below 3.14. For Streamlit Community Cloud, set the Python version to 3.12 or 3.11 in the app's Advanced settings if you want ANN/RNN TensorFlow models available.

CatBoost and LightGBM were removed from the app because they are not used by the current deployment.

The app reports two score types after training:

- Holdout Evaluation: an honest test on rows not used for training.
- Full-Data In-Sample Fit: scored on the same rows used for training, useful to show model fit but not a real unseen test. Decision Tree can reach near-1 here without leakage.
