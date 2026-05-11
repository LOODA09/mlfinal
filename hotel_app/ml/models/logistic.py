from sklearn.linear_model import LogisticRegression
from hotel_app.ml.models.base import BaseHotelModel


class LogisticRegressionModel(BaseHotelModel):
    """Logistic regression baseline with native sigmoid probabilities.

    Doctor-facing notes:
    - estimator: ``LogisticRegression``
    - probability path: native ``predict_proba`` from logistic sigmoid
    - max_iter raised to 1000 to avoid convergence warnings
    """
    name = "Logistic Regression"

    def get_estimator(self) -> LogisticRegression:
        return LogisticRegression(
            max_iter=5000,
            random_state=42,
        )
