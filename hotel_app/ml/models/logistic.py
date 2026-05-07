from sklearn.linear_model import LogisticRegression
from hotel_app.ml.models.base import BaseHotelModel


class LogisticRegressionModel(BaseHotelModel):
    name = "Logistic Regression"

    def get_estimator(self) -> LogisticRegression:
        return LogisticRegression(
            C=0.75,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
