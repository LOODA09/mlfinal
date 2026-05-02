from sklearn.linear_model import LogisticRegression
from hotel_app.ml.models.base import BaseHotelModel


class LogisticRegressionModel(BaseHotelModel):
    name = "Logistic Regression"

    def get_estimator(self) -> LogisticRegression:
        return LogisticRegression(max_iter=1000, random_state=42)
