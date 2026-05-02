from sklearn.ensemble import GradientBoostingClassifier
from hotel_app.ml.models.base import BaseHotelModel


class GradientBoostingModel(BaseHotelModel):
    name = "Gradient Boosting"

    def get_estimator(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(random_state=42)
