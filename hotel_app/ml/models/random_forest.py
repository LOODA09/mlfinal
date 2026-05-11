from sklearn.ensemble import RandomForestClassifier
from hotel_app.ml.models.base import BaseHotelModel


class RandomForestModel(BaseHotelModel):
    """Random forest classifier with default sklearn parameters.

    Doctor-facing notes:
    - estimator: ``RandomForestClassifier``
    - probability path: averaged tree probabilities
    - uses default hyperparameters matching the project notebook
    """
    name = "Random Forest"

    def get_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            random_state=42,
        )
