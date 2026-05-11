from sklearn.neighbors import KNeighborsClassifier
from hotel_app.ml.models.base import BaseHotelModel


class KNNModel(BaseHotelModel):
    """K-nearest neighbors classifier with default sklearn parameters.

    Doctor-facing notes:
    - estimator: ``KNeighborsClassifier``
    - probability path: native neighbor vote probabilities
    - uses default hyperparameters matching the project notebook
    """
    name = "KNN"

    def get_estimator(self) -> KNeighborsClassifier:
        return KNeighborsClassifier()
