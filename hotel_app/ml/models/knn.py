from sklearn.neighbors import KNeighborsClassifier
from hotel_app.ml.models.base import BaseHotelModel, BalancedClassifierWrapper


class KNNModel(BaseHotelModel):
    name = "KNN"

    def get_estimator(self) -> BalancedClassifierWrapper:
        return BalancedClassifierWrapper(
            KNeighborsClassifier(n_neighbors=21, weights="distance", p=2),
            strategy="oversample",
            random_state=42,
        )
