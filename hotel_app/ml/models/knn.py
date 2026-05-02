from sklearn.neighbors import KNeighborsClassifier
from hotel_app.ml.models.base import BaseHotelModel


class KNNModel(BaseHotelModel):
    name = "KNN"

    def get_estimator(self) -> KNeighborsClassifier:
        return KNeighborsClassifier()
