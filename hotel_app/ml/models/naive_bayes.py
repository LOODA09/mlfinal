from sklearn.naive_bayes import GaussianNB
from hotel_app.ml.models.base import BaseHotelModel


class NaiveBayesModel(BaseHotelModel):
    name = "Naive Bayes"

    def get_estimator(self) -> GaussianNB:
        return GaussianNB(var_smoothing=1e-8)
