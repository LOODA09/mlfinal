from sklearn.naive_bayes import GaussianNB
from hotel_app.ml.models.base import BaseHotelModel, BalancedClassifierWrapper


class NaiveBayesModel(BaseHotelModel):
    name = "Naive Bayes"

    def get_estimator(self) -> BalancedClassifierWrapper:
        return BalancedClassifierWrapper(
            GaussianNB(var_smoothing=1e-8),
            strategy="sample_weight",
            random_state=42,
        )
