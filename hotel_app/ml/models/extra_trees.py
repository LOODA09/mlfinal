from sklearn.ensemble import ExtraTreesClassifier
from hotel_app.ml.models.base import BaseHotelModel


class ExtraTreesModel(BaseHotelModel):
    name = "Extra Trees"

    def get_estimator(self) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
