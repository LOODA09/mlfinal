from sklearn.ensemble import RandomForestClassifier
from hotel_app.ml.models.base import BaseHotelModel


class RandomForestModel(BaseHotelModel):
    name = "Random Forest"

    def get_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=450,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
