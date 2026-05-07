from sklearn.ensemble import RandomForestClassifier
from hotel_app.ml.models.base import BaseHotelModel


class RandomForestModel(BaseHotelModel):
    name = "Random Forest"

    def get_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=700,
            max_features=0.4,
            min_samples_leaf=1,
            max_depth=28,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
