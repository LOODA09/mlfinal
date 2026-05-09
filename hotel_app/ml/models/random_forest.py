from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from hotel_app.ml.models.base import BaseHotelModel


class RandomForestModel(BaseHotelModel):
    """Random forest with search-based tuning for the strongest tabular baseline.

    Doctor-facing notes:
    - estimator: ``RandomForestClassifier``
    - probability path: averaged tree probabilities
    - balancing: ``class_weight='balanced_subsample'``
    - tuning: ``RandomizedSearchCV`` over tree count, depth, and split controls
    """
    name = "Random Forest"

    def get_estimator(self) -> RandomizedSearchCV:
        base_estimator = RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        return RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions={
                "n_estimators": [120, 180, 240, 320],
                "max_depth": [12, 18, 24, None],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [0.3, 0.35, 0.45, "sqrt"],
                "bootstrap": [True, False],
            },
            n_iter=8,
            scoring="accuracy",
            cv=3,
            random_state=42,
            n_jobs=-1,
            refit=True,
        )
