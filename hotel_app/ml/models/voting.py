from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from hotel_app.ml.models.base import BaseHotelModel
from hotel_app.ml.models.xgboost_model import XGBoostModel


class VotingEnsembleModel(BaseHotelModel):
    name = "Voting Ensemble"

    def get_estimator(self) -> VotingClassifier:
        estimators = [
            ("logistic", LogisticRegression(max_iter=1000, random_state=42)),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=350,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            ("xgboost", XGBoostModel().get_estimator()),
        ]
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
