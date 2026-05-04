from sklearn.ensemble import StackingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from hotel_app.ml.models.base import BaseHotelModel
from hotel_app.ml.models.xgboost_model import XGBoostModel


class StackingEnsembleModel(BaseHotelModel):
    name = "Stacking Ensemble"

    def get_estimator(self) -> StackingClassifier:
        estimators = [
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=250,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=250,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
            ("xgboost", XGBoostModel().get_estimator()),
            ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            stack_method="predict_proba",
            n_jobs=1,
            passthrough=False,
        )
