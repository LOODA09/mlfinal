from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from hotel_app.ml.models.base import BaseHotelModel


class DecisionTreeModel(BaseHotelModel):
    """Single-tree classifier tuned for an interpretable baseline.

    Doctor-facing notes:
    - estimator: ``DecisionTreeClassifier``
    - probability path: native leaf-class probabilities
    - balancing: ``class_weight='balanced'``
    - tuning: ``GridSearchCV`` over depth and split/leaf controls
    """
    name = "Decision Tree"

    def get_estimator(self) -> GridSearchCV:
        base_estimator = DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        )
        return GridSearchCV(
            estimator=base_estimator,
            param_grid={
                "criterion": ["gini", "entropy"],
                "max_depth": [8, 12, 16],
                "min_samples_split": [2, 8],
                "min_samples_leaf": [4, 8, 12],
            },
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            refit=True,
        )
