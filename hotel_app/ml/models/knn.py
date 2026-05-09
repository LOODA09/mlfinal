from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from hotel_app.ml.models.base import BaseHotelModel, BalancedClassifierWrapper


class KNNModel(BaseHotelModel):
    """Tuned K-nearest neighbors classifier on scaled features.

    Doctor-facing notes:
    - estimator: ``KNeighborsClassifier``
    - probability path: native neighbor vote probabilities
    - balancing: oversampling wrapper before fit
    - tuning: ``GridSearchCV`` over neighbors, distance weighting, and metric
    """
    name = "KNN"

    def get_estimator(self) -> GridSearchCV:
        base_estimator = BalancedClassifierWrapper(
            KNeighborsClassifier(),
            strategy="oversample",
            random_state=42,
        )
        return GridSearchCV(
            estimator=base_estimator,
            param_grid={
                "estimator__n_neighbors": [15, 21, 27],
                "estimator__weights": ["distance", "uniform"],
                "estimator__p": [1, 2],
                "estimator__leaf_size": [20, 30],
            },
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            refit=True,
        )
