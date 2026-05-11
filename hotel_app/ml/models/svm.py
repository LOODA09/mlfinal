from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from hotel_app.ml.models.base import BaseHotelModel, SubsampledEstimatorWrapper


class SVMModel(BaseHotelModel):
    """RBF-kernel support vector machine with bounded training size.

    Doctor-facing notes:
    - estimator: ``SVC(kernel='rbf', probability=True)`` on a stratified subset
    - probability path: native ``predict_proba`` from the fitted SVC
    - balancing: ``class_weight='balanced'``
    - tuning: ``GridSearchCV`` over ``C`` and ``gamma``
    """
    name = "SVM"

    def get_estimator(self) -> GridSearchCV:
        base_estimator = SubsampledEstimatorWrapper(
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
            max_samples=8000,
            random_state=42,
        )
        return GridSearchCV(
            estimator=base_estimator,
            param_grid={
                "estimator__C": [2.0, 5.0],
                "estimator__gamma": ["scale", 0.01],
            },
            scoring="accuracy",
            cv=3,
            n_jobs=1,
            refit=True,
        )
