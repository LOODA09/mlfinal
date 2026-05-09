from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from hotel_app.ml.models.base import BaseHotelModel


class SVMModel(BaseHotelModel):
    """Linear support vector machine with calibrated probabilities.

    Doctor-facing notes:
    - estimator: ``LinearSVC`` wrapped by ``CalibratedClassifierCV``
    - probability path: calibrated probabilities; raw scores can also be
      converted with the shared sigmoid helper in ``hotel_app.ml.data``
    - balancing: ``class_weight='balanced'``
    - tuning: ``GridSearchCV`` over the margin penalty ``C``
    """
    name = "SVM"

    def get_estimator(self) -> GridSearchCV:
        base_estimator = CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", random_state=42, dual="auto"),
            cv=3,
        )
        return GridSearchCV(
            estimator=base_estimator,
            param_grid={
                "estimator__C": [0.35, 0.75, 1.25, 2.0],
            },
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            refit=True,
        )
