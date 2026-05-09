from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from hotel_app.ml.models.base import BaseHotelModel


class SVMModel(BaseHotelModel):
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
