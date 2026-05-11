from sklearn.svm import SVC
from hotel_app.ml.models.base import BaseHotelModel


class SVMModel(BaseHotelModel):
    """Support vector machine with default sklearn parameters.

    Doctor-facing notes:
    - estimator: ``SVC(probability=True)``
    - probability path: native ``predict_proba`` from fitted SVC
    - uses default hyperparameters matching the project notebook
    """
    name = "SVM"

    def get_estimator(self) -> SVC:
        return SVC(
            probability=True,
            random_state=42,
        )
