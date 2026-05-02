from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from hotel_app.ml.models.base import BaseHotelModel


class SVMModel(BaseHotelModel):
    name = "SVM"

    def get_estimator(self) -> CalibratedClassifierCV:
        return CalibratedClassifierCV(estimator=LinearSVC(C=1.0, random_state=42, dual="auto"), cv=3)
