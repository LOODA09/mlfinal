from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from hotel_app.ml.models.base import BaseHotelModel


class AdaBoostModel(BaseHotelModel):
    name = "AdaBoost"

    def get_estimator(self) -> AdaBoostClassifier:
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        try:
            return AdaBoostClassifier(estimator=tree, random_state=42)
        except TypeError:
            return AdaBoostClassifier(base_estimator=tree, random_state=42)
