from sklearn.tree import DecisionTreeClassifier
from hotel_app.ml.models.base import BaseHotelModel


class DecisionTreeModel(BaseHotelModel):
    name = "Decision Tree"

    def get_estimator(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
        )
