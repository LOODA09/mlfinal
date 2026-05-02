from sklearn.tree import DecisionTreeClassifier
from hotel_app.ml.models.base import BaseHotelModel


class DecisionTreeModel(BaseHotelModel):
    name = "Decision Tree"

    def get_estimator(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(random_state=42)
