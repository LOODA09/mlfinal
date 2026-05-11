from sklearn.tree import DecisionTreeClassifier
from hotel_app.ml.models.base import BaseHotelModel


class DecisionTreeModel(BaseHotelModel):
    """Single-tree classifier with default sklearn parameters.

    Doctor-facing notes:
    - estimator: ``DecisionTreeClassifier``
    - probability path: native leaf-class probabilities
    - uses default hyperparameters matching the project notebook
    """
    name = "Decision Tree"

    def get_estimator(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            random_state=42,
        )
