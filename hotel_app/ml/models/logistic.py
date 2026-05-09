from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from hotel_app.ml.models.base import BaseHotelModel


class LogisticRegressionModel(BaseHotelModel):
    """Tuned logistic baseline with native sigmoid probabilities.

    Doctor-facing notes:
    - estimator: ``LogisticRegression``
    - probability path: native ``predict_proba`` from logistic sigmoid
    - balancing: ``class_weight='balanced'``
    - tuning: ``GridSearchCV`` over the regularization strength ``C``
    """
    name = "Logistic Regression"

    def get_estimator(self) -> GridSearchCV:
        base_estimator = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        return GridSearchCV(
            estimator=base_estimator,
            param_grid={
                "C": [0.35, 0.75, 1.25, 2.0],
            },
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            refit=True,
        )
