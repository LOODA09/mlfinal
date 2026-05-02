from __future__ import annotations

from typing import Any, Dict, Type
import importlib

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .deep import KerasTabularClassifier


class BaseHotelModel:
    name = "Base Model"

    def get_estimator(self) -> Any:
        raise NotImplementedError

    def build_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        return Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", self.get_estimator())])


class LogisticRegressionModel(BaseHotelModel):
    name = "Logistic Regression"

    def get_estimator(self) -> LogisticRegression:
        return LogisticRegression(max_iter=1000, random_state=42)


class KNNModel(BaseHotelModel):
    name = "KNN"

    def get_estimator(self) -> KNeighborsClassifier:
        return KNeighborsClassifier()


class DecisionTreeModel(BaseHotelModel):
    name = "Decision Tree"

    def get_estimator(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(random_state=42)


class NaiveBayesModel(BaseHotelModel):
    name = "Naive Bayes"

    def get_estimator(self) -> GaussianNB:
        return GaussianNB()


class SVMModel(BaseHotelModel):
    name = "SVM"

    def get_estimator(self) -> CalibratedClassifierCV:
        return CalibratedClassifierCV(estimator=LinearSVC(C=1.0, random_state=42, dual="auto"), cv=3)


class RandomForestModel(BaseHotelModel):
    name = "Random Forest"

    def get_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=450,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )


class AdaBoostModel(BaseHotelModel):
    name = "AdaBoost"

    def get_estimator(self) -> AdaBoostClassifier:
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        try:
            return AdaBoostClassifier(estimator=tree, random_state=42)
        except TypeError:
            return AdaBoostClassifier(base_estimator=tree, random_state=42)


class GradientBoostingModel(BaseHotelModel):
    name = "Gradient Boosting"

    def get_estimator(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(random_state=42)


class ExtraTreesModel(BaseHotelModel):
    name = "Extra Trees"

    def get_estimator(self) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )


class XGBoostModel(BaseHotelModel):
    name = "XGBoost"

    def get_estimator(self) -> Any:
        try:
            xgboost = importlib.import_module("xgboost")
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for XGBoostModel. Install it with `pip install xgboost` "
                "or remove XGBoost from the selected models."
            ) from exc
        return xgboost.XGBClassifier(
            booster="gbtree",
            learning_rate=0.05,
            max_depth=6,
            n_estimators=320,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )


class VotingEnsembleModel(BaseHotelModel):
    name = "Voting Ensemble"

    def get_estimator(self) -> VotingClassifier:
        estimators = [
            ("logistic", LogisticRegression(max_iter=1000, random_state=42)),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=350,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            ("xgboost", XGBoostModel().get_estimator()),
        ]
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


class StackingEnsembleModel(BaseHotelModel):
    name = "Stacking Ensemble"

    def get_estimator(self) -> StackingClassifier:
        estimators = [
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=250,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=250,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            ("xgboost", XGBoostModel().get_estimator()),
            ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            stack_method="predict_proba",
            n_jobs=-1,
            passthrough=False,
        )


class ANNModel(BaseHotelModel):
    name = "ANN"

    def __init__(self, epochs: int = 250, batch_size: int = 256) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

    def get_estimator(self) -> Any:
        try:
            importlib.import_module("tensorflow")
            return KerasTabularClassifier(model_type="ann", epochs=self.epochs, batch_size=self.batch_size)
        except ImportError:
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                learning_rate_init=0.001,
                batch_size=min(self.batch_size, 256),
                max_iter=self.epochs,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )


class RNNModel(BaseHotelModel):
    name = "RNN"

    def __init__(self, epochs: int = 20, batch_size: int = 256) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

    def get_estimator(self) -> KerasTabularClassifier:
        return KerasTabularClassifier(model_type="rnn", epochs=self.epochs, batch_size=self.batch_size)


MODEL_REGISTRY: Dict[str, Type[BaseHotelModel]] = {
    model.name: model
    for model in (
        LogisticRegressionModel,
        KNNModel,
        DecisionTreeModel,
        NaiveBayesModel,
        SVMModel,
        RandomForestModel,
        AdaBoostModel,
        GradientBoostingModel,
        ExtraTreesModel,
        XGBoostModel,
        VotingEnsembleModel,
        StackingEnsembleModel,
        ANNModel,
        RNNModel,
    )
}
