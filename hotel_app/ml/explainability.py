from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class SHAPAnalyzer:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def explain(
        self,
        model: Pipeline,
        x_background: pd.DataFrame,
        x_explain: pd.DataFrame,
        max_background: int = 100,
    ) -> Any:
        import shap
        from scipy import sparse

        background = x_background.sample(min(max_background, len(x_background)), random_state=self.random_state)
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        background_values = preprocessor.transform(background)
        explain_values = preprocessor.transform(x_explain)
        if sparse.issparse(background_values):
            background_values = background_values.toarray()
        if sparse.issparse(explain_values):
            explain_values = explain_values.toarray()
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{index}" for index in range(background_values.shape[1])]
        background_frame = pd.DataFrame(background_values, columns=feature_names)
        explain_frame = pd.DataFrame(explain_values, columns=feature_names)

        def predict_fn(values: np.ndarray) -> np.ndarray:
            return estimator.predict_proba(np.asarray(values, dtype=np.float32))[:, 1]

        explainer = shap.Explainer(predict_fn, background_frame)
        return explainer(explain_frame)

    def summary_plot(self, shap_values: Any, max_display: int = 15) -> Any:
        import matplotlib.pyplot as plt
        import shap

        shap.summary_plot(shap_values, show=False, max_display=max_display)
        figure = plt.gcf()
        plt.tight_layout()
        return figure
