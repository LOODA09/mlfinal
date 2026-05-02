from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .models import BaseHotelModel
from .training import ModelTrainer


@dataclass
class ValidationRunner:
    trainer: ModelTrainer

    def run(
        self,
        model_specs: Iterable[BaseHotelModel],
        x_data: pd.DataFrame,
        y_data: pd.Series,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        return self.trainer.k_fold_cross_validate(model_specs, x_data, y_data, n_splits=n_splits)
