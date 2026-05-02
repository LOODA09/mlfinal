from __future__ import annotations

from dataclasses import dataclass

from hotel_cancellation_oop import EvaluationMetrics, ModelTester, ModelTrainer


@dataclass
class TrainingService:
    trainer: ModelTrainer

    def prepare_data(self, *args, **kwargs):
        return self.trainer.prepare_data(*args, **kwargs)

    def split_data(self, *args, **kwargs):
        return self.trainer.split_data(*args, **kwargs)

    def train_model(self, *args, **kwargs):
        return self.trainer.train_model(*args, **kwargs)

    def train_many(self, *args, **kwargs):
        return self.trainer.train_many(*args, **kwargs)


@dataclass
class TestingService:
    tester: ModelTester

    def test_model(self, *args, **kwargs):
        return self.tester.test_model(*args, **kwargs)

    def test_many(self, *args, **kwargs):
        return self.tester.test_many(*args, **kwargs)


@dataclass
class ValidationService:
    trainer: ModelTrainer

    def cross_validate(self, *args, **kwargs):
        return self.trainer.k_fold_cross_validate(*args, **kwargs)


class MetricsService:
    @staticmethod
    def evaluate(*args, **kwargs):
        return EvaluationMetrics.evaluate(*args, **kwargs)

    @staticmethod
    def report(*args, **kwargs):
        return EvaluationMetrics.report(*args, **kwargs)
