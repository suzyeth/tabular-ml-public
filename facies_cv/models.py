"""Model factories. Hyper-parameters are fixed up front, not tuned (see README)."""
from __future__ import annotations

from typing import Callable

from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE

ModelFactory = Callable[[], ClassifierMixin]


def majority() -> ClassifierMixin:
    return DummyClassifier(strategy="most_frequent")


def logistic() -> ClassifierMixin:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def lightgbm(n_estimators: int = 300) -> ClassifierMixin:
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
    )


def model_factories(n_estimators: int = 300) -> dict[str, ModelFactory]:
    return {
        "majority": majority,
        "logistic": logistic,
        "lightgbm": lambda: lightgbm(n_estimators),
    }
