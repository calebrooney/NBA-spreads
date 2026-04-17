"""
Model factories for the Phase-1 home-margin prediction task.

This module is intentionally structured to support *testing multiple model families*
under the same walk-forward evaluation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class HistGBConfig:
    """
    Hyperparameters for sklearn's histogram gradient boosting regressor.

    This is a strong, low-maintenance baseline that stays within scikit-learn.
    """

    learning_rate: float = 0.05
    max_depth: int | None = 6
    max_iter: int = 600
    min_samples_leaf: int = 30
    l2_regularization: float = 1e-3
    random_state: int = 7


@dataclass(frozen=True)
class LightGBMConfig:
    """
    Hyperparameters for LightGBM regression on tabular features.

    Defaults are conservative (regularized) to reduce overfitting and to encourage
    better-behaved edge magnitudes for downstream ATS decisioning.
    """

    # Conservative defaults: favor stability + low bias over maximum fit.
    n_estimators: int = 1600
    learning_rate: float = 0.02
    max_depth: int = 4
    num_leaves: int = 31
    min_child_samples: int = 120
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    reg_alpha: float = 0.0
    reg_lambda: float = 10.0
    random_state: int = 7


@dataclass(frozen=True)
class RidgeConfig:
    """
    Hyperparameters for a ridge regression baseline.

    Ridge is often well-calibrated and stable, making it a valuable comparator even
    when tree boosting wins on MAE.
    """

    alpha: float = 5.0
    random_state: int = 7


@dataclass(frozen=True)
class XGBoostConfig:
    """
    Hyperparameters for XGBoost regression on tabular features.

    Defaults are conservative; XGBoost is powerful but can become "spiky" without
    regularization.
    """

    n_estimators: int = 1200
    learning_rate: float = 0.02
    max_depth: int = 4
    min_child_weight: float = 40.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 10.0
    random_state: int = 7


ModelKind = Literal["lightgbm", "ridge", "xgboost", "hist_gb"]


@dataclass(frozen=True)
class ModelSuiteConfig:
    """
    Collection of per-model hyperparameters for a multi-model test suite.

    The walk-forward evaluator can run any subset of models by name; this object keeps
    their configurations together.
    """

    hist_gb: HistGBConfig = HistGBConfig()
    lightgbm: LightGBMConfig = LightGBMConfig()
    ridge: RidgeConfig = RidgeConfig()
    xgboost: XGBoostConfig = XGBoostConfig()


def make_regressor(
    kind: ModelKind,
    cfg: ModelSuiteConfig | None = None,
) -> RegressorMixin:
    """
    Create a configured regressor by family name.

    :param kind: Model family key.
    :param cfg: ModelSuiteConfig.
    :return: A configured regressor implementing sklearn's estimator API.
    """
    c = cfg or ModelSuiteConfig()

    if kind == "hist_gb":
        p = c.hist_gb
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=p.learning_rate,
            max_depth=p.max_depth,
            max_iter=p.max_iter,
            min_samples_leaf=p.min_samples_leaf,
            l2_regularization=p.l2_regularization,
            random_state=p.random_state,
        )

    if kind == "lightgbm":
        p = c.lightgbm
        try:
            from lightgbm import LGBMRegressor
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "LightGBM import failed. On macOS this commonly means the OpenMP runtime "
                "`libomp` is missing. If you use Homebrew, run `brew install libomp`, "
                "then reinstall lightgbm in your virtualenv."
            ) from e
        return LGBMRegressor(
            objective="regression",
            n_estimators=p.n_estimators,
            learning_rate=p.learning_rate,
            max_depth=p.max_depth,
            num_leaves=p.num_leaves,
            min_child_samples=p.min_child_samples,
            subsample=p.subsample,
            colsample_bytree=p.colsample_bytree,
            reg_alpha=p.reg_alpha,
            reg_lambda=p.reg_lambda,
            random_state=p.random_state,
            n_jobs=-1,
        )

    if kind == "ridge":
        p = c.ridge
        # Ridge doesn't need random_state, but we keep it in the config so the suite
        # config has a consistent "seed" field across models if you later expand it.
        _ = p.random_state
        return Ridge(alpha=p.alpha)

    if kind == "xgboost":
        p = c.xgboost
        try:
            from xgboost import XGBRegressor
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "XGBoost import failed. On macOS this commonly means the OpenMP runtime "
                "`libomp` is missing. If you use Homebrew, run `brew install libomp`, "
                "then reinstall xgboost in your virtualenv."
            ) from e
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=p.n_estimators,
            learning_rate=p.learning_rate,
            max_depth=p.max_depth,
            min_child_weight=p.min_child_weight,
            subsample=p.subsample,
            colsample_bytree=p.colsample_bytree,
            reg_alpha=p.reg_alpha,
            reg_lambda=p.reg_lambda,
            random_state=p.random_state,
            n_jobs=-1,
            tree_method="hist",
        )

    # Defensive: `kind` is a Literal, but keep runtime safety for CLI/script usage.
    raise ValueError(f"Unknown model kind: {kind!r}")


def predict_safely(model: RegressorMixin, X: object) -> np.ndarray:
    """
    Predict and coerce output to a 1D float numpy array.

    Some estimators return shapes like (n, 1); this normalizes to (n,).

    :param model: Fitted regressor.
    :param X: Feature matrix compatible with the estimator.
    :return: 1D numpy float predictions.
    """
    pred = model.predict(X)
    return np.asarray(pred, dtype=float).reshape(-1)
