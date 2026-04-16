"""
End-to-end training + walk-forward evaluation for the Phase-1 home-margin model.

This module intentionally:
- Excludes the locked holdout period (2025 playoffs + 2025-26 regular season proxy).
- Uses time-safe, walk-forward validation (no random shuffles).
- Reports MAE/RMSE/bias for several baselines plus a tabular regressor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_spreads.ML.eval import (
    FoldResult,
    expanding_walk_forward_folds,
    regression_metrics,
    summarize_fold_results,
)
from nba_spreads.ML.features import FeatureConfig, build_game_level_features, select_model_matrix
from nba_spreads.ML.model import ModelKind, ModelSuiteConfig, make_regressor, predict_safely


@dataclass(frozen=True)
class TrainEvalConfig:
    """
    Configuration for walk-forward evaluation behavior.
    """

    n_splits: int = 6
    test_days: int = 30
    min_train_days: int = 240
    feature_config: FeatureConfig = FeatureConfig()
    model_suite_config: ModelSuiteConfig = ModelSuiteConfig()
    model_kinds: tuple[ModelKind, ...] = ("lightgbm", "ridge")


def _baseline_predict_zero(n: int) -> np.ndarray:
    """
    Baseline predictor: always predicts 0 points of home margin.

    :param n: Number of predictions.
    :return: Array of zeros.
    """
    return np.zeros(n, dtype=float)


def _baseline_predict_mean(y_train: np.ndarray, n: int) -> np.ndarray:
    """
    Baseline predictor: always predicts the training-set mean.

    :param y_train: Training targets.
    :param n: Number of predictions.
    :return: Array filled with the training mean.
    """
    mu = float(np.mean(y_train))
    return np.full(n, mu, dtype=float)


def _baseline_predict_diff_margin_roll10(X: pd.DataFrame) -> np.ndarray:
    """
    Baseline predictor: uses the simple rolling margin differential (home - away).

    This is a strong "handcrafted" baseline because it encodes recent point-differential
    strength without fitting a model.

    :param X: Feature matrix from `select_model_matrix`.
    :return: Predictions array.
    """
    col = "diff_home_margin_roll10"
    if col not in X.columns:
        raise ValueError(
            f"Required baseline column `{col}` not found. "
            "Ensure rolling windows include 10 and `home_margin` exists."
        )
    return X[col].to_numpy(dtype=float)


def walk_forward_train_eval(
    game_logs: pd.DataFrame, cfg: TrainEvalConfig | None = None
) -> dict[str, pd.DataFrame]:
    """
    Run walk-forward evaluation on pre-holdout games.

    :param game_logs: Raw `nba.game_logs` dataframe.
    :param cfg: TrainEvalConfig.
    :return: Dict mapping model/baseline name -> results dataframe.
    """
    c = cfg or TrainEvalConfig()

    Xy, holdout = build_game_level_features(game_logs, config=c.feature_config)
    if not holdout.empty:
        # Explicitly not used, but useful to show that exclusion is working.
        pass

    X, y = select_model_matrix(Xy)
    dates = Xy["date"]

    folds = expanding_walk_forward_folds(
        dates, n_splits=c.n_splits, test_days=c.test_days, min_train_days=c.min_train_days
    )

    def _fold_window(idx: np.ndarray) -> tuple[str, str]:
        d = dates.iloc[idx]
        return str(d.min().date()), str(d.max().date())

    outputs: dict[str, list[FoldResult]] = {
        "baseline_zero": [],
        "baseline_mean": [],
        "baseline_diff_margin_roll10": [],
    }
    for kind in c.model_kinds:
        outputs[f"model_{kind}"] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds, start=1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        train_start, train_end = _fold_window(train_idx)
        test_start, test_end = _fold_window(test_idx)

        # Baseline: zero
        pred = _baseline_predict_zero(len(test_idx))
        mae, rmse, bias = regression_metrics(y_test, pred)
        outputs["baseline_zero"].append(
            FoldResult(
                fold=fold_i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train=len(train_idx),
                n_test=len(test_idx),
                mae=mae,
                rmse=rmse,
                bias=bias,
            )
        )

        # Baseline: train mean
        pred = _baseline_predict_mean(y_train, len(test_idx))
        mae, rmse, bias = regression_metrics(y_test, pred)
        outputs["baseline_mean"].append(
            FoldResult(
                fold=fold_i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train=len(train_idx),
                n_test=len(test_idx),
                mae=mae,
                rmse=rmse,
                bias=bias,
            )
        )

        # Baseline: diff margin roll10
        pred = _baseline_predict_diff_margin_roll10(X_test)
        mae, rmse, bias = regression_metrics(y_test, pred)
        outputs["baseline_diff_margin_roll10"].append(
            FoldResult(
                fold=fold_i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train=len(train_idx),
                n_test=len(test_idx),
                mae=mae,
                rmse=rmse,
                bias=bias,
            )
        )

        # Candidate models: fit/predict each family under the same fold split.
        for kind in c.model_kinds:
            model = make_regressor(kind, cfg=c.model_suite_config)
            model.fit(X_train, y_train)
            pred = predict_safely(model, X_test)
            mae, rmse, bias = regression_metrics(y_test, pred)
            outputs[f"model_{kind}"].append(
                FoldResult(
                    fold=fold_i,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    mae=mae,
                    rmse=rmse,
                    bias=bias,
                )
            )

    return {k: summarize_fold_results(v) for k, v in outputs.items()}


def format_report(results: dict[str, pd.DataFrame]) -> str:
    """
    Format result tables into a readable text report.

    :param results: Output of `walk_forward_train_eval`.
    :return: Multi-line string report.
    """
    lines: list[str] = []
    for name, df in results.items():
        lines.append(f"\n=== {name} ===")
        if df.empty:
            lines.append("(no results)")
            continue
        show = df[["fold", "n_train", "n_test", "train_end", "test_start", "mae", "rmse", "bias"]].copy()
        lines.append(show.to_string(index=False))
    return "\n".join(lines).lstrip()

