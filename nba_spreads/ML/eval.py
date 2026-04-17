"""
Walk-forward evaluation utilities (time-safe) for the Phase-1 model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldResult:
    """
    Metrics for a single walk-forward fold.
    """

    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    mae: float
    rmse: float
    bias: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """
    Compute MAE, RMSE, and bias for regression predictions.

    Bias is defined as mean(pred - true), in points.

    :param y_true: Ground-truth values.
    :param y_pred: Predicted values.
    :return: (mae, rmse, bias)
    """
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    return mae, rmse, bias


def expanding_walk_forward_folds(
    dates: pd.Series, n_splits: int = 6, test_days: int = 30, min_train_days: int = 240
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate expanding walk-forward folds using calendar-day cutoffs.

    The training set grows each fold; each test set is the next fixed-size date window.

    :param dates: Normalized pandas datetime Series aligned to rows.
    :param n_splits: Number of folds to generate.
    :param test_days: Number of days in each test window.
    :param min_train_days: Minimum number of days required in the first training window.
    :return: List of (train_idx, test_idx) index arrays.
    """
    d = pd.to_datetime(dates).dt.normalize()
    unique_days = np.array(sorted(d.unique()))
    if len(unique_days) < (min_train_days + test_days + 1):
        raise ValueError(
            f"Not enough distinct dates ({len(unique_days)}) for walk-forward: "
            f"need at least ~{min_train_days + test_days + 1}."
        )

    # Evenly space fold cutoffs across the available timeline after the initial training window.
    first_test_start_i = min_train_days
    last_possible_test_start_i = len(unique_days) - test_days
    if last_possible_test_start_i <= first_test_start_i:
        raise ValueError("Timeline too short for requested test_days/min_train_days.")

    test_start_is = np.linspace(
        first_test_start_i, last_possible_test_start_i, num=n_splits, dtype=int
    )

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i, start_i in enumerate(test_start_is, start=1):
        test_start = unique_days[start_i]
        test_end = unique_days[min(start_i + test_days - 1, len(unique_days) - 1)]

        train_mask = d.lt(test_start)
        test_mask = d.ge(test_start) & d.le(test_end)

        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        folds.append((train_idx, test_idx))

    if not folds:
        raise ValueError("No folds generated; adjust parameters.")

    return folds


def summarize_fold_results(results: list[FoldResult]) -> pd.DataFrame:
    """
    Convert fold results into a tidy DataFrame with an overall average row.

    :param results: List of FoldResult.
    :return: DataFrame with per-fold metrics and an overall row.
    """
    rows = [r.__dict__ for r in results]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    overall = {
        "fold": "overall",
        "train_start": df["train_start"].min(),
        "train_end": df["train_end"].max(),
        "test_start": df["test_start"].min(),
        "test_end": df["test_end"].max(),
        "n_train": int(df["n_train"].sum()),
        "n_test": int(df["n_test"].sum()),
        "mae": float(df["mae"].mean()),
        "rmse": float(df["rmse"].mean()),
        "bias": float(df["bias"].mean()),
    }
    return pd.concat([df, pd.DataFrame([overall])], ignore_index=True)

