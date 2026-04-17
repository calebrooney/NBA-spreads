"""
Feature engineering for the Phase-1 home-margin model.

Key constraints enforced here:
- One row per game (home-team row).
- All features are computed from team logs *strictly before* game time via shift+rolling.
- No odds/market inputs are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration for rolling feature horizons and holdout exclusions.

    The holdout rules are intentionally conservative: we exclude the entire period starting
    at the beginning of the 2025 playoffs, through the 2025-26 regular season window.
    This guarantees the locked holdout is never used for training/tuning.
    """

    rolling_windows: tuple[int, ...] = (5, 10, 20)
    holdout_start: str = "2025-04-14"


def _coerce_game_log_types(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dtypes and required columns from `nba.game_logs`.

    :param game_logs: Raw `nba.game_logs` dataframe.
    :return: Copy with normalized `date`, `home`, and numeric columns where possible.
    """
    df = game_logs.copy()

    if "date" not in df.columns:
        raise ValueError("`nba.game_logs` is missing required column `date`.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    if "home" not in df.columns:
        raise ValueError("`nba.game_logs` is missing required column `home`.")
    df["home"] = df["home"].astype(bool)

    for col in ["home_margin", "tm_score", "opp_score", "ortg", "drtg", "pace"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["team", "opp", "game_id"]:
        if col not in df.columns:
            raise ValueError(f"`nba.game_logs` is missing required column `{col}`.")

    return df


def _add_schedule_features_prior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add schedule/rest features that are known before tip-off.

    These features are computed per-team using only prior game dates:
    - rest_days: calendar days since previous game (NaN for first game)
    - b2b: 1 if previous game was yesterday (rest_days == 1)
    - games_last4: count of games in the prior 4 days (excluding current game)
    - games_last6: count of games in the prior 6 days (excluding current game)

    :param df: Team-game log rows (one row per team per game).
    :return: Copy with schedule feature columns appended.
    """
    out = df.copy()
    out = out.sort_values(["team", "date", "game_id"], ascending=True)

    prev_date = out.groupby("team", sort=False)["date"].shift(1)
    rest_days = (out["date"] - prev_date).dt.days.astype("float")
    out["rest_days"] = rest_days
    out["b2b"] = (rest_days == 1).astype("float")

    # Count prior games within rolling day windows using shifted previous dates.
    # We use the last 5 games as candidates; this is cheap and typically enough.
    prior_dates = [out.groupby("team", sort=False)["date"].shift(k) for k in range(1, 6)]
    for window_days, name in [(4, "games_last4"), (6, "games_last6")]:
        cnt = 0.0
        for d in prior_dates:
            cnt += ((out["date"] - d).dt.days <= (window_days - 1)).astype("float")
        # If prior date is NaT, comparison yields False -> contributes 0.
        out[name] = cnt

    return out


def _rolling_mean_prior(
    df: pd.DataFrame, group_col: str, time_col: str, value_cols: Iterable[str], window: int
) -> pd.DataFrame:
    """
    Compute prior-only rolling means for each group.

    Implementation details:
    - Sort by (group, time) to ensure stable rolling windows.
    - Shift by 1 so the current game's stats are never included in its own features.

    :param df: Input dataframe.
    :param group_col: Grouping column (e.g. team).
    :param time_col: Time ordering column (e.g. date).
    :param value_cols: Numeric columns to roll.
    :param window: Rolling window size in games.
    :return: Dataframe of same length with new feature columns appended.
    """
    out = df.copy()
    out = out.sort_values([group_col, time_col, "game_id"], ascending=True)

    for col in value_cols:
        feat = (
            out.groupby(group_col, sort=False)[col]
            .shift(1)
            .rolling(window=window, min_periods=window)
            .mean()
        )
        out[f"{col}_roll{window}"] = feat

    return out


def build_game_level_features(
    game_logs: pd.DataFrame, config: FeatureConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a game-level modeling table with prior-only rolling features.

    Output includes:
    - `Xy` (pre-holdout): features + `home_margin` for walk-forward CV.
    - `holdout` (locked): rows at/after the holdout start date, excluded entirely.

    :param game_logs: Raw `nba.game_logs` dataframe (all rows).
    :param config: FeatureConfig.
    :return: (Xy_pre_holdout, holdout_locked)
    """
    cfg = config or FeatureConfig()
    df = _coerce_game_log_types(game_logs)

    # Keep only rows with a valid game_id/date/team; null dates break time ordering.
    df = df.dropna(subset=["game_id", "date", "team", "opp"])

    # Split out locked holdout by date (conservative).
    holdout_start = pd.Timestamp(cfg.holdout_start)
    holdout_mask = df["date"].ge(holdout_start)
    df_model = df.loc[~holdout_mask].copy()
    df_holdout = df.loc[holdout_mask].copy()

    # Rolling features are computed per team across all games (home+away rows),
    # but always shifted so the current game never leaks into its features.
    numeric_candidates = [
        "home_margin",
        "ortg",
        "drtg",
        "pace",
        "ftr",
        "_3par",
        "TS%",
        "TRB%",
        "AST%",
        "STL%",
        "BLK%",
        "eFG%",
        "TOV%",
        "ORB%",
        "FT/FGA",
        "Opp_eFG%",
        "Opp_TOV%",
        "Opp_ORB%",
        "Opp_FT/FGA",
    ]
    value_cols = [c for c in numeric_candidates if c in df_model.columns]

    rolled = _add_schedule_features_prior(df_model)
    for w in cfg.rolling_windows:
        rolled = _rolling_mean_prior(
            rolled, group_col="team", time_col="date", value_cols=value_cols, window=w
        )

    # IMPORTANT: prevent target leakage from postgame columns.
    #
    # `nba.game_logs` contains per-game outcomes (e.g. tm_score/opp_score) that are only
    # known after the game finishes. We keep *only* prior-only rolling features (the
    # `_roll{w}` columns) and identifiers needed for the game-level join.
    id_cols = ["game_id", "date", "team", "opp", "home", "home_margin"]
    roll_cols = [
        c for c in rolled.columns if any(c.endswith(f"_roll{w}") for w in cfg.rolling_windows)
    ]
    schedule_cols = [c for c in ["rest_days", "b2b", "games_last4", "games_last6"] if c in rolled.columns]
    keep_cols = [c for c in id_cols if c in rolled.columns] + schedule_cols + roll_cols
    rolled = rolled[keep_cols].copy()

    # Build one row per game: start from the home row.
    home_rows = rolled.loc[rolled["home"].eq(True)].copy()
    home_rows = home_rows.drop_duplicates(subset=["game_id"], keep="first")

    # Attach the away team's rolling features by joining on the away row for the same game_id.
    away_rows = rolled.loc[rolled["home"].eq(False)].copy()
    away_rows = away_rows.sort_values(["game_id", "team"]).drop_duplicates(
        subset=["game_id"], keep="first"
    )

    # Suffix columns so model sees home/away separately.
    base_cols = {"game_id", "date", "team", "opp", "home", "home_margin"}
    home_feat_cols = [c for c in home_rows.columns if c not in base_cols]
    away_feat_cols = [c for c in away_rows.columns if c not in base_cols]

    home_feat = home_rows[["game_id", "date", "team", "opp", "home_margin"] + home_feat_cols].copy()
    away_feat = away_rows[["game_id"] + away_feat_cols].copy()

    home_feat = home_feat.rename(columns={c: f"home_{c}" for c in home_feat_cols})
    away_feat = away_feat.rename(columns={c: f"away_{c}" for c in away_feat_cols})

    Xy = home_feat.merge(away_feat, on="game_id", how="inner", validate="one_to_one")

    # Add a few simple difference features (home minus away) for core signals.
    for w in cfg.rolling_windows:
        for base in ["home_margin", "ortg", "drtg", "pace"]:
            h = f"home_{base}_roll{w}"
            a = f"away_{base}_roll{w}"
            if h in Xy.columns and a in Xy.columns:
                Xy[f"diff_{base}_roll{w}"] = Xy[h] - Xy[a]

    for base in ["rest_days", "b2b", "games_last4", "games_last6"]:
        h = f"home_{base}"
        a = f"away_{base}"
        if h in Xy.columns and a in Xy.columns:
            Xy[f"diff_{base}"] = Xy[h] - Xy[a]

    # Normalize key types and enforce chronological ordering. This keeps walk-forward
    # diagnostics and index-based folds easier to reason about.
    Xy["date"] = pd.to_datetime(Xy["date"], errors="coerce").dt.normalize()
    Xy["home_margin"] = pd.to_numeric(Xy["home_margin"], errors="coerce").astype(float)
    Xy = Xy.sort_values(["date", "game_id"], ascending=True)

    # Remove rows without full rolling history (burn-in period).
    feature_cols = [c for c in Xy.columns if c not in {"game_id", "date", "team", "opp", "home_margin"}]
    Xy = Xy.dropna(subset=feature_cols + ["home_margin"]).reset_index(drop=True)

    return Xy, df_holdout


def select_model_matrix(Xy: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Split the modeling table into (X, y) suitable for sklearn estimators.

    :param Xy: Output from `build_game_level_features` (pre-holdout).
    :return: (X dataframe, y numpy array)
    """
    y = Xy["home_margin"].to_numpy(dtype=float)
    drop_cols = {"game_id", "date", "team", "opp", "home_margin"}
    X = Xy.drop(columns=[c for c in drop_cols if c in Xy.columns]).copy()
    return X, y
