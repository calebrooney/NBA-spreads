"""
Game-level table from ``nba.game_logs``: home rows only, target ``home_margin``.

Use the same dedupe rule as ``notebooks/benchmark.ipynb``. Chronological ordering
(``date``, then ``game_id``) is the canonical order for date-ordered and walk-forward
validation — never shuffle across time.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import pandas as pd

# Target column on the home row (home score − away score).
TARGET_COL: str = "home_margin"

# Columns to sort by ascending for time-safe splits (stable tie-break on game_id).
DATE_ORDER_COLS: tuple[str, ...] = ("date", "game_id")

# Columns that are directly outcomes of the current game and must never be used
# unshifted (i.e., not as "current game" features). Using their *prior* rolling
# values is allowed and often predictive (recent scoring form, etc.).
_OUTCOME_COLS: frozenset[str] = frozenset({"home_margin"})


def _sanitize_feature_name(col: str) -> str:
    """
    Convert a source column name into a stable, model-friendly feature suffix.

    This is intentionally conservative so columns like ``TS%`` or ``FT/FGA`` map
    to safe identifiers (e.g. ``ts``, ``ft_fga``) without punctuation.

    :param col: Raw column name from the logs table.
    :return: Lowercased, punctuation-free feature name.
    """
    out = col.strip().lower()
    out = re.sub(r"[^a-z0-9]+", "_", out)
    out = re.sub(r"(^_+|_+$)", "", out)
    return out


def _infer_stat_columns(game_logs: pd.DataFrame) -> list[str]:
    """
    Infer numeric/stat columns suitable for rolling/expanding aggregation.

    Excludes ID/keys and the target label. Keeps score columns because their
    *prior* rolling values are time-safe (they come from past games only).

    :param game_logs: Raw or subset logs table.
    :return: List of column names to aggregate.
    """
    never = {
        "game_id",
        "date",
        "team",
        "opp",
        "home",
        "ot",
        "game",
        *_OUTCOME_COLS,
    }
    numeric = (
        game_logs.select_dtypes(include=["number"]).columns.difference(list(never)).tolist()
    )
    return numeric


def add_prior_team_features(
    game_logs: pd.DataFrame,
    *,
    rolling_windows: Sequence[int] = (5, 10, 20),
    min_periods: int = 3,
    stat_cols: Sequence[str] | None = None,
    include_expanding: bool = True,
) -> pd.DataFrame:
    """
    Add prior-only team rolling/expanding features to *each* team-game row.

    All aggregated features are computed from games strictly before the current
    row's game date (and `game_id` tie-break ordering). This is enforced by
    shifting within each team before applying rolling/expanding windows.

    Also adds rest-day signals computed from the team's prior game date:
    ``rest_days`` and ``back_to_back``.

    :param game_logs: Frame with at least ``date``, ``game_id``, and ``team``.
    :param rolling_windows: Rolling window sizes in number of prior games.
    :param min_periods: Minimum prior games required for a rolling statistic.
    :param stat_cols: Columns to aggregate; if None, infer numeric/stat columns.
    :param include_expanding: If True, add expanding (cumulative) means as well.
    :return: Copy of ``game_logs`` with added feature columns.
    """
    required = {"date", "game_id", "team"}
    missing = required - set(game_logs.columns)
    if missing:
        raise ValueError(f"game_logs missing required columns: {sorted(missing)}")

    df = game_logs.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(list(DATE_ORDER_COLS), kind="mergesort").reset_index(drop=True)

    if stat_cols is None:
        stat_cols = _infer_stat_columns(df)
    stat_cols = [c for c in stat_cols if c in df.columns and c not in _OUTCOME_COLS]

    # Rest features: rest days *before* this game (prior game date → current date).
    prev_date = df.groupby("team", sort=False)["date"].shift(1)
    df["rest_days"] = (df["date"] - prev_date).dt.days
    df["back_to_back"] = df["rest_days"].eq(1)
    df["games_played_before"] = df.groupby("team", sort=False).cumcount()

    # Aggregations: shift within team so current game's stats are never used.
    g = df.groupby("team", sort=False)
    for col in stat_cols:
        safe = _sanitize_feature_name(col)
        shifted = g[col].shift(1)

        for w in rolling_windows:
            feat = f"{safe}_roll{int(w)}_mean"
            df[feat] = shifted.groupby(df["team"], sort=False).rolling(
                window=int(w), min_periods=min_periods
            ).mean().reset_index(level=0, drop=True)

        if include_expanding:
            df[f"{safe}_exp_mean"] = shifted.groupby(df["team"], sort=False).expanding(
                min_periods=min_periods
            ).mean().reset_index(level=0, drop=True)

    return df


def build_game_level_features(
    game_logs: pd.DataFrame,
    *,
    rolling_windows: Sequence[int] = (5, 10, 20),
    min_periods: int = 3,
    stat_cols: Sequence[str] | None = None,
    include_expanding: bool = True,
) -> pd.DataFrame:
    """
    Build a one-row-per-game modeling table with prior-only home/away features.

    Output is keyed on the **home row** (one row per ``game_id``). Home-team
    features are computed from the home team's prior games; away-team features
    are computed from the away team's prior games and merged onto the home row.

    Leakage control:
    - All rolling/expanding features are based on *shifted* values within team.
    - Rest days are based on the team's *previous* game date.

    :param game_logs: Raw or subset frame from ``nba.game_logs``.
    :param rolling_windows: Rolling window sizes in number of prior games.
    :param min_periods: Minimum prior games required for rolling/expanding stats.
    :param stat_cols: Columns to aggregate; if None, infer numeric/stat columns.
    :param include_expanding: If True, add expanding (cumulative) means as well.
    :return: Game-level frame (home row only) with home_*/away_* features added.
    """
    required = {"home", "game_id", "date", "team", "opp", TARGET_COL}
    missing = required - set(game_logs.columns)
    if missing:
        raise ValueError(f"game_logs missing required columns: {sorted(missing)}")

    with_feats = add_prior_team_features(
        game_logs,
        rolling_windows=rolling_windows,
        min_periods=min_periods,
        stat_cols=stat_cols,
        include_expanding=include_expanding,
    )

    # Split into home and away rows (still one row per team-game).
    home_rows = with_feats.loc[with_feats["home"].eq(True)].copy()
    away_rows = with_feats.loc[with_feats["home"].eq(False)].copy()

    if home_rows.empty or away_rows.empty:
        raise ValueError("Expected both home=True and home=False rows per game_id.")

    # Dedupe to a single row per game, mirroring the benchmark notebook convention.
    home_rows = home_rows.drop_duplicates(subset=["game_id"], keep="first")
    away_rows = away_rows.drop_duplicates(subset=["game_id"], keep="first")

    # Identify feature columns added by `add_prior_team_features`.
    base_cols = set(game_logs.columns)
    added_cols = [c for c in with_feats.columns if c not in base_cols]

    # Merge away-team prior features onto the home row using game_id.
    away_side = away_rows[["game_id", *added_cols]].copy()
    away_side = away_side.rename(columns={c: f"away_{c}" for c in added_cols})
    out = home_rows.merge(away_side, on="game_id", how="left", validate="one_to_one")

    # Prefix home-side features for symmetry; keep target and identifiers unprefixed.
    rename_home = {c: f"home_{c}" for c in added_cols if c in out.columns}
    out = out.rename(columns=rename_home)

    return order_games_for_time_splits(out)


def extract_game_level_home(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per ``game_id`` using only home-team rows; label is ``home_margin``.

    Drops the away-team row for each game so downstream rolling features do not
    double-count games. Duplicate ``game_id`` rows, if any, keep the first occurrence
    (same convention as the benchmark notebook).

    :param game_logs: Raw or subset frame from ``nba.game_logs`` (must include
        ``home``, ``game_id``, ``home_margin``, and columns in ``DATE_ORDER_COLS``).
    :return: Game-level frame, unsorted; use ``order_games_for_time_splits`` before splits.
    """
    required = {"home", "game_id", "home_margin", *DATE_ORDER_COLS}
    missing = required - set(game_logs.columns)
    if missing:
        raise ValueError(f"game_logs missing required columns: {sorted(missing)}")

    home_only = game_logs.loc[game_logs["home"].eq(True)].copy()
    if home_only.empty:
        raise ValueError("No rows with home=True; cannot build game-level table.")

    out = home_only.drop_duplicates(subset=["game_id"], keep="first")
    return out


def order_games_for_time_splits(game_level: pd.DataFrame) -> pd.DataFrame:
    """
    Sort games chronologically for time-based train/test or walk-forward CV.

    Primary key for ordering is game calendar ``date``; ``game_id`` breaks ties
    so the order is stable when multiple games share a date.

    :param game_level: Output of ``extract_game_level_home`` (one row per game).
    :return: Copy sorted by ``DATE_ORDER_COLS`` ascending.
    """
    missing = set(DATE_ORDER_COLS) - set(game_level.columns)
    if missing:
        raise ValueError(f"game_level missing columns for ordering: {sorted(missing)}")
    return game_level.sort_values(list(DATE_ORDER_COLS), kind="mergesort").reset_index(
        drop=True
    )
