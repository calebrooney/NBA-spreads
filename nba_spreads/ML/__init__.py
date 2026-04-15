"""Margin-model building blocks: game-level table, splits, training."""

from nba_spreads.ML.game_table import (
    DATE_ORDER_COLS,
    TARGET_COL,
    add_prior_team_features,
    build_game_level_features,
    extract_game_level_home,
    order_games_for_time_splits,
)

__all__ = [
    "DATE_ORDER_COLS",
    "TARGET_COL",
    "add_prior_team_features",
    "build_game_level_features",
    "extract_game_level_home",
    "order_games_for_time_splits",
]
