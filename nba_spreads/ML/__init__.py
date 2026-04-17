"""
Machine-learning utilities for margin prediction from `nba.game_logs`.

This package provides the feature-engineering and time-splitting building blocks used
to train models that predict `home_margin` using only information available before tipoff
(i.e., derived from prior games; no odds/market data).
"""

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
