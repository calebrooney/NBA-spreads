"""
CLI entrypoint to train/evaluate the Phase-1 home-margin model.

This script prints walk-forward MAE/RMSE/bias for baselines and the primary regressor,
using *only* pre-holdout data from `nba.game_logs`.

Usage:
    python -m nba_spreads.ML.train_eval
"""

from __future__ import annotations

from dotenv import load_dotenv

from nba_spreads.ML.db import get_engine_from_env, load_game_logs
from nba_spreads.ML.pipeline import TrainEvalConfig, format_report, walk_forward_train_eval


def main() -> None:
    """
    Load data, run walk-forward evaluation, and print a metrics report.
    """
    load_dotenv()
    engine = get_engine_from_env()
    game_logs = load_game_logs(engine)

    cfg = TrainEvalConfig()
    results = walk_forward_train_eval(game_logs, cfg=cfg)
    print(format_report(results))


if __name__ == "__main__":
    main()

