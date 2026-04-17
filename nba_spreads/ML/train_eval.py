"""
CLI entrypoint to train/evaluate the Phase-1 home-margin model.

This script prints walk-forward MAE/RMSE/bias for baselines and a configurable suite of
regressors, using *only* pre-holdout data from `nba.game_logs`.

It also writes the same results to `model_results/` as JSON for easy diffing across runs.

Usage:
    python -m nba_spreads.ML.train_eval
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from nba_spreads.ML.db import get_engine_from_env, load_game_logs, load_odds
from nba_spreads.ML.pipeline import TrainEvalConfig, format_report, walk_forward_train_eval
from nba_spreads.ML.results_io import next_iteration_results_path, results_payload, save_results_json


def _market_spread_benchmark(odds_df: pd.DataFrame, game_logs_df: pd.DataFrame) -> dict[str, float] | None:
    """
    Compute a simple market benchmark: latest home-team spread vs actual home margin.

    This mirrors the logic in `notebooks/benchmark.ipynb`:
    - Take one home row per game from `nba.game_logs`.
    - Take the latest odds snapshot per (game_id, bookmaker_key, team).
    - Join on (game_id, team) so the spread is expressed on the home team.
    - Residual = home_margin + spread (spread is negative for favorites).

    :param odds_df: Raw `nba.odds` dataframe.
    :param game_logs_df: Raw `nba.game_logs` dataframe.
    :return: Dict with bias/mae/rmse, or None if required columns are missing.
    """
    required_odds = {"game_id", "bookmaker_key", "team"}
    required_logs = {"game_id", "team", "home", "home_margin"}
    if not required_odds.issubset(set(odds_df.columns)):
        return None
    if not required_logs.issubset(set(game_logs_df.columns)):
        return None

    ts_col = "timestamp" if "timestamp" in odds_df.columns else "snapshot_time_utc"
    if ts_col not in odds_df.columns:
        return None
    if "point" not in odds_df.columns:
        return None

    home_games = game_logs_df.loc[game_logs_df["home"].eq(True)].copy()
    home_games = home_games.drop_duplicates(subset=["game_id"], keep="first")
    if home_games.empty:
        return None

    odds_sorted = odds_df.sort_values(ts_col, ascending=False)
    odds_latest = odds_sorted.drop_duplicates(subset=["game_id", "bookmaker_key", "team"], keep="first")

    joined = home_games.merge(
        odds_latest,
        on=["game_id", "team"],
        how="inner",
        validate="one_to_many",
    )
    if joined.empty:
        return None

    home_margin = pd.to_numeric(joined["home_margin"], errors="coerce").astype(float)
    spread = pd.to_numeric(joined["point"], errors="coerce").astype(float)
    residual = (home_margin + spread).to_numpy(dtype=float)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return None

    bias = float(np.mean(residual))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {"bias": bias, "mae": mae, "rmse": rmse, "n": float(residual.size)}


def main() -> None:
    """
    Load data, run walk-forward evaluation, print a metrics report, and persist JSON.
    """
    load_dotenv()
    engine = get_engine_from_env()
    game_logs = load_game_logs(engine)
    odds_df = load_odds(engine)

    cfg = TrainEvalConfig()
    out = walk_forward_train_eval(game_logs, cfg=cfg)
    print(format_report(out.results))
    print(f"\nXy['date'].is_monotonic_increasing = {out.diagnostics.dates_monotonic_increasing}")

    market_benchmark = _market_spread_benchmark(odds_df, game_logs)
    if market_benchmark is not None:
        print(
            "\n=== market_latest_spread_benchmark ===\n"
            f"n={int(market_benchmark['n'])} bias={market_benchmark['bias']:.4f} "
            f"mae={market_benchmark['mae']:.4f} rmse={market_benchmark['rmse']:.4f}"
        )

    out_dir = Path("model_results")
    out_path = next_iteration_results_path(out_dir, stem="walk_forward")
    payload = results_payload(
        results=out.results,
        train_eval_cfg=cfg,
        extra={"diagnostics": out.diagnostics, "market_latest_spread_benchmark": market_benchmark},
    )
    written = save_results_json(payload, out_path)
    print(f"\nWrote JSON results to: {written}")


if __name__ == "__main__":
    main()

