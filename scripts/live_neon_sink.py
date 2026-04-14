"""
Daily or cron-driven Neon ingestion (planned).

Bulk historical loads: run ``scripts/neon_training_sink.py`` once against ``data/raw`` and
``data/raw/historical`` NBAodds CSVs.

This entrypoint is reserved for a lighter, recurring job: append today's live odds snapshot
(see ``nba_spreads.fetch.fetch_csv`` / ``scripts/fetch_odds_daily.py``) and refresh
current-season Basketball-Reference game logs, then ``insert_odds`` / ``insert_game_logs``
in FK order. Implement after the one-time sink is validated.

game logs should only drop all games that have already been ingested into the database from df in cleaning stage.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# Repo root on sys.path for ``nba_spreads``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nba_spreads.fetch import fetch_csv
from nba_spreads.game_data import game_logs

# Load DB + prep helpers from ``scripts/neon_training_sink.py`` (keeps logic centralized).
_sink_spec = importlib.util.spec_from_file_location(
    "neon_training_sink",
    ROOT / "scripts" / "neon_training_sink.py",
)
_neon = importlib.util.module_from_spec(_sink_spec)
assert _sink_spec.loader is not None
_sink_spec.loader.exec_module(_neon)


def _today_pacific() -> date:
    """
    Return today's calendar date in Pacific time.

    GitHub-hosted runners run in UTC; we pin "today" to Pacific to match your snapshot naming
    and the Odds API commenceTimeFrom/To window used in ``nba_spreads.fetch.fetch_csv``.
    """
    pacific = ZoneInfo("America/Los_Angeles")
    return datetime.now(pacific).date()


def _scrape_recent_game_logs(days_back: int) -> pd.DataFrame:
    """
    Scrape BRef advanced team game logs for a small recent date window.

    This keeps daily runs bounded: 30 teams × ~1 season × 1 table ≈ 30 requests.

    :param days_back: Number of days back from today (Pacific) to include, inclusive.
    :return: Prepared dataframe ready for ``insert_game_logs``.
    """
    if days_back < 0:
        raise ValueError("days_back must be >= 0")

    end = _today_pacific()
    start = end - timedelta(days=days_back)

    years = _neon.season_years_in_range(start, end)
    frames: list[pd.DataFrame] = []
    for team in game_logs.teams:
        try:
            df = game_logs.fetch_team_logs_for_date_window(
                team=team,
                season_years=years,
                min_date=start,
                max_date=end,
            )
        except Exception as exc:  # noqa: BLE001 — scraping variance
            print(f"skip BRef {team} {years} {start}..{end}: {exc}")
            continue
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Game_ID", "Team"], keep="first")
    return _neon.game_logs_db_prep(combined, "NBA")


def _load_today_raw_odds_snapshot() -> pd.DataFrame:
    """
    Fetch today's live odds snapshot using the free Odds API key, then load it into a dataframe.

    Implementation note:
    - We reuse ``fetch_csv`` for URL construction and timestamp columns.
    - The CSV is written to the runner filesystem and is not committed to the repo.

    :return: Raw snapshot dataframe (shape produced by ``fetch_csv`` before cleaning).
    """
    today = _today_pacific().isoformat()
    path = fetch_csv(today, live=True)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


def run_live_neon_update(days_back_game_logs: int = 2) -> None:
    """
    Daily ingestion:
    - Refresh recent game logs (BRef) for FK coverage.
    - Fetch today's odds snapshot (Odds API free).
    - Clean → prep → FK-filter → insert into ``nba.odds``.

    :param days_back_game_logs: How many days back of game logs to refresh from BRef.
    """
    # Fail fast if required env vars are missing (common in CI misconfiguration).
    if not os.getenv("DATABASE_URL"):
        raise ValueError("DATABASE_URL is required (set as a GitHub Actions secret).")
    if not os.getenv("ODDS_API_KEY_FREE"):
        raise ValueError("ODDS_API_KEY_FREE is required (set as a GitHub Actions secret).")

    engine = _neon.get_db_engine()

    print(f"Refreshing game logs (last {days_back_game_logs} days)…")
    recent_logs = _scrape_recent_game_logs(days_back_game_logs)
    n_logs = _neon.insert_game_logs(recent_logs, engine=engine)
    print(f"Inserted {n_logs} game_logs rows (ON CONFLICT DO NOTHING).")

    print("Fetching today's odds snapshot…")
    raw = _load_today_raw_odds_snapshot()
    if raw.empty:
        print("No odds rows returned; exiting.")
        return

    cleaned = _neon.pd.DataFrame()  # type: ignore[attr-defined]  # defensive: keep namespace explicit
    # ``clean_odds_data`` lives in nba_spreads; import directly to avoid bloating neon module.
    from nba_spreads.clean_odds_data import clean_odds_data  # local import keeps startup lean

    cleaned = clean_odds_data(raw, live=True)
    odds_ready = _neon.odds_db_prep(cleaned)
    if odds_ready.empty:
        print("Odds prep produced 0 rows; exiting.")
        return

    odds_ready, missing_fk = _neon.filter_odds_to_game_logs_fk(odds_ready, engine=engine)
    if odds_ready.empty:
        print("All odds rows are missing matching FK pairs in nba.game_logs; nothing inserted.")
        return

    n_odds = _neon.insert_odds(odds_ready, engine=engine)
    print(
        f"Inserted {n_odds} odds rows. Dropped {len(missing_fk)} missing FK pairs (see warnings above)."
    )


def main() -> None:
    """
    CLI entrypoint for GitHub Actions or local cron.

    This script is staging-free by design. If you later add ``nba.odds_staging`` for robustness,
    that logic belongs here (not in ``neon_training_sink.py``).
    """
    parser = argparse.ArgumentParser(description="Daily Neon ingestion (live odds + recent game logs).")
    parser.add_argument(
        "--days-back-game-logs",
        type=int,
        default=2,
        help="How many days back to refresh BRef game logs (default: 2).",
    )
    args = parser.parse_args()
    run_live_neon_update(days_back_game_logs=args.days_back_game_logs)


if __name__ == "__main__":
    main()
