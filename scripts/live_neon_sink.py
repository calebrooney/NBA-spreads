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
import logging
import os
import sys
import traceback
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


def _setup_loggers() -> tuple[logging.Logger, logging.Logger]:
    """
    Configure two file-backed loggers under ``logs/``:
    - ``logs/odds_live.log``: Odds API + odds insert path
    - ``logs/bref_game_logs.log``: Basketball-Reference scrape + game log inserts

    This is designed for GitHub Actions: artifacts will include these files after every run.
    """
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    def _make_logger(name: str, filename: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Avoid duplicate handlers if main() is invoked multiple times.
        if logger.handlers:
            return logger

        fh = logging.FileHandler(logs_dir / filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        return logger

    return _make_logger("live_neon.odds", "odds_live.log"), _make_logger("live_neon.bref", "bref_game_logs.log")


def _today_pacific() -> date:
    """
    Return today's calendar date in Pacific time.

    GitHub-hosted runners run in UTC; we pin "today" to Pacific to match your snapshot naming
    and the Odds API commenceTimeFrom/To window used in ``nba_spreads.fetch.fetch_csv``.
    """
    pacific = ZoneInfo("America/Los_Angeles")
    return datetime.now(pacific).date()


def _should_run_now_pacific(
    run_at_local: list[str],
    grace_minutes: int,
) -> bool:
    """
    Return True if current Pacific time is within a grace window of any configured HH:MM.

    This is how we make the schedule DST-proof: GitHub cron is UTC-only, so we schedule the
    workflow frequently and let this gate decide when to do real work.

    :param run_at_local: List like ["06:30", "11:00", "17:30"].
    :param grace_minutes: Allowed +/- window in minutes.
    """
    if grace_minutes < 0:
        raise ValueError("grace_minutes must be >= 0")

    pacific = ZoneInfo("America/Los_Angeles")
    now = datetime.now(pacific)
    now_min = now.hour * 60 + now.minute

    for hhmm in run_at_local:
        hh, mm = hhmm.split(":")
        target = int(hh) * 60 + int(mm)
        if abs(now_min - target) <= grace_minutes:
            return True
    return False


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
    odds_log, bref_log = _setup_loggers()

    pacific = ZoneInfo("America/Los_Angeles")
    now_pt = datetime.now(pacific)
    tzname = now_pt.tzname()
    odds_log.info("start live ingest (PT tz=%s)", tzname)
    bref_log.info("start game logs refresh (PT tz=%s)", tzname)

    # Fail fast if required env vars are missing (common in CI misconfiguration).
    if not os.getenv("DATABASE_URL"):
        odds_log.error("missing env: DATABASE_URL")
        bref_log.error("missing env: DATABASE_URL")
        raise ValueError("DATABASE_URL is required (set as a GitHub Actions secret).")
    if not os.getenv("ODDS_API_KEY_FREE"):
        odds_log.error("missing env: ODDS_API_KEY_FREE")
        bref_log.error("missing env: ODDS_API_KEY_FREE")
        raise ValueError("ODDS_API_KEY_FREE is required (set as a GitHub Actions secret).")

    engine = _neon.get_db_engine()

    bref_log.info("Refreshing game logs (last %s days)…", days_back_game_logs)
    recent_logs = _scrape_recent_game_logs(days_back_game_logs)
    n_logs = _neon.insert_game_logs(recent_logs, engine=engine)
    bref_log.info("Inserted %s game_logs rows (ON CONFLICT DO NOTHING).", n_logs)

    odds_log.info("Fetching today's odds snapshot…")
    raw = _load_today_raw_odds_snapshot()
    if raw.empty:
        odds_log.info("No odds rows returned; exiting.")
        return

    # ``clean_odds_data`` lives in nba_spreads; import directly to avoid bloating neon module.
    from nba_spreads.clean_odds_data import clean_odds_data  # local import keeps startup lean

    cleaned = clean_odds_data(raw, live=True)
    odds_ready = _neon.odds_db_prep(cleaned)
    if odds_ready.empty:
        odds_log.info("Odds prep produced 0 rows; exiting.")
        return

    odds_ready, missing_fk = _neon.filter_odds_to_game_logs_fk(odds_ready, engine=engine)
    if odds_ready.empty:
        odds_log.info("All odds rows are missing matching FK pairs in nba.game_logs; nothing inserted.")
        return

    n_odds = _neon.insert_odds(odds_ready, engine=engine)
    odds_log.info(
        "Inserted %s odds rows. Dropped %s missing FK pairs.",
        n_odds,
        len(missing_fk),
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
    parser.add_argument(
        "--run-at-local",
        type=str,
        default="hourly-0600-0000",
        help=(
            "Pacific-time schedule gate. Use either:\n"
            "- 'hourly-0600-0000' (default): run hourly at :00 from 06:00 through 00:00 PT\n"
            "- or a comma-separated list like '06:30,11:00,17:30'."
        ),
    )
    parser.add_argument(
        "--grace-minutes",
        type=int,
        default=7,
        help="Allowed +/- minutes around --run-at-local times (default: 7).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run regardless of local time gate (useful for workflow_dispatch testing).",
    )
    args = parser.parse_args()

    if args.run_at_local.strip() == "hourly-0600-0000":
        run_times = [f"{h:02d}:00" for h in list(range(6, 24)) + [0]]
    else:
        run_times = [s.strip() for s in args.run_at_local.split(",") if s.strip()]
    if not args.force and not _should_run_now_pacific(run_times, grace_minutes=args.grace_minutes):
        odds_log, bref_log = _setup_loggers()
        pacific = ZoneInfo("America/Los_Angeles")
        now_pt = datetime.now(pacific)
        tzname = now_pt.tzname()
        msg = f"skip (gate): now={now_pt.strftime('%Y-%m-%d %H:%M')} {tzname}, run_at={run_times}, grace={args.grace_minutes}m"
        odds_log.info(msg)
        bref_log.info(msg)
        return

    try:
        run_live_neon_update(days_back_game_logs=args.days_back_game_logs)
    except Exception as exc:  # noqa: BLE001 — we want a log + nonzero exit
        odds_log, bref_log = _setup_loggers()
        tb = traceback.format_exc()
        odds_log.error("fatal error: %s\n%s", exc, tb)
        bref_log.error("fatal error: %s\n%s", exc, tb)
        raise


if __name__ == "__main__":
    main()
