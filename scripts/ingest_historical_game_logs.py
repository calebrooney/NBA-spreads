#!/usr/bin/env python3
"""
Scrape Basketball-Reference advanced team game logs (regular season + playoffs) for every
NBA team and every season from 2021-22 through the current BRef season, then optionally
write a CSV and/or insert into Neon using the same prep as ``neon_training_sink``.

``neon_training_sink`` only pulls game logs for dates covered by processed odds CSVs and
does not request playoff tables; this script fills full historical coverage first.

Rate limit: ``game_logs.scrape_team_adv_game_log`` sleeps ~3.3s per request (~10/min). For five
BRef seasons (all teams, reg + playoffs), that is about 300 requests → on the order of ~20
minutes of throttle time plus HTTP/parsing.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Repo root on sys.path for ``nba_spreads``
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nba_spreads.game_data import game_logs

# Load DB helpers from ``scripts/neon_training_sink.py`` (not a package).
_sink_spec = importlib.util.spec_from_file_location(
    "neon_training_sink",
    ROOT / "scripts" / "neon_training_sink.py",
)
_neon = importlib.util.module_from_spec(_sink_spec)
assert _sink_spec.loader is not None
_sink_spec.loader.exec_module(_neon)


def bref_season_end_years_through_today() -> list[int]:
    """
    Return BRef season end years from 2022 (2021-22) through the season that ``today``
    falls in, inclusive.
    """
    today = date.today()
    # Same mapping as ``neon_training_sink.bref_season_end_year``
    if today.month >= 10:
        current_end = today.year + 1
    elif today.month <= 6:
        current_end = today.year
    else:
        current_end = today.year + 1
    return list(range(2022, current_end + 1))


def scrape_clean(team: str, season_end: int, playoffs: bool) -> pd.DataFrame | None:
    """
    Fetch one advanced game log table and return cleaned rows, or None if unavailable.

    :param team: Three-letter abbreviation.
    :param season_end: BRef URL year (e.g. 2026 for 2025-26).
    :param playoffs: Regular season (False) or playoffs (True).
    """
    try:
        raw = game_logs.scrape_team_adv_game_log(team, season_end, playoffs=playoffs)
    except Exception as exc:  # noqa: BLE001 — BRef / HTML variance
        print(f"  skip {team} {season_end} playoffs={playoffs}: {exc}")
        return None
    if raw is None or raw.empty or len(raw) < 2:
        return None
    try:
        return game_logs.clean_team_log(raw, team)
    except Exception as exc:  # noqa: BLE001
        print(f"  clean failed {team} {season_end} playoffs={playoffs}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full historical BRef advanced game logs (reg + playoffs) for all teams."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write combined CSV (e.g. data/processed/historical_game_logs.csv).",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Insert into Neon via game_logs_db_prep / insert_game_logs (requires DATABASE_URL).",
    )
    args = parser.parse_args()
    if not args.csv and not args.db:
        parser.error(
            "Specify --csv PATH and/or --db. A 5-season full run is ~300 requests × ~3.3s throttle "
            "(order of ~20 minutes) plus network; refusing to scrape without a persistence target."
        )

    season_years = bref_season_end_years_through_today()
    n_req = len(game_logs.teams) * len(season_years) * 2
    est_min = n_req * 3.3 / 60.0
    print(
        f"BRef season end years: {season_years[0]}–{season_years[-1]} "
        f"({len(game_logs.teams)} teams × {len(season_years)} seasons × 2 phases ≈ {n_req} requests, "
        f"~{est_min:.0f} min throttle-only) …"
    )

    # One HTTP request per (team, season, phase); ``scrape_team_adv_game_log`` enforces ~10/min.
    frames: list[pd.DataFrame] = []
    for team in game_logs.teams:
        for season_end in season_years:
            # Regular season and playoffs are separate BRef tables; both map through ``clean_team_log``.
            for playoffs in (False, True):
                label = "post" if playoffs else "reg"
                print(f"{team} {season_end} {label} …")
                cleaned = scrape_clean(team, season_end, playoffs=playoffs)
                if cleaned is not None and not cleaned.empty:
                    frames.append(cleaned)

    if not frames:
        print("No rows collected; exiting.")
        return

    combined = pd.concat(frames, ignore_index=True)
    # Defensive: reg vs post should not duplicate the same (Game_ID, Team) row.
    # Do NOT dedupe on Game_ID alone: each game has two team-rows (one per team).
    combined = combined.drop_duplicates(subset=["Game_ID", "Team"], keep="first")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out, index=False)
        print(f"Wrote {len(combined)} rows to {out}")

    if args.db:
        # Second arg is unused in ``game_logs_db_prep``; placeholder for API compatibility.
        prepped = _neon.game_logs_db_prep(combined, "NBA")
        engine = _neon.get_db_engine()
        n = _neon.insert_game_logs(prepped, engine=engine)
        print(f"Inserted {n} rows into nba.game_logs.")


if __name__ == "__main__":
    main()
