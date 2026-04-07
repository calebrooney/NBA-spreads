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


def main() -> None:
    """Placeholder until daily automation is implemented."""
    print(
        "live_neon_update: not implemented yet. "
        "Use scripts/neon_training_sink.py for one-time bulk loads."
    )
    print(
        "Intended follow-up: today's NBAodds CSV under data/raw, clean_odds_data(live=True), "
        "odds_db_prep, insert_odds; scrape current BRef season, game_logs_db_prep, insert_game_logs."
    )


if __name__ == "__main__":
    main()
