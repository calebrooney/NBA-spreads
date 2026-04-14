/*
Create ``nba.odds_staging`` (no FK) for raw/live snapshot ingestion.

Rationale:
- ``nba.odds`` is assumed to have a FK to ``nba.game_logs`` on (game_id, team).
- Raw snapshots can be ingested before game logs exist; staging holds those rows until a
  reconcile step inserts only FK-valid rows into ``nba.odds``.
*/

create schema if not exists nba;

create table if not exists nba.odds_staging (
    "timestamp" timestamptz null,
    snapshot_time_pacific timestamptz null,
    snapshot_time_utc timestamptz null,
    commence_time timestamptz null,
    home_team text null,
    away_team text null,
    bookmaker_key text null,
    bookmaker_title text null,
    bookmaker_last_update timestamptz null,
    team char(3) null,
    price integer null,
    point numeric null,
    game_id char(18) null
);

-- Reconcile join key (game_id, team) should be fast.
create index if not exists odds_staging_game_id_team_idx
    on nba.odds_staging (game_id, team);

-- Optional helper indexes for common filtering/debugging.
create index if not exists odds_staging_commence_time_idx
    on nba.odds_staging (commence_time);

create index if not exists odds_staging_snapshot_time_pacific_idx
    on nba.odds_staging (snapshot_time_pacific);

