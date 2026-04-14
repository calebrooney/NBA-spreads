## historical odds data has dupe rows, ie positive and negative for each team involved

## game margin column in BRef game logs will be standardized to (home team - away team), so can remove all spreads that give score from away team pov

import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Ensure repo root is on sys.path so we can import ``nba_spreads`` from scripts.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nba_spreads.clean_odds_data import clean_odds_data

teams_dict = {'ATL':'Atlanta Hawks', 'BOS':'Boston Celtics', 'BRK':'Brooklyn Nets', 'CHI':'Chicago Bulls', 'CHO':'Charlotte Hornets', 'CLE':'Cleveland Cavaliers', 'DAL':'Dallas Mavericks', 'DEN':'Denver Nuggets', 'DET':'Detroit Pistons', 'GSW':'Golden State Warriors', 'HOU':'Houston Rockets', 'IND':'Indiana Pacers', 'LAC':'LA Clippers', 'LAL':'Los Angeles Lakers', 'MEM':'Memphis Grizzlies', 'MIA':'Miami Heat', 'MIL':'Milwaukee Bucks', 'MIN':'Minnesota Timberwolves', 'NOP':'New Orleans Pelicans', 'NYK':'New York Knicks', 'OKC':'Oklahoma City Thunder', 'ORL':'Orlando Magic', 'PHI':'Philadelphia 76ers', 'PHO':'Phoenix Suns', 'POR':'Portland Trail Blazers', 'SAC':'Sacramento Kings', 'SAS':'San Antonio Spurs', 'TOR':'Toronto Raptors', 'UTA':'Utah Jazz', 'WAS':'Washington Wizards'}

# Reverse lookup: full team name -> abbreviation
teams_dict_reverse = {v: k for k, v in teams_dict.items()}

# for odds data:
# create game_id col: 
# if 'home_team' != 'name' in row, then drop row
# drop previous and next timestamp columns

def odds_db_prep(df):
    """
    Prepare odds data for database insertion.
    
    Transforms cleaned odds data to match nba.odds schema:
    - Creates game_id in format YYYY-MM-DD-HOME-AWAY (using abbreviations)
    - Filters to only home team perspective rows
    - Adds team column (CHAR(3)) from name field
    - Drops unnecessary timestamp columns
    - Ensures column names match schema
    
    :param df: flattened odds (same shape as ``clean_odds_data`` output or ``data/processed/odds*.csv``)
    :return: dataframe ready for database insertion
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Normalize key string fields to make team-name lookups deterministic.
    # Some source CSVs contain trailing whitespace or minor formatting inconsistencies.
    for col in ("home_team", "away_team", "name"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    
    # Convert full team names to abbreviations for game_id
    def get_team_abbrev(full_name):
        """Convert full team name to abbreviation"""
        return teams_dict_reverse.get(full_name, None)
    
    # Add abbreviation columns
    df['home_team_abbrev'] = df['home_team'].apply(get_team_abbrev)
    df['away_team_abbrev'] = df['away_team'].apply(get_team_abbrev)
    
    # Extract the *local* game date from commence_time.
    #
    # ``commence_time`` comes in as UTC (e.g. "...Z"). Basketball-Reference game logs use
    # the local/US calendar date of the game. For late-night games, the UTC date can be
    # the next day, which would create a mismatched ``game_id`` and break the
    # ``odds(game_id, team) -> game_logs(game_id, team)`` foreign key.
    #
    # We therefore convert to a US timezone before taking ``.date``.
    # (Eastern is the most consistent reference for NBA schedules and matches BRef dates
    # much better than UTC.)
    commence_utc = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    df["commence_date"] = commence_utc.dt.tz_convert("America/New_York").dt.date
    
    # Create game_id in format: YYYY-MM-DD-HOME-AWAY
    df.loc[:, "game_id"] = df.apply(
        lambda row: f"{row['commence_date']}-{row['home_team_abbrev']}-{row['away_team_abbrev']}" 
        if pd.notna(row['home_team_abbrev']) and pd.notna(row['away_team_abbrev']) else None,
        axis=1
    )
    
    # Filter rows where home_team == name (keep only home team perspective)
    df = df[df['home_team'] == df['name']].copy()
    
    # Add team column (CHAR(3)) by converting name (full team name) to abbreviation
    df.loc[:, "team"] = df['name'].apply(get_team_abbrev)
    
    # Drop rows where team abbreviation couldn't be found
    df = df[df['team'].notna()].copy()
    
    # Ensure team is exactly 3 characters (CHAR(3))
    df['team'] = df['team'].astype(str).str[:3]
    
    # Drop unnecessary columns
    columns_to_drop = ["previous_timestamp", "next_timestamp", "home_team_abbrev", "away_team_abbrev", "commence_date"]
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=existing_cols_to_drop, inplace=True)
    
    # Rename columns to match schema (lowercase, underscores)
    # Schema columns: timestamp, snapshot_time_pacific, snapshot_time_utc, commence_time,
    # home_team, away_team, bookmaker_key, bookmaker_title, bookmaker_last_update,
    # team, price, point, game_id
    column_mapping = {
        'id': 'game_id',  # This will be overwritten, but ensure we have game_id
    }
    
    # If 'id' column exists and we created game_id, drop id
    if 'id' in df.columns and 'game_id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    
    # Ensure game_id is CHAR(18) - should be exactly 18 chars: YYYY-MM-DD-XXX-XXX
    # Drop rows where we couldn't build a game_id; these will violate the FK.
    n_null_game_id = int(df["game_id"].isna().sum()) if "game_id" in df.columns else 0
    if n_null_game_id:
        df = df[df["game_id"].notna()].copy()
    df['game_id'] = df['game_id'].astype(str).str[:18]
    
    # Ensure data types match schema
    # Convert timestamps to datetime with timezone
    timestamp_cols = ['timestamp', 'snapshot_time_pacific', 'snapshot_time_utc', 'commence_time', 'bookmaker_last_update']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Ensure numeric columns
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'point' in df.columns:
        df['point'] = pd.to_numeric(df['point'], errors='coerce')
    
    # Select and order columns to match schema
    schema_columns = [
        'timestamp', 'snapshot_time_pacific', 'snapshot_time_utc', 'commence_time',
        'home_team', 'away_team', 'bookmaker_key', 'bookmaker_title', 'bookmaker_last_update',
        'team', 'price', 'point', 'game_id'
    ]
    
    # Keep only columns that exist in dataframe and match schema
    available_cols = [col for col in schema_columns if col in df.columns]
    df = df[available_cols].copy()
    
    return df

def game_logs_db_prep(df, team):
    """
    Prepare game logs data for database insertion.
    
    Takes output from clean_team_log() and maps columns to match nba.game_logs schema exactly.
    Preserves special characters in column names (%, /) as PostgreSQL allows them with quotes.
    
    :param df: dataframe from clean_team_log() function
    :param team: team abbreviation (for validation)
    :return: dataframe ready for database insertion
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Rename 3PAr to _3par (lowercase, starts with underscore)
    if '3PAr' in df.columns:
        df.rename(columns={'3PAr': '_3par'}, inplace=True)
    
    # Map column names from clean_team_log() output to schema
    # Schema preserves special characters and case for percentage columns
    column_mapping = {
        'Game': 'game',
        'Date': 'date',
        'Home': 'home',
        'Opp': 'opp',
        'Tm_Score': 'tm_score',
        'Opp_Score': 'opp_score',
        'OT': 'ot',
        'ORtg': 'ortg',
        'DRtg': 'drtg',
        'Pace': 'pace',
        'FTr': 'ftr',
        'Home_Margin': 'home_margin',
        'Team': 'team',
        'Game_ID': 'game_id',
        # Percentage columns preserve case and special characters
        'TS%': 'TS%',
        'TRB%': 'TRB%',
        'AST%': 'AST%',
        'STL%': 'STL%',
        'BLK%': 'BLK%',
        'eFG%': 'eFG%',
        'TOV%': 'TOV%',
        'ORB%': 'ORB%',
        'FT/FGA': 'FT/FGA',
        'Opp_eFG%': 'Opp_eFG%',
        'Opp_TOV%': 'Opp_TOV%',
        'Opp_ORB%': 'Opp_ORB%',
        'Opp_FT/FGA': 'Opp_FT/FGA',
    }
    
    # Apply column mapping (only for columns that exist)
    existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
    df.rename(columns=existing_mappings, inplace=True)
    
    # Ensure data types match schema
    # SMALLINT columns
    smallint_cols = ['game', 'tm_score', 'opp_score', 'home_margin']
    for col in smallint_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # Nullable integer
    
    # BOOLEAN columns
    bool_cols = ['home', 'ot']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    # DATE column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    
    # CHAR(3) columns
    char3_cols = ['opp', 'team']
    for col in char3_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str[:3]
    
    # CHAR(18) column
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str).str[:18]
    
    # NUMERIC columns with specific precisions
    # NUMERIC(4, 1): ortg, drtg, pace, TRB%, AST%, STL%, BLK%, TOV%, ORB%, Opp_TOV%, Opp_ORB%
    numeric_4_1_cols = ['ortg', 'drtg', 'pace', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'ORB%', 'Opp_TOV%', 'Opp_ORB%']
    for col in numeric_4_1_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # NUMERIC(5, 3): ftr, _3par, TS%, eFG%, FT/FGA, Opp_eFG%, Opp_FT/FGA
    numeric_5_3_cols = ['ftr', '_3par', 'TS%', 'eFG%', 'FT/FGA', 'Opp_eFG%', 'Opp_FT/FGA']
    for col in numeric_5_3_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Select columns in schema order
    schema_columns = [
        'game', 'date', 'home', 'opp', 'tm_score', 'opp_score', 'ot',
        'ortg', 'drtg', 'pace', 'ftr', '_3par', 'TS%', 'TRB%', 'AST%', 'STL%', 'BLK%',
        'eFG%', 'TOV%', 'ORB%', 'FT/FGA', 'Opp_eFG%', 'Opp_TOV%', 'Opp_ORB%', 'Opp_FT/FGA',
        'home_margin', 'team', 'game_id'
    ]
    
    # Keep only columns that exist in dataframe and match schema
    available_cols = [col for col in schema_columns if col in df.columns]
    df = df[available_cols].copy()
    
    return df

def get_db_engine():
    """
    Create SQLAlchemy engine for Neon PostgreSQL database.
    
    :return: SQLAlchemy engine
    """
    # Load .env from repo root (one level above /scripts)
    ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=ENV_PATH)
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing. Your .env did not load or is malformed.")
    
    # Create engine
    # Hide bound parameters in exceptions so failures don't dump huge payloads to the terminal.
    engine = create_engine(database_url, hide_parameters=True)
    return engine

def insert_game_logs(
    df,
    engine=None,
    schema: str = 'nba',
    table: str = 'game_logs',
    if_exists: str = 'append',
    chunksize: int = 1000,
):
    """
    Insert game logs data into Neon database.

    This loader is meant to be safe to re-run. ``nba.game_logs`` has a primary key on
    ``(game_id, team)``, so a naive append will fail if you've already loaded any rows.
    We therefore use Postgres ``ON CONFLICT DO NOTHING`` on that key and insert in chunks.

    :param df: dataframe prepared by game_logs_db_prep()
    :param engine: SQLAlchemy engine (if None, creates new one)
    :param schema: database schema name (default: 'nba')
    :param table: table name (default: 'game_logs')
    :param if_exists: what to do if table exists ('append', 'replace', 'fail')
    :param chunksize: number of rows per batch insert
    :return: number of rows actually inserted (excludes conflicts)
    """
    if engine is None:
        engine = get_db_engine()

    if df.empty:
        return 0

    try:
        # Reflect the existing table so SQLAlchemy can properly quote odd column names
        # like "TS%" and "FT/FGA" during inserts.
        md = MetaData(schema=schema)
        tbl = Table(table, md, autoload_with=engine)

        # Convert NaN/NaT to None so psycopg2 sends proper SQL NULLs.
        clean = df.where(pd.notna(df), None)
        rows = clean.to_dict(orient="records")

        # Insert with ON CONFLICT DO NOTHING on the PK (game_id, team) so reruns are safe.
        inserted = 0
        with engine.begin() as conn:
            for i in range(0, len(rows), chunksize):
                batch = rows[i : i + chunksize]
                if not batch:
                    continue
                stmt = (
                    pg_insert(tbl)
                    .values(batch)
                    .on_conflict_do_nothing(index_elements=["game_id", "team"])
                )
                res = conn.execute(stmt)
                # psycopg2 returns inserted rows for INSERT .. ON CONFLICT DO NOTHING.
                inserted += int(res.rowcount or 0)
    except SQLAlchemyError as e:
        orig = getattr(e, "orig", None)
        orig_msg = f"{type(orig).__name__}: {orig}" if orig is not None else str(e)
        raise RuntimeError(
            f"Failed inserting game logs into {schema}.{table} (rows={len(df)}, chunksize={chunksize}). "
            f"Underlying error: {orig_msg}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed inserting game logs into {schema}.{table} (rows={len(df)}, chunksize={chunksize}). "
            f"Unexpected error: {type(e).__name__}: {e}"
        ) from e

    return inserted

def insert_odds(
    df,
    engine=None,
    schema: str = 'nba',
    table: str = 'odds',
    if_exists: str = 'append',
    chunksize: int = 1000,
):
    """
    Insert odds data into Neon database (fail-fast on duplicate keys).

    Note: game_logs must be inserted first due to foreign key constraint.

    :param df: dataframe prepared by odds_db_prep()
    :param engine: SQLAlchemy engine (if None, creates new one)
    :param schema: database schema name (default: 'nba')
    :param table: table name (default: 'odds')
    :param if_exists: what to do if table exists ('append', 'replace', 'fail')
    :return: number of rows inserted
    """
    if engine is None:
        engine = get_db_engine()

    if df.empty:
        return 0

    # Use a conservative chunksize to avoid Postgres parameter limits and massive
    # error dumps that include entire multi-row INSERT statements.
    try:
        df.to_sql(
            name=table,
            con=engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            method='multi',
            chunksize=chunksize,
        )
    except SQLAlchemyError as e:
        orig = getattr(e, "orig", None)
        orig_msg = f"{type(orig).__name__}: {orig}" if orig is not None else str(e)
        raise RuntimeError(
            f"Failed inserting odds into {schema}.{table} (rows={len(df)}, chunksize={chunksize}). "
            f"Underlying error: {orig_msg}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed inserting odds into {schema}.{table} (rows={len(df)}, chunksize={chunksize}). "
            f"Unexpected error: {type(e).__name__}: {e}"
        ) from e

    return len(df)


def filter_odds_to_game_logs_fk(
    odds_df: pd.DataFrame,
    engine,
    schema: str = "nba",
    game_logs_table: str = "game_logs",
    sample_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter odds rows to those whose (game_id, team) foreign keys exist in ``nba.game_logs``.

    This prevents a bulk odds insert from failing halfway through due to a small number of
    missing game logs rows (e.g. incomplete historical scrape / CSV gaps).

    :param odds_df: Odds dataframe prepared by ``odds_db_prep``.
    :param engine: SQLAlchemy engine connected to Neon.
    :param schema: Schema name containing the game logs table.
    :param game_logs_table: Game logs table name.
    :param sample_n: Number of missing key examples to print for debugging.
    :return: (filtered_odds_df, missing_keys_df) where missing_keys_df has columns [game_id, team].
    """
    if odds_df.empty:
        return odds_df, pd.DataFrame(columns=["game_id", "team"])
    if not {"game_id", "team"}.issubset(odds_df.columns):
        raise ValueError("odds_df must contain columns: game_id, team")

    # Unique FK pairs implied by odds.
    odds_keys = odds_df[["game_id", "team"]].dropna().drop_duplicates()
    if odds_keys.empty:
        return odds_df.iloc[0:0].copy(), pd.DataFrame(columns=["game_id", "team"])

    # Query existing keys from DB for only the involved game_ids (reduces DB work).
    game_ids = odds_keys["game_id"].dropna().unique().tolist()
    if not game_ids:
        return odds_df.iloc[0:0].copy(), odds_keys.copy()

    qry = text(
        f"""
        select game_id, team
        from {schema}.{game_logs_table}
        where game_id = any(:game_ids)
        """
    )
    existing = pd.read_sql(qry, con=engine, params={"game_ids": game_ids})
    existing = existing.drop_duplicates(subset=["game_id", "team"])

    merged = odds_keys.merge(existing, on=["game_id", "team"], how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"][["game_id", "team"]].copy()

    if not missing.empty:
        # High-signal examples for debugging (don’t dump huge output).
        examples = missing.head(sample_n).to_dict(orient="records")
        print(
            f"Warning: {len(missing)} odds FK pairs missing from {schema}.{game_logs_table}. "
            f"Dropping those odds rows. Examples: {examples}"
        )

    # Keep only odds rows whose FK pair exists.
    keep_keys = existing
    filtered = odds_df.merge(keep_keys, on=["game_id", "team"], how="inner")
    return filtered, missing


def bref_season_end_year(d: date) -> int:
    """
    Map a calendar date to the Basketball-Reference season URL year (e.g. 2025-26 -> 2026).

    October–December map to the following calendar year; January–June map to the same year;
    July–September map to the upcoming season (preseason / gap between seasons).
    """
    if d.month >= 10:
        return d.year + 1
    if d.month <= 6:
        return d.year
    return d.year + 1


def season_years_in_range(d0: date, d1: date) -> list[int]:
    """Return sorted unique BRef season end years for every day in [d0, d1] inclusive."""
    years: set[int] = set()
    cur = d0
    step = timedelta(days=1)
    while cur <= d1:
        years.add(bref_season_end_year(cur))
        cur += step
    return sorted(years)


def discover_processed_odds_csv_paths(repo_root: Path) -> list[Path]:
    """
    Find processed, flattened odds CSVs under ``data/processed/`` (e.g. ``odds_2021_2022.csv``).
    """
    proc = repo_root / "data" / "processed"
    if not proc.is_dir():
        return []
    return sorted(proc.glob("odds*.csv"))


def discover_processed_game_logs_csv_path(repo_root: Path) -> Path:
    """
    Find the processed historical game logs CSV under ``data/processed/``.

    This script is meant to be a one-time, zero-CLI bulk loader. To keep usage dead simple,
    we auto-discover the expected file instead of requiring a CLI argument.

    Expected filename: ``historical_game_logs.csv``.

    :param repo_root: repository root path
    :return: resolved path to the historical game logs CSV
    """
    proc = repo_root / "data" / "processed"

    # Primary expected filename.
    canonical = proc / "historical_game_logs.csv"
    if canonical.is_file():
        return canonical

    # Secondary fallback for slightly different naming.
    matches = sorted(proc.glob("*historical*game*logs*.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find historical game logs CSV under {proc}. "
            "Expected 'historical_game_logs.csv'."
        )
    raise ValueError(
        f"Multiple candidate historical game logs CSVs found under {proc}: {matches}. "
        "Rename the one you want to 'historical_game_logs.csv'."
    )

def load_historical_game_logs_csv(
    csv_path: Path,
    min_date: date,
    max_date: date,
) -> pd.DataFrame:
    """
    Load pre-scraped historical game logs CSV, filter to date window, and prepare for DB insert.

    :param csv_path: CSV path from ``scripts/ingest_historical_game_logs.py`` output.
    :param min_date: Inclusive lower date bound implied by selected odds rows.
    :param max_date: Inclusive upper date bound implied by selected odds rows.
    :return: dataframe ready for ``insert_game_logs``.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Historical game logs CSV not found: {csv_path}")

    logs = pd.read_csv(csv_path)
    if logs.empty:
        return pd.DataFrame()
    if "Date" not in logs.columns:
        raise ValueError(
            f"Historical game logs CSV is missing required column 'Date': {csv_path}"
        )

    # Restrict to the same date window as selected odds rows so FK coverage matches insert scope.
    game_dates = pd.to_datetime(logs["Date"], errors="coerce").dt.date
    logs = logs.assign(_date=game_dates).dropna(subset=["_date"])
    logs = logs[(logs["_date"] >= min_date) & (logs["_date"] <= max_date)].copy()
    logs = logs.drop(columns=["_date"])
    if logs.empty:
        return pd.DataFrame()

    # ``game_logs_db_prep`` handles final column mapping/types for Neon schema.
    prepped = game_logs_db_prep(logs, "NBA")
    # ``nba.game_logs`` is keyed by (game_id, team). Do NOT dedupe on game_id alone,
    # or you will drop one of the two team-rows per game and odds inserts will fail FK checks.
    if {"game_id", "team"}.issubset(prepped.columns):
        prepped = prepped.drop_duplicates(subset=["game_id", "team"], keep="first")
    return prepped


def discover_raw_odds_snapshot_paths(
    repo_root: Path,
    since: date,
    until: date,
) -> list[Path]:
    """
    Find ``data/raw/NBAodds_YYYY-MM-DD_HHMM.csv`` snapshot files within an inclusive date window.

    The backfill uses only already-collected snapshots (no paid historical endpoints).

    :param repo_root: Repository root.
    :param since: Inclusive lower bound on the local (filename) day.
    :param until: Inclusive upper bound on the local (filename) day.
    :return: Sorted list of matching paths.
    """
    raw_dir = repo_root / "data" / "raw"
    if not raw_dir.is_dir():
        return []

    # Example: NBAodds_2026-04-09_0630.csv
    pat = re.compile(r"^NBAodds_(\d{4}-\d{2}-\d{2})_\d{4}\.csv$")
    paths: list[Path] = []
    for p in sorted(raw_dir.glob("NBAodds_*.csv")):
        m = pat.match(p.name)
        if not m:
            continue
        d = pd.to_datetime(m.group(1), format="%Y-%m-%d").date()
        if since <= d <= until:
            paths.append(p)
    return paths


def load_and_prepare_raw_odds_snapshot(csv_path: Path) -> pd.DataFrame:
    """
    Load one raw live snapshot CSV and prepare it for direct insertion into ``nba.odds``.

    :param csv_path: Path like ``data/raw/NBAodds_YYYY-MM-DD_HHMM.csv``.
    :return: Dataframe prepared by ``odds_db_prep`` (may be empty).
    """
    raw = pd.read_csv(csv_path)
    if raw.empty:
        return pd.DataFrame()

    # Raw snapshots are in the Odds API "live" shape (nested bookmakers/markets/outcomes).
    cleaned = clean_odds_data(raw, live=True)
    if cleaned.empty:
        return pd.DataFrame()

    return odds_db_prep(cleaned)


def scrape_missing_game_logs_window(min_date: date, max_date: date) -> pd.DataFrame:
    """
    Scrape Basketball-Reference advanced team game logs for a narrow date window.

    This is intended for small gaps where the processed CSV is known to end early (e.g. it ends
    at 2026-04-06 but you need 2026-04-07→2026-04-13).

    :param min_date: Inclusive start date for game logs.
    :param max_date: Inclusive end date for game logs.
    :return: Prepared dataframe ready for ``insert_game_logs`` (may be empty if no games).
    """
    from nba_spreads.game_data import game_logs

    years = season_years_in_range(min_date, max_date)
    frames: list[pd.DataFrame] = []
    for team in game_logs.teams:
        try:
            df = game_logs.fetch_team_logs_for_date_window(
                team=team,
                season_years=years,
                min_date=min_date,
                max_date=max_date,
            )
        except Exception as exc:  # noqa: BLE001 — scraping variance
            print(f"skip BRef {team} {years} {min_date}..{max_date}: {exc}")
            continue
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Game_ID", "Team"], keep="first")
    return game_logs_db_prep(combined, "NBA")


def run_backfill_from_raw(
    since: date,
    until: date,
    game_logs_csv: str | None = None,
) -> None:
    """
    Backfill from raw snapshots directly into FK-protected tables (no staging).

    Steps:
    - Insert game logs from processed CSV for ``since..min(until, 2026-04-06)``.
    - Scrape BRef game logs for ``max(since, 2026-04-07)..until`` (the known CSV gap).
    - Load/clean raw odds snapshots from ``data/raw`` for ``since..until``.
    - Filter odds rows to only FK-valid ``(game_id, team)`` pairs, then insert into ``nba.odds``.

    :param since: Inclusive start date (YYYY-MM-DD).
    :param until: Inclusive end date (YYYY-MM-DD).
    :param game_logs_csv: Optional override path (relative to repo root or absolute).
    """
    repo_root = Path(__file__).resolve().parents[1]
    engine = get_db_engine()

    raw_paths = discover_raw_odds_snapshot_paths(repo_root, since=since, until=until)
    if not raw_paths:
        raise FileNotFoundError(
            f"No raw snapshot files under {repo_root / 'data' / 'raw'} for {since}..{until}."
        )

    csv_cutoff = date(2026, 4, 6)
    csv_min = since
    csv_max = min(until, csv_cutoff)

    if game_logs_csv is None:
        csv_path = discover_processed_game_logs_csv_path(repo_root)
    else:
        csv_path = Path(game_logs_csv)
        if not csv_path.is_absolute():
            csv_path = repo_root / csv_path

    if csv_min <= csv_max:
        print(f"Loading game logs from CSV for {csv_min}..{csv_max}: {csv_path}")
        logs_csv = load_historical_game_logs_csv(csv_path, csv_min, csv_max)
        n_csv = insert_game_logs(logs_csv, engine=engine)
        print(f"Inserted {n_csv} game_logs rows from CSV.")

    scrape_min = max(since, csv_cutoff + timedelta(days=1))
    scrape_max = until
    if scrape_min <= scrape_max:
        print(f"Scraping BRef game logs for {scrape_min}..{scrape_max} (CSV gap).")
        scraped = scrape_missing_game_logs_window(scrape_min, scrape_max)
        n_scraped = insert_game_logs(scraped, engine=engine)
        print(f"Inserted {n_scraped} game_logs rows from BRef scrape.")

    total_prepared = 0
    total_inserted = 0
    for p in raw_paths:
        prepared = load_and_prepare_raw_odds_snapshot(p)
        if prepared.empty:
            continue
        total_prepared += len(prepared)

        filtered, _missing = filter_odds_to_game_logs_fk(prepared, engine=engine)
        if filtered.empty:
            continue
        total_inserted += insert_odds(filtered, engine=engine)

    print(
        f"Prepared {total_prepared} odds rows from {len(raw_paths)} raw files; "
        f"inserted {total_inserted} FK-valid odds rows into nba.odds."
    )


def run_bulk_load(
    since: date | None = None,
    until: date | None = None,
    game_logs_csv: str | None = None,
) -> None:
    """
    One-time bulk load: read processed odds from ``data/processed/odds*.csv`` and insert into Neon.
    Game logs are sourced from processed CSV (pre-scraped via ingest_historical_game_logs.py),
    filtered to the odds date window so odds foreign keys can be satisfied.
    """
    repo_root = Path(__file__).resolve().parents[1]
    csv_paths = discover_processed_odds_csv_paths(repo_root)
    if not csv_paths:
        raise FileNotFoundError(
            f"No odds*.csv files under {repo_root / 'data' / 'processed'}."
        )

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        frames.append(pd.read_csv(path))

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        print("No rows after loading processed odds; exiting.")
        return

    if "commence_time" not in combined.columns:
        raise ValueError("Combined odds data has no commence_time column.")

    # Use the same local-date logic as ``odds_db_prep`` so our game-logs date window
    # matches the ``game_id`` dates derived from odds.
    commence_utc = pd.to_datetime(combined["commence_time"], utc=True, errors="coerce")
    cd = commence_utc.dt.tz_convert("America/New_York").dt.date
    combined = combined.assign(_commence_date=cd)
    combined = combined.dropna(subset=["_commence_date"])
    if since is not None:
        combined = combined[combined["_commence_date"] >= since]
    if until is not None:
        combined = combined[combined["_commence_date"] <= until]
    combined = combined.drop(columns=["_commence_date"])

    if combined.empty:
        print("No rows in selected date window; exiting.")
        return

    commence_utc2 = pd.to_datetime(combined["commence_time"], utc=True, errors="coerce")
    commence_local_date = commence_utc2.dt.tz_convert("America/New_York").dt.date
    min_date = commence_local_date.min()
    max_date = commence_local_date.max()
    odds_ready = odds_db_prep(combined)

    if game_logs_csv is None:
        csv_path = discover_processed_game_logs_csv_path(repo_root)
    else:
        csv_path = Path(game_logs_csv)
        if not csv_path.is_absolute():
            csv_path = repo_root / csv_path
    print(f"Loading game logs from CSV (no BRef scraping): {csv_path}")
    game_logs_df = load_historical_game_logs_csv(csv_path, min_date, max_date)

    if game_logs_df.empty and not odds_ready.empty:
        raise ValueError(
            "Game logs dataframe is empty but odds are not; cannot satisfy odds → game_logs FK."
        )

    engine = get_db_engine()
    n_logs = insert_game_logs(game_logs_df, engine=engine)
    odds_ready, missing_fk = filter_odds_to_game_logs_fk(odds_ready, engine=engine)
    if odds_ready.empty:
        raise ValueError(
            "All prepared odds rows are missing matching (game_id, team) keys in nba.game_logs. "
            "Your historical_game_logs.csv is likely incomplete for this date range."
        )
    n_odds = insert_odds(odds_ready, engine=engine)
    print(f"Inserted {n_logs} game_logs rows and {n_odds} odds rows.")


def _parse_opt_date(s: str | None) -> date | None:
    """Parse YYYY-MM-DD or return None."""
    if s is None:
        return None
    return pd.to_datetime(s, format="%Y-%m-%d").date()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Neon bulk loaders.\n\n"
            "processed: load data/processed/odds*.csv + historical_game_logs.csv (FK-protected).\n"
            "raw-backfill: backfill from data/raw snapshots and (if needed) scrape missing game logs.\n"
            "Note: this script does NOT use odds staging; that belongs in live_neon_sink.py."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="processed",
        choices=["processed", "raw-backfill"],
        help="Which loader to run (default: processed).",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Inclusive start date (YYYY-MM-DD). Required for --mode raw-backfill; optional for processed.",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="Inclusive end date (YYYY-MM-DD). Required for --mode raw-backfill; optional for processed.",
    )
    parser.add_argument(
        "--game-logs-csv",
        type=str,
        default=None,
        help="Optional override path to historical game logs CSV.",
    )
    args = parser.parse_args()

    since_d = _parse_opt_date(args.since)
    until_d = _parse_opt_date(args.until)

    if args.mode == "processed":
        run_bulk_load(since=since_d, until=until_d, game_logs_csv=args.game_logs_csv)
    else:
        if since_d is None or until_d is None:
            raise SystemExit("--mode raw-backfill requires --since and --until (YYYY-MM-DD).")
        run_backfill_from_raw(since=since_d, until=until_d, game_logs_csv=args.game_logs_csv)