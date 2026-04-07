## historical odds data has dupe rows, ie positive and negative for each team involved

## game margin column in BRef game logs will be standardized to (home team - away team), so can remove all spreads that give score from away team pov

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

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
    
    # Convert full team names to abbreviations for game_id
    def get_team_abbrev(full_name):
        """Convert full team name to abbreviation"""
        return teams_dict_reverse.get(full_name, None)
    
    # Add abbreviation columns
    df['home_team_abbrev'] = df['home_team'].apply(get_team_abbrev)
    df['away_team_abbrev'] = df['away_team'].apply(get_team_abbrev)
    
    # Extract date from commence_time (format: YYYY-MM-DDTHH:MM:SSZ)
    df['commence_date'] = pd.to_datetime(df['commence_time']).dt.date
    
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
    engine = create_engine(database_url)
    return engine

def insert_game_logs(df, engine=None, schema='nba', table='game_logs', if_exists='append'):
    """
    Insert game logs data into Neon database (fail-fast on duplicate keys).

    :param df: dataframe prepared by game_logs_db_prep()
    :param engine: SQLAlchemy engine (if None, creates new one)
    :param schema: database schema name (default: 'nba')
    :param table: table name (default: 'game_logs')
    :param if_exists: what to do if table exists ('append', 'replace', 'fail')
    :return: number of rows inserted
    """
    if engine is None:
        engine = get_db_engine()

    if df.empty:
        return 0

    df.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        method='multi'
    )

    return len(df)

def insert_odds(df, engine=None, schema='nba', table='odds', if_exists='append'):
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

    df.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        method='multi'
    )

    return len(df)


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
    if "game_id" in prepped.columns:
        prepped = prepped.drop_duplicates(subset=["game_id"], keep="first")
    return prepped


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

    cd = pd.to_datetime(combined["commence_time"], errors="coerce").dt.date
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

    min_date = pd.to_datetime(combined["commence_time"]).dt.date.min()
    max_date = pd.to_datetime(combined["commence_time"]).dt.date.max()
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
    n_odds = insert_odds(odds_ready, engine=engine)
    print(f"Inserted {n_logs} game_logs rows and {n_odds} odds rows.")


def _parse_opt_date(s: str | None) -> date | None:
    """Parse YYYY-MM-DD or return None."""
    if s is None:
        return None
    return pd.to_datetime(s, format="%Y-%m-%d").date()


if __name__ == "__main__":
    # One-time bulk load, no CLI required:
    # - loads all processed odds under data/processed/odds*.csv
    # - loads historical game logs from data/processed/historical_game_logs.csv
    # - inserts game_logs first, then odds (FK dependency)
    run_bulk_load()