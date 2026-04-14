from datetime import date

# unused but maybe useful for later
from basketball_reference_scraper.players import get_stats, get_game_logs, get_player_headshot
from basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats, get_team_misc
from basketball_reference_scraper.injury_report import get_injury_report

import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import time

teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

def dedupe_columns(cols):
    '''
    Dedupe columns by appending .1, .2, etc to duplicates
    Use: df.columns = dedupe_columns(df.columns)
    :param cols: df.columns list
    '''
    seen = {}
    out = []
    for c in cols:
        k = c
        if k in seen:
            seen[k] += 1
            out.append(f"{k}.{seen[k]}")
        else:
            seen[k] = 0
            out.append(k)
    return out

def scrape_team_adv_game_log(team, season, playoffs=False):
    """
    Scrape advanced team game logs from Basketball-Reference for a given team + season.

    This function is intentionally defensive:
    - Requests use timeouts so the scraper cannot hang indefinitely on a slow/stuck response.
    - A small retry loop handles transient network/HTTP issues.
    - If the expected table is missing, we raise a clear error (handled by callers).

    :param team: 3-letter NBA team abbreviation (e.g. "BOS").
    :param season: End year of season (e.g. 2026 for 2025-26).
    :param playoffs: True for playoff table, False for regular season table.
    :returns: DataFrame for the raw BRef table.
    """
    team_stats_url = (
        f"https://www.basketball-reference.com/teams/{team}/{season}/gamelog-advanced/"
    )
    selector = "team_game_log_adv_post" if playoffs else "team_game_log_adv_reg"

    # Identify as a normal browser; helps avoid occasional blocked/odd responses.
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Gambling-Project/1.0)"}

    # Keep the retry loop tiny; we already throttle below and callers can skip failures.
    max_tries = 3
    last_exc: Exception | None = None
    for attempt in range(1, max_tries + 1):
        try:
            # Hard timeout so we don't hang at a single team/season forever.
            # (connect timeout, read timeout)
            resp = requests.get(team_stats_url, headers=headers, timeout=(10, 30))
            resp.raise_for_status()

            team_soup = BeautifulSoup(resp.content, "lxml")
            table = team_soup.find(name="table", attrs={"id": selector})
            if table is None:
                raise ValueError("No tables found")

            df_list = pd.read_html(StringIO(str(table)))
            return df_list[0]
        except Exception as exc:  # noqa: BLE001 — network/HTML variance
            last_exc = exc
            if attempt < max_tries:
                # Short backoff for transient failures; still keep global request rate low.
                time.sleep(2.0 * attempt)
            else:
                break
        finally:
            # Can't exceed 20 requests/min → wait ~3.3s between attempts/requests.
            time.sleep(3.3)

    # If we got here, all retries failed.
    assert last_exc is not None
    raise last_exc

def clean_team_log(df, team):
    """
    Clean a raw Basketball-Reference advanced team game log table into a consistent schema.

    Notes:
    - BRef tables sometimes include repeated header rows and/or missing values (e.g., blank
      score cells for incomplete rows). This function coerces numeric columns defensively
      and uses pandas nullable dtypes to avoid hard failures during ingestion.
    - Column names may vary slightly across seasons/phases; score columns are detected via
      name heuristics with a positional fallback.
    """
    def _pick_first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
        """
        Return the first column name in ``candidates`` that exists in ``frame``, else None.

        :param frame: Dataframe whose columns will be searched.
        :param candidates: Ordered list of candidate column names.
        :return: First matching column name, or None if none exist.
        """
        for name in candidates:
            if name in frame.columns:
                return name
        return None

    def _detect_score_columns(frame: pd.DataFrame) -> tuple[str, str]:
        """
        Detect the team/opponent score columns from a BRef advanced game log table.

        The table includes an opponent abbreviation column named ``Opp`` (used for IDs),
        plus separate score columns (often ``Tm`` and a de-duped ``Opp.1`` after
        ``dedupe_columns``). This helper finds the likely score columns with a safe
        fallback to legacy positional indexing.

        :param frame: Cleaned dataframe after header-row removal and column de-duping.
        :return: (team_score_col, opp_score_col)
        """
        tm_col = _pick_first_existing_column(frame, ["Tm", "Tm.1", "PTS", "Tm_Score"])
        # Common after de-duping: ``Opp`` (abbr) + ``Opp.1`` (score).
        opp_col = _pick_first_existing_column(frame, ["Opp.1", "Opp PTS", "Opp.2", "Opp_Score"])

        # If detection fails, preserve prior behavior (positional) but do it explicitly.
        if tm_col is None or opp_col is None:
            tm_col = frame.columns[5]
            opp_col = frame.columns[6]
        return tm_col, opp_col

    # remove top-level of column multiindex hierarchy
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    
    # dedupe columns
    df.columns = dedupe_columns(df.columns)
     
    # get rid of extra 'column name' rows
    df = df[df["Rk"] != "Rk"].copy()  # copy() to avoid SettingWithCopyWarning
    df.reset_index(drop=True, inplace=True)

    #unnamed_3_level1 --> home/away col
    df.loc[:, "Unnamed: 3_level_1"] = df["Unnamed: 3_level_1"].fillna("Home")
    df.loc[:, "Unnamed: 3_level_1"] = df["Unnamed: 3_level_1"].replace({"@": "Away"})
    df.rename(columns={"Unnamed: 3_level_1": "Home"}, inplace=True)
    # match "home" col to SQL schema
    df["Home"] = df["Home"].map({"Home": True, "Away": False})
                       

    # Set OT to nullable boolean. BRef/pandas often reads this column as float64; assigning
    # boolean into that dtype triggers a pandas FutureWarning. Normalize via string first.
    df["OT"] = (
        df["OT"]
        .astype("string")
        .fillna("")
        .str.strip()
        .eq("OT")
        .astype("boolean")
    )

    # remove redundant column 'rk, rename 'Gtm' to Game
    df.drop(columns=["Rk"], inplace=True)
    df.rename(columns={"Gtm": "Game"}, inplace=True)

    # rename score columns and opponent stats columns
    tm_score_col, opp_score_col = _detect_score_columns(df)
    df.rename(
        columns={
            tm_score_col: "Tm_Score",
            opp_score_col: "Opp_Score",
        },
        inplace=True,
    )

    df.rename(columns={
        df.columns[-4]: "Opp_eFG%"
        ,df.columns[-3]: "Opp_TOV%"
        ,df.columns[-2]: "Opp_ORB%"
        ,df.columns[-1]: "Opp_FT/FGA"},
        inplace=True)

    # # create home margin col --> if home True then team pts - opp pts else opp pts - team pts
    # first ensure Tm and Opp cols are numeric
    # Use nullable integers so missing/blank score cells don't crash ingestion.
    df["Tm_Score"] = pd.to_numeric(df["Tm_Score"], errors="coerce").astype("Int64")
    df["Opp_Score"] = pd.to_numeric(df["Opp_Score"], errors="coerce").astype("Int64")

    # Vectorized margin computation; if either score is missing, result remains <NA>.
    home_margin = df["Tm_Score"] - df["Opp_Score"]
    away_margin = df["Opp_Score"] - df["Tm_Score"]
    df.loc[:, "Home_Margin"] = home_margin.where(df["Home"], away_margin).astype("Int64")

    # add team abbreviation column
    df.loc[:, "Team"] = team

    # create unique game ID col -- date-home-away
    # if home: date-team-opp else date-opp-team
    df.loc[:, "Game_ID"] = df.apply(
        lambda row: f"{row['Date']}-{team}-{row['Opp']}" if row["Home"] == True else f"{row['Date']}-{row['Opp']}-{team}",
        axis=1
    )

    # convert date & numeric columns to appropriate dtypes
    df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d')

    # rslt is reduntant with Home_Margin col
    df.drop(columns=["Rslt"], inplace=True)

    return df

def fetch_team_logs_for_date_window(
    team: str,
    season_years: list[int],
    min_date: date,
    max_date: date,
) -> pd.DataFrame:
    """
    Scrape advanced game logs for one team across one or more BRef seasons, clean them, and
    return only rows whose game date falls in ``[min_date, max_date]`` (inclusive).

    :param team: Three-letter team abbreviation (e.g. ``'BOS'``).
    :param season_years: Basketball-Reference season end years (URL year, e.g. ``2026`` for 2025-26).
    :param min_date: Lower bound on ``Date`` (game calendar date).
    :param max_date: Upper bound on ``Date``.
    :return: Cleaned dataframe (same columns as ``clean_team_log``), possibly empty.
    """
    frames: list[pd.DataFrame] = []
    for season in season_years:
        raw = scrape_team_adv_game_log(team, season)
        cleaned = clean_team_log(raw, team)
        day = cleaned["Date"].dt.date
        mask = (day >= min_date) & (day <= max_date)
        frames.append(cleaned.loc[mask])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
