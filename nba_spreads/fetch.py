# package
import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import zoneinfo as tz
from zoneinfo import ZoneInfo
from dotenv import load_dotenv #access hidden keys

def fetch_csv(
        date,
        live = True
        ):
    """
    Docstring for fetch_csv
    
    :param date: timestamp of date, formatted as 'YYYY-MM-DD' (string)
    :param live: default True, set to False for historical data (need valid paid API key for live = False)
    """
    load_dotenv()
    
    sports_key = "basketball_nba"
    regions = 'us,us2'
    market = 'spreads'

    # # limit responses to only games from given date (use PST, = UTC-8)
    # timestamp_utc = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")

    # pacific_day_start = timestamp_utc.replace(hour=8, minute=0, second=0)
    # pacific_day_end   = pacific_day_start + timedelta(days=1) - timedelta(seconds=1)

    # commence_time_from = pacific_day_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    # commence_time_to   = pacific_day_end.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Parse the Pacific date
    game_date = datetime.strptime(date, "%Y-%m-%d").date()
    day = date  # for filenames/logging

    # Build Pacific day window (00:00:00 → 23:59:59) and convert to UTC
    pacific = ZoneInfo("America/Los_Angeles")
    window_start_local = datetime.combine(game_date, datetime.min.time(), tzinfo=pacific)
    window_end_local = window_start_local + timedelta(days=1) - timedelta(seconds=1)

    commence_time_from = window_start_local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    commence_time_to = window_end_local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Snapshot time: 9:00 AM Pacific on that date, converted to UTC (for historical endpoint)
    snapshot_local = window_start_local.replace(hour=9, minute=0, second=0)
    snapshot_utc = snapshot_local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")

    if live:
        api_key = os.getenv("ODDS_API_KEY_FREE")
        base = "https://api.the-odds-api.com/v4"
        link_params = (
            f"sports/{sports_key}/odds/"
            f"?apiKey={api_key}&regions={regions}&markets={market}"
            f"&commenceTimeFrom={commence_time_from}&commenceTimeTo={commence_time_to}"
        )
    # for historical data, extra date parameter required
    else:
        api_key = os.getenv("ODDS_API_KEY_PAID")
        base = "https://api.the-odds-api.com"
        link_params = (
            f"v4/historical/sports/{sports_key}/odds"
            f"?apiKey={api_key}&regions={regions}&markets={market}"
            f"&commenceTimeFrom={commence_time_from}&commenceTimeTo={commence_time_to}"
            f"&date={snapshot_utc}"
        )
        
    # load data
    url = f"{base}/{link_params}"
    response = requests.get(url)
    data = response.json()
    response.raise_for_status() #raise exception if data error occurs such as rate-limit quota hit, incorrect params, etc
    ODDSdf = pd.json_normalize(data, sep='_')
    if ODDSdf.empty:
        print(f"{day}: no games — nothing saved")
        return None

    # save to csv in data/raw
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
    os.makedirs(RAW_DIR, exist_ok=True)
    output_path = os.path.join(RAW_DIR, f"NBAodds_{day}.csv")
    ODDSdf.to_csv(output_path, index=False)


    # print(f"{day} data saved to data/raw")
    # return f"../nba_spreads/data/raw/NBAodds_{day}.csv"

    print(f"{day}: saved {len(ODDSdf)} rows to data/raw")
    return output_path


today = str(pd.Timestamp.now(tz=tz.ZoneInfo('US/Pacific')).date())
fetch_csv(today)