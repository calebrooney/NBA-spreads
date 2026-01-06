# package
import requests
import pandas as pd
import os
import zoneinfo as tz
from dotenv import load_dotenv

def fetch_csv(
        date,
        live = True
        ):
    """
    Docstring for fetch_csv
    
    :param date: timestamp of date, formatted as 'YYYY-MM-DDT17:00:00Z' (string)
    :param live: default True, set to False for historical data (need valid paid API key for live = False)
    """
    load_dotenv()
    
    sports_key = "basketball_nba"
    regions = 'us,us2'
    market = 'spreads'

    # limit responses to only games from given date
    day = date.split("T")[0]
    commence_time_from = f"{day}T00:00:00Z"
    commence_time_to   = f"{day}T23:59:59Z"

    if live:
        api_key = os.getenv("ODDS_API_KEY_FREE")
        base = "https://api.the-odds-api.com/v4"
        link_params = (
            f"sports/{sports_key}/odds/"
            f"?apiKey={api_key}&regions={regions}&markets={market}"
            f"&commenceTimeFrom={commence_time_from}&commenceTimeTo={commence_time_to}"
        )
        url = f"{base}/{link_params}"
    # for historical data, extra date parameter required
    else:
        api_key = os.getenv("ODDS_API_KEY_PAID")
        base = "https://api.the-odds-api.com"
        link_params = (
            f"v4/historical/sports/{sports_key}/odds"
            f"?apiKey={api_key}&regions={regions}&markets={market}"
            f"&commenceTimeFrom={commence_time_from}&commenceTimeTo={commence_time_to}"
            f"&date={date}"
        )
        url = f"{base}/{link_params}"


    # load data
    response = requests.get(url)
    data = response.json()
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


date = "2026-01-06T17:00:00Z"
fetch_csv(date)