# used to get spreads of the day

import requests
import pandas as pd
import zoneinfo as tz
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ODDS_API_KEY_FREE")
sports_key = "basketball_nba"
regions = 'us,us2'
market = 'spreads'
commenceTimeTo = pd.Timestamp.now(tz=tz.ZoneInfo('US/Pacific')).date()
today = str(commenceTimeTo)

# url = f'https://api.the-odds-api.com/v4/sports/{sports_key}/odds/?apiKey={api_key}&regions={regions}&markets={market}'

# response = requests.get(url)
# data = response.json()

# def fetch_csv(
#         commenceTimeTo = pd.Timestamp.now(tz=tz.ZoneInfo('US/Pacific')).date()
#         ,api_key = os.getenv("ODDS_API_KEY_FREE")
#         ):
#     load_dotenv()
    
#     sports_key = "basketball_nba"
#     regions = 'us,us2'
#     market = 'spreads'

#     return 

# ODDSdf = pd.json_normalize(data, sep='_')
# print(ODDSdf.sample(5))
# print(today)

# #save to csv
# ODDSdf.to_csv(f'NBAodds{today}.csv', index=False)
print(commenceTimeTo)

#### refactor this file to call fetch_csv function from nba_spreads package