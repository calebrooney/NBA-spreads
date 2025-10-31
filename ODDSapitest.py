import requests
import pandas as pd
import zoneinfo as tz

api_key ="e8c86bacda3f222976d61f089fff2848"
sports_key = "basketball_nba" #americanfootball_nfl"#'basketball_nba'
regions = 'us,us2'
market = 'spreads'
commenceTimeTo = pd.Timestamp.now(tz=tz.ZoneInfo('US/Pacific')).date()
today = str(commenceTimeTo)

url = f'https://api.the-odds-api.com/v4/sports/{sports_key}/odds/?apiKey={api_key}&regions={regions}&markets={market}'

response = requests.get(url)
data = response.json()


ODDSdf = pd.json_normalize(data, sep='_')
# print(ODDSdf.sample(5))
# print(today)

#save to csv
ODDSdf.to_csv(f'NBAodds{today}.csv', index=False)
print(commenceTimeTo)
