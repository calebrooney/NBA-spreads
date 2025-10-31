import requests
import pandas as pd

api_key ="e8c86bacda3f222976d61f089fff2848"
sports_key = "basketball_nba" #americanfootball_nfl"#'basketball_nba'
regions = 'us,us2'
market = 'spreads'
commenceTimeTo = pd.Timestamp.now().isoformat()


# url = f'https://api.the-odds-api.com/v4/sports/{sports_key}/odds/?apiKey={api_key}&regions={regions}&markets={market}'

# response = requests.get(url)
# data = response.json()
# print(data)

# ODDSdf = pd.json_normalize(data, sep='_')
# print(ODDSdf.sample(8))


# #save to csv
# ODDSdf.to_csv('NBAodds10-24-25.csv', index=False)
print(commenceTimeTo)
