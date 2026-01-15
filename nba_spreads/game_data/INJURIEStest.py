from datetime import datetime
from nba_spreads.game_data.injuries import fetch_injury_report_df

dt = datetime(2025, 11, 24, 17, 30)
df = fetch_injury_report_df(dt)
print(df)
