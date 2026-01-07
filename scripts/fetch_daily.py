# scripts/fetch_daily.py
# cron job runs at 6:30 am, 11 am, 3:30 pm (17:30) PT

from datetime import datetime
from zoneinfo import ZoneInfo

from nba_spreads.fetch import fetch_csv  # adjust to your module path

def main():
    pacific = ZoneInfo("America/Los_Angeles")
    today_pacific = datetime.now(pacific).date().isoformat()  # 'YYYY-MM-DD'
    fetch_csv(today_pacific)

if __name__ == "__main__":
    main()
