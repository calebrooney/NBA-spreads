## historical odds data has dupe rows, ie positive and negative for each team involved

## game margin column in BRef game logs will be standardized to (home team - away team), so can remove all spreads that give score from away team pov

teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
teams_dict = {'ATL':'Atlanta Hawks', 'BOS':'Boston Celtics', 'BRK':'Brooklyn Nets', 'CHI':'Chicago Bulls', 'CHO':'Charlotte Hornets', 'CLE':'Cleveland Cavaliers', 'DAL':'Dallas Mavericks', 'DEN':'Denver Nuggets', 'DET':'Detroit Pistons', 'GSW':'Golden State Warriors', 'HOU':'Houston Rockets', 'IND':'Indiana Pacers', 'LAC':'LA Clippers', 'LAL':'Los Angeles Lakers', 'MEM':'Memphis Grizzlies', 'MIA':'Miami Heat', 'MIL':'Milwaukee Bucks', 'MIN':'Minnesota Timberwolves', 'NOP':'New Orleans Pelicans', 'NYK':'New York Knicks', 'OKC':'Oklahoma City Thunder', 'ORL':'Orlando Magic', 'PHI':'Philadelphia 76ers', 'PHO':'Phoenix Suns', 'POR':'Portland Trail Blazers', 'SAC':'Sacramento Kings', 'SAS':'San Antonio Spurs', 'TOR':'Toronto Raptors', 'UTA':'Utah Jazz', 'WAS':'Washington Wizards'}

# for odds data:
# create game_id col: 
# if 'home_team' != 'name' in row, then drop row
# drop previous and next timestamp columns

def odds_db_prep(df):
    # create game_id col
    df.loc[:, "Game_ID"] = df.apply(
        lambda row: f"{row['date']}-{row['home_team']}-{row['away_team']}" if row['home_team'] == row['name'] else None,
        axis=1
    )

    # drop rows where home_team != name
    df = df[df["Game_ID"].notna()]

    # drop previous and next timestamp columns
    df.drop(columns=["previous_timestamp", "next_timestamp"], inplace=True)

    return df

def game_logs_db_prep(df, team):
    # 3PAr --> _3PAr
    df.rename(columns={"3PAr": "_3PAr"}, inplace=True)