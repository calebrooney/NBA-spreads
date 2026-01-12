# could possibly train model on game logs? this has margin data at least, so can use for benchmarking spreads

## possible features: "recent (last 10? last 20? idk yet)" game performances, 
## h2h if previous matchup exists, performance against common opponents (could be a bit trickier to calculate)
## "recent" or overall (aggregated) advanced metrics
## live injury data not in this dataset but could be worth pulling in
## create primarykey(date+hometeam+awayteam), join respective team game logs to create train/validation/test dataset

# https://www.basketball-reference.com/teams/BOS/2026/gamelog-advanced/


# from brefscraper (for basketball reference data)
import requests
from bs4 import BeautifulSoup
import pandas as pd

## example link: AD basktball reference 21-22 regular season
## "https://www.basketball-reference.com/players/d/davisan02/gamelog-advanced/2022/"

def getPlayerID(player_name): #returns basketball reference ID for given player

    df = pd.read_csv("Player_IDs_update_2023-09-14.csv")
    bref_id_List = df['bref_id'].tolist()
    bref_IDs = {k:v for k,v in zip(df['name'], bref_id_List)} # zip function to create dictionary of player names and IDs

    player_name = player_name.title() #allows for case insensitivity (except for names like 'DeMar DeRozan' or 'LeBron James')
    name_list = list(bref_IDs.keys())

    if player_name not in name_list:
        print(f"{player_name} not found. Please check the spelling of the player's name and capitalize all appropriate lettters.")
        return
    return bref_IDs[player_name]

def player_advBoxScore(player: str, season: int, saveJSON=False): #returns DF of advanced box scores for player in given season
    try:
        playerID = getPlayerID(player)
        if playerID is None:
            print("Player not found, try again.")
            return None

        playerStats_url = f"https://www.basketball-reference.com/players/d/{playerID}/gamelog-advanced/{season}/"
        playerRequest = requests.get(playerStats_url)
        playerSoup = BeautifulSoup(playerRequest.content, 'lxml')

        player_per_game = playerSoup.find(name='table', attrs={'id': 'pgl_advanced'})

        df_list = pd.read_html(str(player_per_game)) #creates list of len 1 dataframes
        df = df_list[0]

        ## cleaning df
        #get rid of extra column name rows
        df = df[df["Rk"] != "Rk"].copy()  # copy() to avoid SettingWithCopyWarning
        df.reset_index(drop=True, inplace=True)

        #unnamed_5 --> home/away col
        df.loc[:, "Unnamed: 5"] = df["Unnamed: 5"].fillna("Home")
        df.loc[:, "Unnamed: 5"] = df["Unnamed: 5"].replace({"@": "Away"})
        df.rename(columns={"Unnamed: 5": "Home/Away"}, inplace=True)

        #rename and split unnamed_7 col
        df.rename(columns={"Unnamed: 7": "Result"}, inplace=True)
        result_margin_split = df["Result"].str.split("(", expand=True)
        df.loc[:, "Result"] = result_margin_split[0].str.strip()
        df.loc[:, "Margin"] = result_margin_split[1].str.replace(")", "", regex=True).str.replace("+", "", regex=True).str.strip()

        # Move the Margin column to the desired position
        col = df.pop('Margin')
        df.insert(df.columns.get_loc("Result") + 1, col.name, col)
        
        #remove age column
        df.drop(columns = ["Age"], inplace= True)

        #set columns to appropriate data types
        float_columns = df.columns[-17:]
        df[float_columns] = df[float_columns].apply(pd.to_numeric, errors='coerce').astype('float64')

        df[df.columns[0]] = df[df.columns[0]].apply(pd.to_numeric, errors='coerce').astype('int64')

        str_columns = df.columns[3:7]
        df[str_columns] = df[str_columns].astype('str')

        df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d')


        if saveJSON:
            df.to_json(f"{player}_{season}_advBoxScore.json", orient='records',indent=2)

        return df

    except ImportError: ## 'html5lib not found' if player not in IDs csv
        print("html5lib not found. Please install it.")
        return None


## view all columns
# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)

## Example usage
# AC22 = player_advBoxScore("Alex Caruso", 2022, saveJSON=True)

# print(AC22.head())
# print(AC22.tail())
# print(AC22.dtypes) 

# thanks to Gabriel Cano 
# https://medium.com/analytics-vidhya/web-scraping-nba-data-with-pandas-beautifulsoup-and-regex-pt-1-e3d73679950a

## from nbascraper (NBA.com scraper):
teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
TeamIDs = {'ATL':1610612737, 'BOS':1610612738,'BRK':1610612751, 'CHI':1610612741, 'CHO':1610612766, 'CLE':1610612739, 'DAL':1610612742, 'DEN':1610612743, 'DET':1610612765, 'GSW':1610612744, 'HOU':1610612745, 'IND':1610612754, 'LAC':1610612746, 'LAL':1610612747, 'MEM':1610612763, 'MIA':1610612748, 'MIL':1610612749, 'MIN':1610612750, 'NOP':1610612740, 'NYK':1610612752, 'OKC':1610612760, 'ORL':1610612753, 'PHI':1610612755, 'PHO':1610612756, 'POR':1610612757, 'SAC':1610612758, 'SAS':1610612759, 'TOR':1610612761, 'UTA':1610612762, 'WAS':1610612764}
seasonTypes = ["Regular+Season", "Playoffs","PlayIn","Pre+Season","IST"]


def importBoxScores(team, season, seasonType = "Regular+Season"):
    
    if type(season) != str or len(season) != 7:
        print("Invalid season format. Please use the format 'YYYY-YY'")
        return

    OppTeamID = TeamIDs[team]
    url = f"https://www.nba.com/stats/teams/boxscores-traditional?OpponentTeamID={str(OppTeamID)}&Season={season}&SeasonType={seasonType}"
    
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    tables = soup.find_all("table")
    df = pd.read_html(str(tables))
    print(url)
    print(df)

#importBoxScores('BOS', '2022-23')
#current output : bullshit calendar

# get NBA.com player ID for given player
# playerIDS found thanks to Wfordh on github https://github.com/wfordh/ottobasket_values/blob/main/data/mappings_update_2023-09-14.csv

def getPlayerID(player_name):

    df = pd.read_csv("Player_IDs_update_2023-09-14.csv")
    NBA_ID_List =df['nba_player_id'].tolist()
    NBA_ID_List = [str(x)[:-2] for x in NBA_ID_List]
    nba_IDs = {k:v for k,v in zip(df['name'], NBA_ID_List)} # zip function to create dictionary of player names and IDs

    player_name = player_name.title()
    if player_name not in nba_IDs.keys():
        print("Player not found. Please check the spelling of the player's name.")
        return
    return nba_IDs[player_name]

# print(getPlayerID("Victor Wembanyama"))

## functions to import data from NBA Stats API https://documenter.getpostman.com/view/24232555/2s93shzpR3#0b757468-b123-4d74-9513-d2f19f4f6c30

import pandas as pd

## set year and stop to get JSON of advanced stats for all players in a season from seasons (year-stop)
## 1993 is earliest year available, 2025 is latest year available

def importADVbySeas(year, stop):

    url = "http://rest.nbaapi.com/api/PlayerDataAdvanced/season/" + str(year)

    df = pd.read_json(url) 
    frames = [df]
    start = year

    while year < stop:
        year += 1
        url = "http://rest.nbaapi.com/api/PlayerDataAdvanced/season/" + str(year)
        df1 = pd.read_json(url)
        frames.append(df1) 

    # ignore_index=True ensures final DF has continous, unique indices
    bigDF = pd.concat(frames, ignore_index=True)

    # Save the DataFrame to a JSON file
    output_file = f"ADVPlayerData_{start}-{stop}.json"
    bigDF.to_json(output_file, orient='records')


    print(bigDF)
    print(f"Data saved to {output_file}")

## set year and stop to create JSON of player totals (playoffs) for all players in a season from seasons (year-stop)

def importTOTALSbySeasYoffs(year, stop):

    url = "http://rest.nbaapi.com/api/PlayerDataTotalsPlayoffs/season/" + str(year)

    df = pd.read_json(url) 
    frames = [df]
    start = year

    while year < stop:
        year += 1
        url = "http://rest.nbaapi.com/api/PlayerDataTotalsPlayoffs/season/" + str(year)
        df1 = pd.read_json(url)
        frames.append(df1) 

    bigDF = pd.concat(frames, ignore_index=True)

    # Save the DataFrame to a JSON file
    output_file = f"PlayoffPlayerDataTotals_{start}-{stop}.json"
    bigDF.to_json(output_file, orient='records')


    print(bigDF)
    print(f"Data saved to {output_file}")

## set year and stop to create JSON of player totals (regular season) for all players in a season from seasons (year-stop)

def importTOTALSbySeas(year, stop):

    url = "http://rest.nbaapi.com/api/PlayerDataTotals/season/" + str(year)

    df = pd.read_json(url) 
    frames = [df]
    start = year

    while year < stop:
        year += 1
        url = "http://rest.nbaapi.com/api/PlayerDataTotals/season/" + str(year)
        df1 = pd.read_json(url)
        frames.append(df1) 

    bigDF = pd.concat(frames, ignore_index=True)

    # Save the DataFrame to a JSON file
    output_file = f"PlayerDataTotals_{start}-{stop}.json"
    bigDF.to_json(output_file, orient='records')

    print(bigDF)
    print(f"Data saved to {output_file}")

# importADVbySeas(1993, 2025)
# importTOTALSbySeasYoffs(1993, 2024)

## get advanced stats far all players by given team, for all teams

def importADVAllTeams():
    teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

    frames =[]
    frames += [pd.read_json(f"http://rest.nbaapi.com/api/PlayerDataAdvanced/team/{team}") for team in teams]
    bigDF = pd.concat(frames, ignore_index=True)

    # Save the DataFrame to a JSON file
    output_file = f"ADVPlayerDataByTeam.json"
    bigDF.to_json(output_file, orient='records')

    print(bigDF)
    print(f"Data saved to {output_file}")

## get advanced stats far all players by given team (provide three-letter team abbreviation)
## can update to take list of teams if wanted

def importADVbyTeam(team):

    url = f"http://rest.nbaapi.com/api/PlayerDataAdvanced/team/{team}"
    df = pd.read_json(url)
    output_file = f"ADVPlayerData_{team}.json"
    df.to_json(output_file, orient='records')
    print(df)
    print(f"Data saved to {output_file}")

    return df

# importADVbyTeam('BOS')

###scraping advanced box scores
#nba part needs work
def getPlayerID(player_name, site): #site: bref, nba

    #pd.read_csv("Player_IDs_update_2023-09-14.csv").columns
    #returns index(str) col list
    
    df = pd.read_csv("Player_IDs_update_2023-09-14.csv")

    sites = ["bref", "NBA"]
    if site not in sites:
        return  

    site_id_List =df[f'{site}_id'].tolist()

    site_id_List = [str(x) for x in site_id_List]
    site_IDs = {k:v for k,v in zip(df['name'], site_id_List)} # zip function to create dictionary of player names and IDs
    player_name = player_name.title()
    if player_name not in site_IDs.keys():
        print("Player not found. Please check the spelling of the player's name.")
        return
    return site_IDs[player_name]
    