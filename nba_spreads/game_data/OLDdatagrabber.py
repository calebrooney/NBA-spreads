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
def getPlayerID2(player_name, site): #site: bref, nba

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

