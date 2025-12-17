import pandas as pd
import ast

# designed to clean odds data from a given day, compatible with results from odds-api free version.
# daily dataframe is created by calling odds API in another script and saving to csv.

def safe_parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return None
    return x

def clean_odds_data(df):
    # prepare bookmakers column for explosion/normalization
    df['bookmakers'] = df['bookmakers'].apply(safe_parse)

    # 1) explode bookmakers (creates dictionary like data)
    df1 = df.explode('bookmakers').reset_index(drop=True)
    # flatten bookmaker dict
    bk = pd.json_normalize(df1['bookmakers'])
    bk = bk.rename(columns={
        'key': 'bookmaker_key',
        'title': 'bookmaker_title',
        'last_update': 'bookmaker_last_update'
    })
    df1 = pd.concat([df1.drop(columns='bookmakers'), bk], axis=1)

    # 2) explode markets
    df2 = df1.explode('markets').reset_index(drop=True)
    # flatten market dict
    mkt = pd.json_normalize(df2['markets'])
    mkt = mkt.rename(columns={
        'key': 'market_key',
        'last_update': 'market_last_update'
    })
    df2 = pd.concat([df2.drop(columns='markets'), mkt], axis=1)

    # 3) explode outcomes
    df3 = df2.explode('outcomes').reset_index(drop=True)
    # flatten outcome dict
    out = pd.json_normalize(df3['outcomes'])

    # save explosion/normalization operations as final df
    final = pd.concat([df3.drop(columns='outcomes'), out], axis=1)

    # drop unneccasry columns (they just indicate NBA data is being used)
    final = final.drop(columns=['sport_key', 'sport_title'])

    return final

