import pandas as pd

df = pd.read_csv('NBAoddsTEST1.csv')

# print(df.head())
print(df.columns)

bookmakers = df['bookmakers'].apply(pd.Series)
print(bookmakers[5])

