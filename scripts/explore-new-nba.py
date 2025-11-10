import pandas as pd

df = pd.read_csv('NBAodds10-24-25.csv')

print(df.columns)
print(df.sample(5))