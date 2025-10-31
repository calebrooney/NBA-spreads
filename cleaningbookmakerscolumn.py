import pandas as pd
from dateutil import parser

OYdf = pd.read_csv("NBAoddsTEST1.csv")



# # Replace this with your JSON data
# raw_data = [ 
#     # ... paste your entire JSON data here ...
# ]

# # Clean and normalize the data
# records = []
# for book in raw_data:
#     for market in book['markets']:
#         for outcome in market['outcomes']:
#             records.append({
#                 'Bookmaker': book['title'],
#                 'Team': outcome['name'],
#                 'Point Spread': outcome['point'],
#                 'Price': outcome['price'],
#                 'Last Update': parser.parse(book['last_update'])
#             })

# # Convert to DataFrame
# df = pd.DataFrame(records)

# # Pivot the DataFrame for better readability
# pivot_df = df.pivot(index='Bookmaker', columns='Team', values=['Point Spread', 'Price'])

# # Clean up index and column names
# pivot_df = pivot_df.sort_index()
# pivot_df.reset_index(inplace=True)
# pivot_df.columns.names = [None, None]

# # Display the cleaned table
# print(pivot_df)


print(OYdf['bookmakers'])
