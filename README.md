# NBA-spreads
Attempting to predict the margin of a given NBA game well enough to exploit unusual spreads. 

Data:
- NBA games (compiled in struggling celtics)
- Live/historical spreads from various US bookmakers, via ODDSapi

- Hypothetical: could try to incorporate some sort of social media 

Process:
1. ETL historical spreads data
2. ETL nba box score data for matching time period
3. Merge datasets
4. Plot/analyze historical spreads vs actual game margins to develop a benchmark for the model (expected margin of error of sportsbooks compared to model's margin prediction error)
5. Train model
6. Build pipeline to check spreads daily/weekly, and identify in real time possible games
7. Simulate how arbitrary selections betting one unit each perform vs the spread.


Ideas to test:
- define some large markets (ie Boston, LA, NY, etc). Could also try nationally televised games
- I think books might adjust the line beyond what they think will actually happen to balance betting on both sides (especially in big markets), which could lead to exploitable lines
- instead of betting a constant unit size, perhaps unit size could be a function of models confidence
