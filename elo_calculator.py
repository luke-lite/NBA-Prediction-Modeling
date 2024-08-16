import pandas as pd

def calculate_historical_elos():
    nba_all_elo = pd.read_csv('https://raw.githubusercontent.com/luke-lite/NBA-Prediction-Modeling/main/data/nbaallelo.csv')
    