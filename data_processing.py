import pandas as pd
import numpy as np

# Pandas options for tables displaying in terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('polish_ekstraklasa_data.csv')

# Convert date from string to date format and sort 
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Date', 'Time']).reset_index(drop=True)

# Dictionaries to keep point and goals data from last games for each team
team_points = {}
team_goals_scored = {}
team_goals_conceded = {}

home_points_last5 = []
home_goals_scored_last5 = []
home_goals_conceded_last5 = []
away_points_last5 = []
away_goals_scored_last5 = []
away_goals_conceded_last5 = []

# Matches number we want to calculate team form from
MAX_MATCHES_TO_CHECK = 5

for index, row in df.iterrows():
    home = row['Home']
    away = row['Away']
    
    # if there is no team history
    if home not in team_points:
        team_points[home] = []
        team_goals_scored[home] = []
        team_goals_conceded[home] = []

    if away not in team_points:
        team_points[away] = []
        team_goals_scored[away] = []
        team_goals_conceded[away] = []

    # Calculating home team form
    home_pts = 0
    home_gs = 0
    home_gc = 0
    home_matches_played = len(team_points[home]) 

    if home_matches_played > 0:
        # to determine how many past matches to check, we take min value from MAX_MATCHES_TO_CHECK and home_matches_played
        matches_to_check = min(MAX_MATCHES_TO_CHECK, home_matches_played)

        start_idx = home_matches_played - matches_to_check

        for i in range(start_idx, home_matches_played):
            home_pts += team_points[home][i]
            home_gs += team_goals_scored[home][i]
            home_gc += team_goals_conceded[home][i]

    # Calculating away team form
    away_pts = 0
    away_gs = 0
    away_gc = 0
    away_matches_played = len(team_points[away])

    if away_matches_played > 0:
        matches_to_check = min(MAX_MATCHES_TO_CHECK, away_matches_played)
        start_idx = away_matches_played - matches_to_check
        
        for i in range(start_idx, away_matches_played):
            away_pts += team_points[away][i]
            away_gs += team_goals_scored[away][i]
            away_gc += team_goals_conceded[away][i]
            
    home_points_last5.append(home_pts)
    away_points_last5.append(away_pts)
    
    home_goals_scored_last5.append(home_gs)
    home_goals_conceded_last5.append(home_gc)
    
    away_goals_scored_last5.append(away_gs)
    away_goals_conceded_last5.append(away_gc)
    
    hg = row['HG']
    ag = row['AG']
    
    # saving home and away goals scored and conceded
    team_goals_scored[home].append(hg)
    team_goals_conceded[home].append(ag)
    
    team_goals_scored[away].append(ag)
    team_goals_conceded[away].append(hg)
    
    # adding points
    if hg > ag:
        team_points[home].append(3)
        team_points[away].append(0)
    elif hg == ag:
        team_points[home].append(1)
        team_points[away].append(1)
    else:
        team_points[home].append(0)
        team_points[away].append(3)

# Adding new columns
df['Home_Pts_Last5'] = home_points_last5
df['Away_Pts_Last5'] = away_points_last5
df['Home_GS_Last5'] = home_goals_scored_last5
df['Home_GC_Last5'] = home_goals_conceded_last5
df['Away_GS_Last5'] = away_goals_scored_last5
df['Away_GC_Last5'] = away_goals_conceded_last5


columns_to_show = ['Date', 'Home', 'Away', 'HG', 'AG', 'Home_Pts_Last5', 'Away_Pts_Last5', 'Home_GS_Last5', 'Home_GC_Last5', 'Away_GS_Last5', 'Away_GC_Last5']
print(df[columns_to_show].tail(10))