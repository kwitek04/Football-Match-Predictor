import pandas as pd

RAW_DATA_PATH = 'premier_league_data.csv'
PREPARED_DATA_PATH = 'premier_league_prepared.csv'

MAX_MATCHES_TO_CHECK = 5
MAX_H2H_MATCHES = 5
MIN_PRIOR_MATCHES = 5
REFEREE_FORM_WINDOW = 10

# Elo rating system constants
INITIAL_ELO = 1500  # Starting points for every new team
ELO_K = 20          # Ratings change after match
ELO_HOME_ADVANTAGE = 100 # Bonus points added to the home team

# All columns that will be used by the ML model
FORM_FEATURES = [
    'Home_Pts_Last5', 'Away_Pts_Last5',
    'Home_GS_Last5', 'Home_GC_Last5',
    'Away_GS_Last5', 'Away_GC_Last5',
    'Home_GD_Last5', 'Away_GD_Last5',
    'Home_Home_Pts_Last5', 'Home_Home_GS_Last5', 'Home_Home_GC_Last5',
    'Away_Away_Pts_Last5', 'Away_Away_GS_Last5', 'Away_Away_GC_Last5',
    'Pts_Diff', 'GD_Diff', 'Home_Venue_Pts_Diff', 'Venue_GD_Diff',
    'Home_Shots_Last5', 'Away_Shots_Last5',
    'Home_ShotsTarget_Last5', 'Away_ShotsTarget_Last5',
    'Home_Corners_Last5', 'Away_Corners_Last5',
    'Home_Fouls_Last5', 'Away_Fouls_Last5',
    'Home_Yellow_Last5', 'Away_Yellow_Last5',
    'Home_Red_Last5', 'Away_Red_Last5',
    'Home_Shots_Against_Last5', 'Away_Shots_Against_Last5',
    'Home_ShotsTarget_Against_Last5', 'Away_ShotsTarget_Against_Last5',
    'Home_Corners_Against_Last5', 'Away_Corners_Against_Last5',
    'Home_Fouls_Against_Last5', 'Away_Fouls_Against_Last5',
    'Home_Yellow_Against_Last5', 'Away_Yellow_Against_Last5',
    'Home_Red_Against_Last5', 'Away_Red_Against_Last5',
    'Home_Home_Shots_Last5', 'Away_Away_Shots_Last5',
    'Home_Home_ShotsTarget_Last5', 'Away_Away_ShotsTarget_Last5',
    'Home_Home_Corners_Last5', 'Away_Away_Corners_Last5',
    'Home_Home_Fouls_Last5', 'Away_Away_Fouls_Last5',
    'Home_Home_Yellow_Last5', 'Away_Away_Yellow_Last5',
    'Home_Home_Red_Last5', 'Away_Away_Red_Last5',
    'Shots_Diff', 'ShotsTarget_Diff', 'Corners_Diff',
    'Fouls_Diff', 'Yellow_Diff', 'Red_Diff',
    'Home_Elo', 'Away_Elo', 'Elo_Diff',
    'H2H_Prior_Matches', 'H2H_Home_Pts_Last5', 'H2H_Away_Pts_Last5',
    'H2H_Home_GS_Last5', 'H2H_Away_GS_Last5', 'H2H_Pts_Diff', 'H2H_GD_Diff',
    'Referee_Prior_Matches', 'Referee_Avg_Goals_Last10',
    'Referee_Avg_Yellows_Last10', 'Referee_Avg_Reds_Last10',
    'Referee_Home_Win_Rate_Last10',
]


def load_raw_data(path=RAW_DATA_PATH):
    """Loads the CSV file and sorts it from oldest to newest match"""
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='ISO8601', errors='coerce')
    df = df.sort_values(by=['Date', 'Time'], na_position='last').reset_index(drop=True)
    return df


def new_team_history():
    """Creates empty lists for a team that appears for the first time"""
    return {
        'pts': [], 'gs': [], 'gc': [],
        'home_pts': [], 'home_gs': [], 'home_gc': [],
        'away_pts': [], 'away_gs': [], 'away_gc': [],
        'shots_for': [], 'shots_against': [],
        'shots_target_for': [], 'shots_target_against': [],
        'corners_for': [], 'corners_against': [],
        'fouls_for': [], 'fouls_against': [],
        'yellow_for': [], 'yellow_against': [],
        'red_for': [], 'red_against': [],
        'home_shots_for': [], 'home_shots_target_for': [],
        'home_corners_for': [], 'home_fouls_for': [],
        'home_yellow_for': [], 'home_red_for': [],
        'away_shots_for': [], 'away_shots_target_for': [],
        'away_corners_for': [], 'away_fouls_for': [],
        'away_yellow_for': [], 'away_red_for': [],
    }


def sum_last(values, max_matches):
    """Calculates the total sum of a specific stat from the last N matches"""
    if len(values) == 0:
        return 0
    return sum(values[-max_matches:])


def avg_last(values, max_matches):
    """Calculates the average of a specific stat from the last N matches"""
    if len(values) == 0:
        return 0
    recent = values[-max_matches:]
    return sum(recent) / len(recent)


def get_match_points(home_goals, away_goals):
    """3 pts for a win, 1 for a draw, 0 for a loss"""
    if home_goals > away_goals:
        return 3, 0
    if home_goals == away_goals:
        return 1, 1
    return 0, 3


def get_h2h_key(team_a, team_b):
    """Creates a unique dictionary key for two teams to track their head-to-head history"""
    if team_a < team_b:
        return team_a, team_b
    return team_b, team_a


def add_form_features(df, max_matches=MAX_MATCHES_TO_CHECK):
    """
    Engine of the script. It walks through every match chronologically,
    calculates stats based on past data saves them, and then updates the history 
    with the current match's result.
    """
    team_histories = {}
    elo_ratings = {}
    h2h_history = {}
    referee_history = {}
    feature_rows = []

    for _, row in df.iterrows():
        home = row['Home']
        away = row['Away']

        # Initialize tracking for teams making their first appearance in the dataset
        if home not in team_histories:
            team_histories[home] = new_team_history()
        if away not in team_histories:
            team_histories[away] = new_team_history()

        home_hist = team_histories[home]
        away_hist = team_histories[away]

        # Calculate form features based on historical data up to this match
        # Points and goals
        home_pts = sum_last(home_hist['pts'], max_matches)
        away_pts = sum_last(away_hist['pts'], max_matches)
        home_gs = sum_last(home_hist['gs'], max_matches)
        home_gc = sum_last(home_hist['gc'], max_matches)
        away_gs = sum_last(away_hist['gs'], max_matches)
        away_gc = sum_last(away_hist['gc'], max_matches)

        home_gd = home_gs - home_gc
        away_gd = away_gs - away_gc

        # Points and goals specifically for home team playing at home, and away team playing away
        home_home_pts = sum_last(home_hist['home_pts'], max_matches)
        home_home_gs = sum_last(home_hist['home_gs'], max_matches)
        home_home_gc = sum_last(home_hist['home_gc'], max_matches)
        away_away_pts = sum_last(away_hist['away_pts'], max_matches)
        away_away_gs = sum_last(away_hist['away_gs'], max_matches)
        away_away_gc = sum_last(away_hist['away_gc'], max_matches)

        # Advanced stats (shots, corners, fouls, cards)
        home_shots = sum_last(home_hist['shots_for'], max_matches)
        away_shots = sum_last(away_hist['shots_for'], max_matches)
        home_shots_target = sum_last(home_hist['shots_target_for'], max_matches)
        away_shots_target = sum_last(away_hist['shots_target_for'], max_matches)
        home_corners = sum_last(home_hist['corners_for'], max_matches)
        away_corners = sum_last(away_hist['corners_for'], max_matches)
        home_fouls = sum_last(home_hist['fouls_for'], max_matches)
        away_fouls = sum_last(away_hist['fouls_for'], max_matches)
        home_yellow = sum_last(home_hist['yellow_for'], max_matches)
        away_yellow = sum_last(away_hist['yellow_for'], max_matches)
        home_red = sum_last(home_hist['red_for'], max_matches)
        away_red = sum_last(away_hist['red_for'], max_matches)

        # Defensive advanced stats (shots allowed, corners allowed, cards against)
        home_shots_against = sum_last(home_hist['shots_against'], max_matches)
        away_shots_against = sum_last(away_hist['shots_against'], max_matches)
        home_shots_target_against = sum_last(home_hist['shots_target_against'], max_matches)
        away_shots_target_against = sum_last(away_hist['shots_target_against'], max_matches)
        home_corners_against = sum_last(home_hist['corners_against'], max_matches)
        away_corners_against = sum_last(away_hist['corners_against'], max_matches)
        home_fouls_against = sum_last(home_hist['fouls_against'], max_matches)
        away_fouls_against = sum_last(away_hist['fouls_against'], max_matches)
        home_yellow_against = sum_last(home_hist['yellow_against'], max_matches)
        away_yellow_against = sum_last(away_hist['yellow_against'], max_matches)
        home_red_against = sum_last(home_hist['red_against'], max_matches)
        away_red_against = sum_last(away_hist['red_against'], max_matches)

        # Advanced stats specifically for home team playing at home, and away team playing away
        home_home_shots = sum_last(home_hist['home_shots_for'], max_matches)
        away_away_shots = sum_last(away_hist['away_shots_for'], max_matches)
        home_home_shots_target = sum_last(home_hist['home_shots_target_for'], max_matches)
        away_away_shots_target = sum_last(away_hist['away_shots_target_for'], max_matches)
        home_home_corners = sum_last(home_hist['home_corners_for'], max_matches)
        away_away_corners = sum_last(away_hist['away_corners_for'], max_matches)
        home_home_fouls = sum_last(home_hist['home_fouls_for'], max_matches)
        away_away_fouls = sum_last(away_hist['away_fouls_for'], max_matches)
        home_home_yellow = sum_last(home_hist['home_yellow_for'], max_matches)
        away_away_yellow = sum_last(away_hist['away_yellow_for'], max_matches)
        home_home_red = sum_last(home_hist['home_red_for'], max_matches)
        away_away_red = sum_last(away_hist['away_red_for'], max_matches)

        # Elo system calculation
        home_elo = elo_ratings.get(home, INITIAL_ELO)
        away_elo = elo_ratings.get(away, INITIAL_ELO)

        # Head-to-Head stats
        h2h_key = get_h2h_key(home, away)
        past_h2h = h2h_history.get(h2h_key, [])
        recent_h2h = past_h2h[-MAX_H2H_MATCHES:]

        h2h_home_pts = 0
        h2h_away_pts = 0
        h2h_home_gs = 0
        h2h_away_gs = 0

        for past_match in recent_h2h:
            if past_match['home'] == home:
                h2h_home_pts += past_match['home_pts']
                h2h_away_pts += past_match['away_pts']
                h2h_home_gs += past_match['home_gs']
                h2h_away_gs += past_match['away_gs']
            else:
                h2h_home_pts += past_match['away_pts']
                h2h_away_pts += past_match['home_pts']
                h2h_home_gs += past_match['away_gs']
                h2h_away_gs += past_match['home_gs']

        # Referee stats
        referee = row.get('Referee')
        ref_history = referee_history.get(referee, [])
        ref_recent = ref_history[-REFEREE_FORM_WINDOW:]

        if len(ref_recent) == 0:
            ref_prior = 0
            ref_avg_goals = 0
            ref_avg_yellows = 0
            ref_avg_reds = 0
            ref_home_win_rate = 0
        else:
            ref_prior = len(ref_history)
            ref_avg_goals = avg_last([m['goals'] for m in ref_recent], REFEREE_FORM_WINDOW)
            ref_avg_yellows = avg_last([m['yellows'] for m in ref_recent], REFEREE_FORM_WINDOW)
            ref_avg_reds = avg_last([m['reds'] for m in ref_recent], REFEREE_FORM_WINDOW)
            ref_home_win_rate = avg_last(
                [m['home_win'] for m in ref_recent],
                REFEREE_FORM_WINDOW,
            )

        # Save calculated features
        feature_rows.append({
            'Home_Prior_Matches': len(home_hist['pts']),
            'Away_Prior_Matches': len(away_hist['pts']),
            'Home_Prior_Home_Matches': len(home_hist['home_pts']),
            'Away_Prior_Away_Matches': len(away_hist['away_pts']),
            'Home_Pts_Last5': home_pts,
            'Away_Pts_Last5': away_pts,
            'Home_GS_Last5': home_gs,
            'Home_GC_Last5': home_gc,
            'Away_GS_Last5': away_gs,
            'Away_GC_Last5': away_gc,
            'Home_GD_Last5': home_gd,
            'Away_GD_Last5': away_gd,
            'Home_Home_Pts_Last5': home_home_pts,
            'Home_Home_GS_Last5': home_home_gs,
            'Home_Home_GC_Last5': home_home_gc,
            'Away_Away_Pts_Last5': away_away_pts,
            'Away_Away_GS_Last5': away_away_gs,
            'Away_Away_GC_Last5': away_away_gc,
            'Pts_Diff': home_pts - away_pts,
            'GD_Diff': home_gd - away_gd,
            'Home_Venue_Pts_Diff': home_home_pts - away_away_pts,
            'Venue_GD_Diff': (home_home_gs - home_home_gc) - (away_away_gs - away_away_gc),
            'Home_Shots_Last5': home_shots,
            'Away_Shots_Last5': away_shots,
            'Home_ShotsTarget_Last5': home_shots_target,
            'Away_ShotsTarget_Last5': away_shots_target,
            'Home_Corners_Last5': home_corners,
            'Away_Corners_Last5': away_corners,
            'Home_Fouls_Last5': home_fouls,
            'Away_Fouls_Last5': away_fouls,
            'Home_Yellow_Last5': home_yellow,
            'Away_Yellow_Last5': away_yellow,
            'Home_Red_Last5': home_red,
            'Away_Red_Last5': away_red,
            'Home_Shots_Against_Last5': home_shots_against,
            'Away_Shots_Against_Last5': away_shots_against,
            'Home_ShotsTarget_Against_Last5': home_shots_target_against,
            'Away_ShotsTarget_Against_Last5': away_shots_target_against,
            'Home_Corners_Against_Last5': home_corners_against,
            'Away_Corners_Against_Last5': away_corners_against,
            'Home_Fouls_Against_Last5': home_fouls_against,
            'Away_Fouls_Against_Last5': away_fouls_against,
            'Home_Yellow_Against_Last5': home_yellow_against,
            'Away_Yellow_Against_Last5': away_yellow_against,
            'Home_Red_Against_Last5': home_red_against,
            'Away_Red_Against_Last5': away_red_against,
            'Home_Home_Shots_Last5': home_home_shots,
            'Away_Away_Shots_Last5': away_away_shots,
            'Home_Home_ShotsTarget_Last5': home_home_shots_target,
            'Away_Away_ShotsTarget_Last5': away_away_shots_target,
            'Home_Home_Corners_Last5': home_home_corners,
            'Away_Away_Corners_Last5': away_away_corners,
            'Home_Home_Fouls_Last5': home_home_fouls,
            'Away_Away_Fouls_Last5': away_away_fouls,
            'Home_Home_Yellow_Last5': home_home_yellow,
            'Away_Away_Yellow_Last5': away_away_yellow,
            'Home_Home_Red_Last5': home_home_red,
            'Away_Away_Red_Last5': away_away_red,
            'Shots_Diff': home_shots - away_shots,
            'ShotsTarget_Diff': home_shots_target - away_shots_target,
            'Corners_Diff': home_corners - away_corners,
            'Fouls_Diff': home_fouls - away_fouls,
            'Yellow_Diff': home_yellow - away_yellow,
            'Red_Diff': home_red - away_red,
            'Home_Elo': home_elo,
            'Away_Elo': away_elo,
            'Elo_Diff': home_elo - away_elo,
            'H2H_Prior_Matches': len(past_h2h),
            'H2H_Home_Pts_Last5': h2h_home_pts,
            'H2H_Away_Pts_Last5': h2h_away_pts,
            'H2H_Home_GS_Last5': h2h_home_gs,
            'H2H_Away_GS_Last5': h2h_away_gs,
            'H2H_Pts_Diff': h2h_home_pts - h2h_away_pts,
            'H2H_GD_Diff': h2h_home_gs - h2h_away_gs,
            'Referee_Prior_Matches': ref_prior,
            'Referee_Avg_Goals_Last10': ref_avg_goals,
            'Referee_Avg_Yellows_Last10': ref_avg_yellows,
            'Referee_Avg_Reds_Last10': ref_avg_reds,
            'Referee_Home_Win_Rate_Last10': ref_home_win_rate,
        })

        # Update histories with actual match result
        hg = row['HG'] if pd.notna(row['HG']) else 0
        ag = row['AG'] if pd.notna(row['AG']) else 0
        home_points, away_points = get_match_points(hg, ag)

        # Extract advanced match stats
        home_shots_match = row['Home_Shots'] if pd.notna(row['Home_Shots']) else 0
        away_shots_match = row['Away_Shots'] if pd.notna(row['Away_Shots']) else 0
        home_shots_target_match = row['Home_Shots_Target'] if pd.notna(row['Home_Shots_Target']) else 0
        away_shots_target_match = row['Away_Shots_Target'] if pd.notna(row['Away_Shots_Target']) else 0
        home_corners_match = row['Home_Corners'] if pd.notna(row['Home_Corners']) else 0
        away_corners_match = row['Away_Corners'] if pd.notna(row['Away_Corners']) else 0
        home_fouls_match = row['Home_Fouls'] if pd.notna(row['Home_Fouls']) else 0
        away_fouls_match = row['Away_Fouls'] if pd.notna(row['Away_Fouls']) else 0
        home_yellow_match = row['Home_Yellow_Cards'] if pd.notna(row['Home_Yellow_Cards']) else 0
        away_yellow_match = row['Away_Yellow_Cards'] if pd.notna(row['Away_Yellow_Cards']) else 0
        home_red_match = row['Home_Red_Cards'] if pd.notna(row['Home_Red_Cards']) else 0
        away_red_match = row['Away_Red_Cards'] if pd.notna(row['Away_Red_Cards']) else 0

        # Push data into the home team historical record
        home_hist['pts'].append(home_points)
        home_hist['gs'].append(hg)
        home_hist['gc'].append(ag)
        home_hist['home_pts'].append(home_points)
        home_hist['home_gs'].append(hg)
        home_hist['home_gc'].append(ag)
        home_hist['shots_for'].append(home_shots_match)
        home_hist['shots_against'].append(away_shots_match)
        home_hist['shots_target_for'].append(home_shots_target_match)
        home_hist['shots_target_against'].append(away_shots_target_match)
        home_hist['corners_for'].append(home_corners_match)
        home_hist['corners_against'].append(away_corners_match)
        home_hist['fouls_for'].append(home_fouls_match)
        home_hist['fouls_against'].append(away_fouls_match)
        home_hist['yellow_for'].append(home_yellow_match)
        home_hist['yellow_against'].append(away_yellow_match)
        home_hist['red_for'].append(home_red_match)
        home_hist['red_against'].append(away_red_match)
        home_hist['home_shots_for'].append(home_shots_match)
        home_hist['home_shots_target_for'].append(home_shots_target_match)
        home_hist['home_corners_for'].append(home_corners_match)
        home_hist['home_fouls_for'].append(home_fouls_match)
        home_hist['home_yellow_for'].append(home_yellow_match)
        home_hist['home_red_for'].append(home_red_match)

        # Push data into the away team historical record
        away_hist['pts'].append(away_points)
        away_hist['gs'].append(ag)
        away_hist['gc'].append(hg)
        away_hist['away_pts'].append(away_points)
        away_hist['away_gs'].append(ag)
        away_hist['away_gc'].append(hg)
        away_hist['shots_for'].append(away_shots_match)
        away_hist['shots_against'].append(home_shots_match)
        away_hist['shots_target_for'].append(away_shots_target_match)
        away_hist['shots_target_against'].append(home_shots_target_match)
        away_hist['corners_for'].append(away_corners_match)
        away_hist['corners_against'].append(home_corners_match)
        away_hist['fouls_for'].append(away_fouls_match)
        away_hist['fouls_against'].append(home_fouls_match)
        away_hist['yellow_for'].append(away_yellow_match)
        away_hist['yellow_against'].append(home_yellow_match)
        away_hist['red_for'].append(away_red_match)
        away_hist['red_against'].append(home_red_match)
        away_hist['away_shots_for'].append(away_shots_match)
        away_hist['away_shots_target_for'].append(away_shots_target_match)
        away_hist['away_corners_for'].append(away_corners_match)
        away_hist['away_fouls_for'].append(away_fouls_match)
        away_hist['away_yellow_for'].append(away_yellow_match)
        away_hist['away_red_for'].append(away_red_match)

        # Update Elo Ratings
        expected_home = 1 / (1 + 10 ** ((away_elo - (home_elo + ELO_HOME_ADVANTAGE)) / 400))
        expected_away = 1 - expected_home

        if hg > ag:
            score_home, score_away = 1.0, 0.0
        elif hg == ag:
            score_home, score_away = 0.5, 0.5
        else:
            score_home, score_away = 0.0, 1.0

        elo_ratings[home] = home_elo + ELO_K * (score_home - expected_home)
        elo_ratings[away] = away_elo + ELO_K * (score_away - expected_away)

        # Update Head-to-Head tracker
        if h2h_key not in h2h_history:
            h2h_history[h2h_key] = []
        h2h_history[h2h_key].append({
            'home': home,
            'away': away,
            'home_pts': home_points,
            'away_pts': away_points,
            'home_gs': hg,
            'away_gs': ag,
        })

        # Update referee statistics tracker
        if pd.notna(referee):
            if referee not in referee_history:
                referee_history[referee] = []
            referee_history[referee].append({
                'goals': hg + ag,
                'yellows': home_yellow_match + away_yellow_match,
                'reds': home_red_match + away_red_match,
                'home_win': 1 if hg > ag else 0,
            })

    # Combine original data with newly calculated features and return it
    return pd.concat([df.copy(), pd.DataFrame(feature_rows)], axis=1)


def prepare_dataset(path=RAW_DATA_PATH):
    """Convenience function to load and process the data in one step."""
    df = load_raw_data(path)
    return add_form_features(df)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    df = prepare_dataset()
    df.to_csv(PREPARED_DATA_PATH, index=False)

    columns_to_show = [
        'Season', 'Date', 'Home', 'Away', 'HG', 'AG', 'Res',
        'Home_Pts_Last5', 'Away_Pts_Last5',
        'Home_Elo', 'Away_Elo', 'Shots_Diff',
    ]
    print(df[columns_to_show].tail(10))
    print(f'\nFeature count: {len(FORM_FEATURES)}')
    print(f'Saved prepared dataset to {PREPARED_DATA_PATH}')