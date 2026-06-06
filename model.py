import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_processing import FORM_FEATURES, MIN_PRIOR_MATCHES

# Convert the match results into a numerical classification target
TARGET_MAPPING = {'H': 1, 'D': 0, 'A': 2}
TARGET_NAMES = ['Draw (0)', 'Home Win (1)', 'Away Win (2)']

# Reverse mapping to convert model numbers back letters
REVERSE_TARGET_MAPPING = {1: 'H', 0: 'D', 2: 'A'}

TEST_SEASON = '2025/26'


def build_ml_dataset(df):
    """
    Cleans the raw dataset and prepares the final X (features) and y (target) variables.
    It removes matches that don't have enough historical context.
    """
    # Select only the columns we explicitly want the model to see
    columns_for_ml = FORM_FEATURES + ['Res', 'Season', 'Home_Prior_Matches', 'Away_Prior_Matches', 'Date', 'Home', 'Away']
    df_ml = df[columns_for_ml].copy()
    
    # Drop rows with missing values
    df_ml = df_ml.dropna().reset_index(drop=True)

    total_matches = len(df_ml)

    # Filter out early-season matches.
    df_ml = df_ml[
        (df_ml['Home_Prior_Matches'] >= MIN_PRIOR_MATCHES)
        & (df_ml['Away_Prior_Matches'] >= MIN_PRIOR_MATCHES)
    ].reset_index(drop=True)

    removed_matches = total_matches - len(df_ml)
    
    # Translate letters (H, D, A) into target numbers (1, 0, 2)
    df_ml['Target'] = df_ml['Res'].map(TARGET_MAPPING)

    # X represents inputs
    X = df_ml[FORM_FEATURES]
    # y represents target
    y = df_ml['Target']
    
    seasons = df_ml['Season']

    # Keeps the readable text data synchronized with X and y
    meta = df_ml[['Date', 'Home', 'Away', 'Res']]
    
    return X, y, seasons, removed_matches, total_matches, meta


def split_by_season(X, y, seasons, meta):
    """
    Separates the data chronologically.
    """
    train_mask = seasons < TEST_SEASON
    test_mask = seasons == TEST_SEASON
    
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask], seasons[train_mask], seasons[test_mask], meta[train_mask], meta[test_mask]


def print_baseline_accuracy(y_test):
    """
    Calculates the accuracy of home team winning every single match. 
    The ML model should beat this percentage.
    """
    home_baseline = (y_test == 1).mean()
    print('Baseline metrics:')
    print(f'Always predict home win: {home_baseline * 100:.2f}%\n')


def main():
    df = pd.read_csv('premier_league_prepared.csv')
    
    # Clean the data and extract X and y
    X, y, seasons, removed_matches, total_matches, meta = build_ml_dataset(df)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test, train_seasons, test_seasons, meta_train, meta_test = split_by_season(X, y, seasons, meta)

    print('Dataset summary:')
    print(f'Total matches with results: {total_matches}')
    print(f'Removed early matches (< {MIN_PRIOR_MATCHES} prior games): {removed_matches}')
    print(f'Matches used for ML: {len(X)}\n')

    print('Dataset split summary:')
    print(f'Train seasons: {sorted(train_seasons.unique())}')
    print(f'Test season: {sorted(test_seasons.unique())}')
    print(f'Training on {len(X_train)} matches')
    print(f'Testing on {len(X_test)} matches\n')
    
    # Initialize the model with specific parameters to prevent overfitting
    model = RandomForestClassifier(
        n_estimators=400,        # Number of decision trees in the forest
        min_samples_split=10,    # Minimum data points required to split a node
        max_depth=20,            # Limits how deep the trees can grow
        max_features='sqrt',     # Number of features to consider when looking for the best split
        random_state=42,         # Ensures reproducible results on every run
        n_jobs=-1,               # Uses all available CPU cores to speed up training
    )
    
    # Training the model to find patterns in the historical data
    model.fit(X_train, y_train)

    # Model predicts the outcomes of the test season
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print('Final results:')
    print(f'Model accuracy: {accuracy * 100:.2f}%\n')
    
    print_baseline_accuracy(y_test)
    
    print('Detailed classification report:')
    print(classification_report(y_test, predictions, target_names=TARGET_NAMES, zero_division=0))

    # Feature importance shows which stats the model found most useful for predicting wins
    print('Top 15 most important features:')
    ranked_features = sorted(
        zip(FORM_FEATURES, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    
    for feature_name, importance in ranked_features[:15]:
        print(f'  {feature_name}: {importance:.3f}')

    print('\nPredictions vs actual results for last 20 matches:')
    
    prediction_analysis = meta_test.copy()
    
    # Convert model prediction numbers back to letters
    prediction_analysis['Predicted'] = [REVERSE_TARGET_MAPPING[p] for p in predictions]
    
    # Check if the model guessed correctly
    prediction_analysis['Correct'] = prediction_analysis['Res'] == prediction_analysis['Predicted']

    display_columns = ['Date', 'Home', 'Away', 'Predicted', 'Res', 'Correct']
    print(prediction_analysis[display_columns].tail(20).to_string(index=False))


if __name__ == '__main__':
    main()