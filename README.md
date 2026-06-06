# Football Match Predictor

A Machine Learning project to predict English Premier League match results.

**Data Source:** Match history are sourced from [football-data.co.uk](https://www.football-data.co.uk).

Key engineered features include:
* **Dynamic Elo ratings:** Calculates real-time team strength
* **Form & advanced stats:** Tracks points, goals, shots on target, corners, and disciplinary records
* **Head-to-Head records:** Tracks the historical performance between specific teams
* **Referee trends:** Studying the assigned referee's recent tendencies regarding goals and cards.

### Machine Learning Model
The predictions are powered by a **Random Forest Classifier** built with `scikit-learn`. Model is trained entirely on past seasons and tested on current season.