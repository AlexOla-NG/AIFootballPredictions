"""
AI Football Predictions Script: Predicts both goals (Over 2.5) and corners (Over 10.5) for upcoming matches.

This script loads pre-trained machine learning models for both goal and corner predictions.

Example usage:
    python scripts/make_predictions.py --input_leagues_models_dir models --input_data_predict_dir data/processed --final_predictions_out_file final_predictions.txt --next_matches data/next_matches.json --predict_corners
"""

import pandas as pd
import os
import json
import pickle
import numpy as np
from datetime import datetime
import argparse

VALID_LEAGUES = ["E0", "I1", "D1", "SP1", "F1"]

HOME_TEAM_FEATURES = [
    'HomeTeam', 'FTHG', 'HG', 'HTHG', 'HS', 'HST', 'HHW', 'HC', 'HF', 'HFKC', 'HO', 'HY', 'HR', 'HBP',
    'B365H', 'BFH', 'BSH', 'BWH', 'GBH', 'IWH', 'LBH', 'PSH', 'SOH', 'SBH', 'SJH', 'SYH', 'VCH', 'WHH',
    'BbMxH', 'BbAvH', 'MaxH', 'AvgH', 'BFEH', 'BbMxAHH', 'BbAvAHH', 'GBAHH', 'LBAHH', 'B365AHH', 'PAHH',
    'MaxAHH', 'AvgAHH', 'BbAHh', 'AHh', 'GBAH', 'LBAH', 'B365AH', 'AvgHomeGoalsScored', 'AvgHomeGoalsConceded',
    'HomeOver2.5Perc', 'AvgLast5HomeGoalsScored', 'AvgLast5HomeGoalsConceded', 'Last5HomeOver2.5Count', 'Last5HomeOver2.5Perc',
    'AvgHomeCorners', 'AvgLast5HomeCorners', 'AvgLast5HomeCornersConceded', 'PredictedHomeCorners'
]

AWAY_TEAM_FEATURES = [
    'AwayTeam', 'FTAG', 'AG', 'HTAG', 'AS', 'AST', 'AHW', 'AC', 'AF', 'AFKC', 'AO', 'AY', 'AR', 'ABP',
    'B365A', 'BFA', 'BSA', 'BWA', 'GBA', 'IWA', 'LBA', 'PSA', 'SOA', 'SBA', 'SJA', 'SYA', 'VCA', 'WHA',
    'BbMxA', 'BbAvA', 'MaxA', 'AvgA', 'BFEA', 'BbMxAHA', 'BbAvAHA', 'GBAHA', 'LBAHA', 'B365AHA', 'PAHA',
    'MaxAHA', 'AvgAHA', 'AvgAwayGoalsScored', 'AvgAwayGoalsConceded', 'AwayOver2.5Perc', 'AvgLast5AwayGoalsScored',
    'AvgLast5AwayGoalsConceded', 'Last5AwayOver2.5Count', 'Last5AwayOver2.5Perc',
    'AvgAwayCorners', 'AvgLast5AwayCorners', 'AvgLast5AwayCornersConceded', 'PredictedAwayCorners'
]

def load_model(filepath: str):
    """Load a machine learning model from pickle file."""
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def load_feature_list(models_dir: str, league: str, corner: bool = False):
    """Load the saved feature list JSON for a league. Returns None if not found."""
    fname = f"{league}_corner_features.json" if corner else f"{league}_features.json"
    path = os.path.join(models_dir, fname)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def load_league_data(filepath: str) -> pd.DataFrame:
    """Load league data from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def prepare_row_to_predict(home_team_df: pd.DataFrame, away_team_df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    """Prepare a single row for prediction by averaging team statistics."""
    row_to_predict = pd.DataFrame(columns=numeric_columns)
    row_to_predict.loc[0] = [None] * len(row_to_predict.columns)

    home_team_final_df = home_team_df.head(5)[numeric_columns]
    away_team_final_df = away_team_df.head(5)[numeric_columns]

    for column in row_to_predict.columns:
        if column in HOME_TEAM_FEATURES:
            row_to_predict.loc[0, column] = home_team_final_df[column].mean()
        elif column in AWAY_TEAM_FEATURES:
            row_to_predict.loc[0, column] = away_team_final_df[column].mean()
        else:
            row_to_predict.loc[0, column] = (away_team_final_df[column].mean() + home_team_final_df[column].mean()) / 2

    return row_to_predict

def make_predictions(league: str, goals_model, corners_model, league_data: pd.DataFrame, competitions: dict, predict_corners: bool, models_dir: str = 'models') -> str:
    """Make predictions for a league (goals and optionally corners)."""
    league_section = ""
    for competition_league, competitions_info in competitions.items():
        if competition_league == league:
            league_section = f"**{competitions_info['name']}**:\n"
            for match in competitions_info["next_matches"]:
                home_team = match['home_team']
                away_team = match['away_team']

                if home_team not in league_data['HomeTeam'].values or away_team not in league_data['AwayTeam'].values:
                    continue

                home_team_df = league_data[league_data['HomeTeam'] == home_team]
                away_team_df = league_data[league_data['AwayTeam'] == away_team]

                numeric_columns = league_data.select_dtypes(include=['number']).columns.tolist()
                if 'Over2.5' in numeric_columns:
                    numeric_columns.remove('Over2.5')
                if 'OverUnder10.5Corners' in numeric_columns:
                    numeric_columns.remove('OverUnder10.5Corners')

                # Align numeric_columns with saved feature lists if available
                goals_feature_list = load_feature_list(models_dir, league, corner=False)
                corners_feature_list = load_feature_list(models_dir, league, corner=True)

                if goals_feature_list:
                    # keep only features present in both CSV and the saved list, preserving the saved order
                    numeric_columns = [c for c in goals_feature_list if c in league_data.columns and c not in ['Over2.5', 'OverUnder10.5Corners']]

                row_to_predict = prepare_row_to_predict(home_team_df, away_team_df, numeric_columns)
                X_test = row_to_predict.values

                # Goals prediction
                goals_pred = goals_model.predict(X_test)[0]
                goals_proba = goals_model.predict_proba(X_test)[0]
                
                if goals_pred == 1:
                    goals_result = f"Over 2.5 Goals! 🔥 ({round(goals_proba[1] * 100, 2)}% chance)"
                else:
                    goals_result = f"Under 2.5 Goals ({round(goals_proba[0] * 100, 2)}% chance)"

                # Corners prediction
                if predict_corners and corners_model:
                    try:
                        # Align features for corners using the saved feature list if available
                        if corners_feature_list:
                            corners_cols = [c for c in corners_feature_list if c in league_data.columns and c not in ['Over2.5', 'OverUnder10.5Corners']]
                            # Build aligned array for corners in the saved order
                            X_aligned = np.array([[row_to_predict.loc[0, c] if c in row_to_predict.columns else np.nan for c in corners_cols]])
                        else:
                            # fallback to previous behavior (min features)
                            goals_n_features = goals_model.n_features_in_ if hasattr(goals_model, 'n_features_in_') else X_test.shape[1]
                            corners_n_features = corners_model.n_features_in_ if hasattr(corners_model, 'n_features_in_') else X_test.shape[1]
                            min_features = min(goals_n_features, corners_n_features, X_test.shape[1])
                            X_aligned = X_test[:, :min_features]
                        
                        corners_pred = corners_model.predict(X_aligned)[0]
                        corners_proba = corners_model.predict_proba(X_aligned)[0]
                        
                        if corners_pred == 1:
                            corners_result = f" | Over 10.5 Corners: {round(corners_proba[1] * 100, 2)}%"
                        else:
                            corners_result = f" | Under 10.5 Corners: {round(corners_proba[0] * 100, 2)}%"
                    except Exception as e:
                        print(f"Warning: Could not make corner prediction: {e}")
                        corners_result = ""
                else:
                    corners_result = ""

                league_section += f"- ⚽ **{home_team}** 🆚 **{away_team}**: {goals_result}{corners_result}\n"

    return league_section

def main(input_leagues_models_dir: str, input_data_predict_dir: str, final_predictions_out_file: str, next_matches: str, predict_corners: bool):
    """Main prediction function."""
    try:
        print("Loading JSON file with upcoming matches...\n")
        with open(next_matches, 'r', encoding='utf-16') as json_file:
            competitions = json.load(json_file)
    except Exception as e:
        raise Exception(f"Error loading JSON file: {e}")

    title = "🎯 **AI Football Predictions: Goals & Corners** 🎯" if predict_corners else "🎯 **AI Football Predictions: Over 2.5 Goals?** 🎯"
    predictions_message = f"{title}\n\nCheck out the latest predictions for the upcoming football matches!\n PREDICTIONS DONE: {datetime.now().strftime('%Y-%m-%d')} \n\n"

    for league in VALID_LEAGUES:
        print(f"----------------------------------")
        print(f"\nMaking predictions for {league}...\n")
        
        goals_model_path = os.path.join(input_leagues_models_dir, f"{league}_voting_classifier.pkl")
        corners_model_path = os.path.join(input_leagues_models_dir, f"{league}_corner_voting_classifier.pkl")
        data_path = os.path.join(input_data_predict_dir, f"{league}_merged_preprocessed.csv")

        if not os.path.exists(goals_model_path) or not os.path.exists(data_path):
            print(f"Missing data or goals model for {league}. Skipping...")
            continue

        try:
            goals_model = load_model(goals_model_path)
            league_data = load_league_data(data_path)
            print(f"Loaded goals model and data for {league}.")
            
            corners_model = None
            if predict_corners and os.path.exists(corners_model_path):
                corners_model = load_model(corners_model_path)
                print(f"Loaded corners model for {league}.")
            
            print(f"Predicting matches for {league}...")
            league_section = make_predictions(league, goals_model, corners_model, league_data, competitions, predict_corners, models_dir=input_leagues_models_dir)
            print(f"Predictions made for {league}.")
            predictions_message += league_section + "\n"
        except Exception as e:
            print(f"Error making predictions for {league}: {e}")
            continue

    with open(final_predictions_out_file, 'w', encoding='utf-8') as file:
        file.write(predictions_message)
        print(f"\n Predictions saved to {final_predictions_out_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Football Predictions Script (Goals & Corners)")
    parser.add_argument('--input_leagues_models_dir', type=str, required=True, help="Directory containing model files")
    parser.add_argument('--input_data_predict_dir', type=str, required=True, help="Directory containing processed data files")
    parser.add_argument('--final_predictions_out_file', type=str, required=True, help="File path to save predictions")
    parser.add_argument('--next_matches', type=str, required=True, help="Path to JSON file with upcoming matches")
    parser.add_argument('--predict_corners', action='store_true', help="Include corner predictions in output")

    args = parser.parse_args()
    main(args.input_leagues_models_dir, args.input_data_predict_dir, args.final_predictions_out_file, args.next_matches, args.predict_corners)
