"""
Script to train machine learning models for predicting corners in football matches.

This script is similar to train_models.py but focuses on predicting whether matches will have
Over/Under 10.5 total corners instead of Over/Under 2.5 goals.

Usage:
------
Run this script from the terminal as follows, from the root directory of the project:

    python scripts/train_corner_models.py --processed_data_input_dir data/processed --trained_models_output_dir models --metric_choice accuracy --n_splits 10 --voting soft

Parameters:
-----------
processed_data_input_dir : str
    Path to the folder containing the preprocessed CSV files.
trained_models_output_dir : str
    Directory where the trained voting classifier models will be saved.
metric_choice : str
    Evaluation metric ('accuracy', 'precision', 'recall', 'f1').
n_splits : int
    Number of splits for cross-validation.
voting : str
    Type of voting ('hard' or 'soft').
"""

import os
import argparse
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train corner prediction models for football matches.")
    parser.add_argument("--processed_data_input_dir", required=True, type=str, help="Path to preprocessed data.")
    parser.add_argument("--trained_models_output_dir", required=True, type=str, help="Directory to save trained models.")
    parser.add_argument("--metric_choice", type=str, default="accuracy", choices=["accuracy", "precision", "recall", "f1"], help="Evaluation metric.")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of cross-validation splits.")
    parser.add_argument("--voting", type=str, default="soft", choices=["hard", "soft"], help="Type of voting.")
    return parser.parse_args()

def load_preprocessed_data(filepath: str) -> pd.DataFrame:
    """Load preprocessed CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def prepare_data(df: pd.DataFrame, target_column: str):
    """Prepare data for model training with feature scaling."""
    # Drop rows with missing target
    df = df.dropna(subset=[target_column])
    
    # Select numeric columns (excluding target and date)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    if 'Over2.5' in numeric_cols:
        numeric_cols.remove('Over2.5')
    
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    y = df[target_column].astype(int)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, numeric_cols, scaler

def train_model(model, X_train, y_train):
    """Train a single model."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y, metric_choice, n_splits):
    """Evaluate model using cross-validation."""
    scoring = {'accuracy': 'accuracy', 'precision': 'precision_weighted', 'recall': 'recall_weighted', 'f1': 'f1_weighted'}
    cv_results = cross_validate(model, X, y, cv=n_splits, scoring=scoring[metric_choice], return_train_score=False)
    mean_score = cv_results['test_score'].mean()
    std_score = cv_results['test_score'].std()
    return mean_score, std_score

def train_corner_models(processed_data_dir: str, output_dir: str, metric_choice: str, n_splits: int, voting_type: str):
    """Main training function for corner prediction models."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Valid leagues
    valid_leagues = ["E0", "I1", "D1", "SP1", "F1"]
    
    for league in valid_leagues:
        print(f"\nProcessing league: {league}")
        data_file = os.path.join(processed_data_dir, f"{league}_merged_preprocessed.csv")
        
        if not os.path.exists(data_file):
            print(f"Data file not found for {league}. Skipping...")
            continue
        
        try:
            df = load_preprocessed_data(data_file)
        except Exception as e:
            print(f"Error loading data for {league}: {e}")
            continue
        
        # Check if corner target exists
        if 'OverUnder10.5Corners' not in df.columns:
            print(f"Corner target 'OverUnder10.5Corners' not found for {league}. Skipping...")
            continue
        
        print(f"Preparing data for {league}...")
        try:
            X, y, numeric_cols, scaler = prepare_data(df, 'OverUnder10.5Corners')
        except Exception as e:
            print(f"Error preparing data for {league}: {e}")
            continue
        
        if len(X) < 50:
            print(f"Insufficient data for {league} ({len(X)} samples). Skipping...")
            continue
        
        print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Initialize models
        models = {
            'LogisticRegression': LogisticRegression(max_iter=3000, random_state=42),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(probability=True, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, verbosity=0),
            'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
            'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42)
        }
        
        best_models = []
        
        for model_name, model in models.items():
            logging.info(f"\nEvaluating {model_name}...")
            try:
                mean_score, std_score = evaluate_model(model, X, y, metric_choice, n_splits)
                logging.info(f"{model_name} - {metric_choice}: {mean_score:.4f} ± {std_score:.4f}")
                best_models.append((model_name, model, mean_score))
            except Exception as e:
                logging.warning(f"Error evaluating {model_name}: {e}")
                continue
        
        # Select top 3 models for voting
        best_models = sorted(best_models, key=lambda x: x[2], reverse=True)[:3]
        
        if len(best_models) < 2:
            print(f"Not enough models trained for {league}. Skipping...")
            continue
        
        logging.info(f"\nTraining final models for voting ensemble...")
        voting_models = [(name, model) for name, model, _ in best_models]
        
        # Train voting classifier
        voting_clf = VotingClassifier(estimators=voting_models, voting=voting_type)
        voting_clf.fit(X, y)
        mean_score, std_score = evaluate_model(voting_clf, X, y, metric_choice, n_splits)
        logging.info(f"Voting Classifier ({voting_type}) - {metric_choice}: {mean_score:.4f} ± {std_score:.4f}")
        
        # Save corner model
        model_path = os.path.join(output_dir, f"{league}_corner_voting_classifier.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(voting_clf, f)
        logging.info(f"Corner model saved to {model_path}")

        # Save feature list used
        try:
            features_path = os.path.join(output_dir, f"{league}_corner_features.json")
            with open(features_path, 'w', encoding='utf-8') as hf:
                json.dump(numeric_cols, hf, ensure_ascii=False, indent=2)
            logging.info(f"Corner feature list saved to {features_path}")
        except Exception as e:
            logging.warning(f"Could not save corner feature list for {league}: {e}")

        # Write sentinel file for completion
        try:
            from datetime import datetime
            sentinel = os.path.join(output_dir, f"{league}_corner_training_complete.txt")
            with open(sentinel, 'w') as s:
                s.write(f"league={league}\n")
                s.write(f"model={os.path.basename(model_path)}\n")
                s.write(f"metric={metric_choice}\n")
                s.write(f"mean_cv_score={mean_score:.6f}\n")
                s.write(f"std_cv_score={std_score:.6f}\n")
                s.write(f"completed_at={datetime.now().isoformat()}\n")
            logging.info(f"Sentinel written: {sentinel}")
        except Exception as e:
            logging.warning(f"Could not write sentinel file: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    
    if not os.path.exists(args.processed_data_input_dir):
        print(f"Input directory {args.processed_data_input_dir} does not exist.")
        exit(1)
    
    if not os.path.exists(args.trained_models_output_dir):
        os.makedirs(args.trained_models_output_dir)
    
    train_corner_models(
        args.processed_data_input_dir,
        args.trained_models_output_dir,
        args.metric_choice,
        args.n_splits,
        args.voting
    )
    print("\nCorner model training completed!")
