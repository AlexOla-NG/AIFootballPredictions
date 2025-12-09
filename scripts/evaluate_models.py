"""
Script to evaluate and track model performance metrics.

This script provides detailed evaluation of trained models including:
- Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices
- Model comparison over time
- Backtesting against actual results

Usage:
------
python scripts/evaluate_models.py --processed_data_input_dir data/processed --trained_models_output_dir models

"""

import os
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import cross_validate, KFold
import matplotlib.pyplot as plt
import seaborn as sns


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained football prediction models.")
    parser.add_argument('--processed_data_input_dir', type=str, required=True, 
                        help="Path to folder containing preprocessed CSV files.")
    parser.add_argument('--trained_models_output_dir', type=str, required=True, 
                        help="Path to folder containing trained models.")
    parser.add_argument('--metrics_output_dir', type=str, default='metrics',
                        help="Directory to save evaluation metrics and plots.")
    parser.add_argument('--target', type=str, default='Over2.5',
                        help="Target variable to evaluate (Over2.5 or OverUnder10.5Corners).")
    return parser.parse_args()


def load_data(processed_data_input_dir: str) -> dict:
    """Load preprocessed CSV files."""
    data = {}
    for file in os.listdir(processed_data_input_dir):
        if file.endswith('_merged_preprocessed.csv'):
            league_name = file.split('_')[0]
            filepath = os.path.join(processed_data_input_dir, file)
            data[league_name] = pd.read_csv(filepath)
    return data


def prepare_data(df: pd.DataFrame, target: str = 'Over2.5'):
    """Prepare data for evaluation."""
    from sklearn.preprocessing import StandardScaler
    
    df = df.dropna(subset=[target])
    
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    if target in numerical_columns:
        numerical_columns.remove(target)
    if 'Over2.5' in numerical_columns and target != 'Over2.5':
        numerical_columns.remove('Over2.5')
    if 'OverUnder10.5Corners' in numerical_columns and target != 'OverUnder10.5Corners':
        numerical_columns.remove('OverUnder10.5Corners')
    
    X = df[numerical_columns].fillna(df[numerical_columns].mean())
    y = df[target].astype(int)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, numerical_columns


def evaluate_model(model, X, y, league_name: str, target: str = 'Over2.5', n_splits: int = 5):
    """Evaluate model with multiple metrics."""
    
    # Cross-validation metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision_weighted': 'precision_weighted',
        'recall_weighted': 'recall_weighted',
        'f1_weighted': 'f1_weighted',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(model, X, y, cv=n_splits, scoring=scoring, return_train_score=True)
    
    metrics = {
        'league': league_name,
        'target': target,
        'timestamp': datetime.now().isoformat(),
        'cross_validation_folds': n_splits,
    }
    
    # Store mean and std for each metric
    for metric_name in scoring.keys():
        test_key = f'test_{metric_name}'
        train_key = f'train_{metric_name}'
        
        metrics[f'{metric_name}_test_mean'] = float(cv_results[test_key].mean())
        metrics[f'{metric_name}_test_std'] = float(cv_results[test_key].std())
        metrics[f'{metric_name}_train_mean'] = float(cv_results[train_key].mean())
        metrics[f'{metric_name}_train_std'] = float(cv_results[train_key].std())
    
    return metrics, cv_results


def save_metrics_json(metrics: dict, output_dir: str, league_name: str, target: str):
    """Save metrics to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{league_name}_{target.replace('.', '_')}_metrics.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return filepath


def print_metrics(metrics: dict):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"League: {metrics['league']} | Target: {metrics['target']}")
    print(f"Evaluation Time: {metrics['timestamp']}")
    print(f"{'='*60}")
    
    print("\n📊 CROSS-VALIDATION RESULTS (5-Fold):")
    print(f"{'Metric':<20} {'Test Mean':<15} {'Test Std':<15} {'Train Mean':<15}")
    print("-" * 65)
    
    for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']:
        test_mean = metrics[f'{metric}_test_mean']
        test_std = metrics[f'{metric}_test_std']
        train_mean = metrics[f'{metric}_train_mean']
        
        print(f"{metric:<20} {test_mean:<15.4f} {test_std:<15.4f} {train_mean:<15.4f}")
    
    print(f"{'='*60}\n")


def create_comparison_report(all_metrics: list, output_dir: str):
    """Create a comparison report across all leagues and targets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier viewing
    df_metrics = pd.DataFrame(all_metrics)
    
    # Save comparison
    comparison_file = os.path.join(output_dir, 'model_comparison.csv')
    
    # Select key columns
    key_cols = ['league', 'target', 'timestamp', 'accuracy_test_mean', 'precision_weighted_test_mean', 
                'recall_weighted_test_mean', 'f1_weighted_test_mean', 'roc_auc_test_mean']
    
    df_comparison = df_metrics[key_cols].copy()
    df_comparison = df_comparison.rename(columns={
        'accuracy_test_mean': 'Accuracy',
        'precision_weighted_test_mean': 'Precision',
        'recall_weighted_test_mean': 'Recall',
        'f1_weighted_test_mean': 'F1-Score',
        'roc_auc_test_mean': 'ROC-AUC'
    })
    
    df_comparison.to_csv(comparison_file, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(df_comparison.to_string(index=False))
    print("="*80 + "\n")
    
    return comparison_file


def plot_metrics_comparison(all_metrics: list, output_dir: str):
    """Create visualization of metrics across leagues."""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(all_metrics)
    
    # Separate goals and corner models
    goals_df = df[df['target'] == 'Over2.5']
    corners_df = df[df['target'] == 'OverUnder10.5Corners']
    
    # Create figure for goals models
    if not goals_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Goals Prediction Models (Over/Under 2.5) - Performance Metrics', fontsize=14, fontweight='bold')
        
        # Accuracy, Precision, Recall, F1
        ax = axes[0]
        metrics_to_plot = ['accuracy_test_mean', 'precision_weighted_test_mean', 'recall_weighted_test_mean', 'f1_weighted_test_mean']
        goals_df.set_index('league')[metrics_to_plot].plot(kind='bar', ax=ax)
        ax.set_title('Standard Metrics')
        ax.set_ylabel('Score')
        ax.set_ylim([0.7, 1.0])
        ax.legend(['Accuracy', 'Precision', 'Recall', 'F1'], loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        
        # ROC-AUC
        ax = axes[1]
        goals_df.set_index('league')['roc_auc_test_mean'].plot(kind='bar', ax=ax, color='orange')
        ax.set_title('ROC-AUC Score')
        ax.set_ylabel('ROC-AUC')
        ax.set_ylim([0.7, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'goals_metrics_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: goals_metrics_comparison.png")
        plt.close()
    
    # Create figure for corner models
    if not corners_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Corner Prediction Models (Over/Under 10.5) - Performance Metrics', fontsize=14, fontweight='bold')
        
        ax = axes[0]
        metrics_to_plot = ['accuracy_test_mean', 'precision_weighted_test_mean', 'recall_weighted_test_mean', 'f1_weighted_test_mean']
        corners_df.set_index('league')[metrics_to_plot].plot(kind='bar', ax=ax)
        ax.set_title('Standard Metrics')
        ax.set_ylabel('Score')
        ax.set_ylim([0.5, 1.0])
        ax.legend(['Accuracy', 'Precision', 'Recall', 'F1'], loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        
        ax = axes[1]
        corners_df.set_index('league')['roc_auc_test_mean'].plot(kind='bar', ax=ax, color='green')
        ax.set_title('ROC-AUC Score')
        ax.set_ylabel('ROC-AUC')
        ax.set_ylim([0.5, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'corners_metrics_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: corners_metrics_comparison.png")
        plt.close()


def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.metrics_output_dir, exist_ok=True)
    
    # Load data
    print("Loading preprocessed data...")
    data = load_data(args.processed_data_input_dir)
    
    all_metrics = []
    
    # Evaluate goals models
    print("\n" + "="*60)
    print("EVALUATING GOALS PREDICTION MODELS (Over/Under 2.5)")
    print("="*60)
    
    for league_name, df in data.items():
        model_path = os.path.join(args.trained_models_output_dir, f"{league_name}_voting_classifier.pkl")
        
        if not os.path.exists(model_path):
            print(f"⚠ Model not found for {league_name}")
            continue
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Prepare data
        X, y, _ = prepare_data(df, target='Over2.5')
        
        # Evaluate
        metrics, _ = evaluate_model(model, X, y, league_name, target='Over2.5', n_splits=5)
        all_metrics.append(metrics)
        print_metrics(metrics)
        
        # Save metrics
        save_metrics_json(metrics, args.metrics_output_dir, league_name, 'goals')
    
    # Evaluate corner models
    print("\n" + "="*60)
    print("EVALUATING CORNER PREDICTION MODELS (Over/Under 10.5)")
    print("="*60)
    
    for league_name, df in data.items():
        model_path = os.path.join(args.trained_models_output_dir, f"{league_name}_corner_voting_classifier.pkl")
        
        if not os.path.exists(model_path):
            print(f"⚠ Corner model not found for {league_name}")
            continue
        
        if 'OverUnder10.5Corners' not in df.columns:
            print(f"⚠ Corner target not found for {league_name}")
            continue
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Prepare data
        X, y, _ = prepare_data(df, target='OverUnder10.5Corners')
        
        # Evaluate
        metrics, _ = evaluate_model(model, X, y, league_name, target='OverUnder10.5Corners', n_splits=5)
        all_metrics.append(metrics)
        print_metrics(metrics)
        
        # Save metrics
        save_metrics_json(metrics, args.metrics_output_dir, league_name, 'corners')
    
    # Create comparison report
    if all_metrics:
        print("\nGenerating comparison reports...")
        create_comparison_report(all_metrics, args.metrics_output_dir)
        plot_metrics_comparison(all_metrics, args.metrics_output_dir)
        
        print(f"\n✓ All metrics saved to: {args.metrics_output_dir}/")
        print(f"  - Detailed JSON metrics for each league")
        print(f"  - model_comparison.csv (summary table)")
        print(f"  - Visualization plots (PNG)")


if __name__ == "__main__":
    main()
