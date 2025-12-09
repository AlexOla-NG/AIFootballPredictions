"""
Backtesting script to evaluate predictions against actual match results.

This script tests predictions on historical data to assess:
- Overall accuracy
- Precision/Recall for Over and Under predictions
- Performance per league
- Performance over time

Usage:
------
python scripts/backtest_predictions.py --processed_data_input_dir data/processed --trained_models_output_dir models

"""

import os
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Backtest football prediction models.")
    parser.add_argument('--processed_data_input_dir', type=str, required=True,
                        help="Path to folder containing preprocessed CSV files.")
    parser.add_argument('--trained_models_output_dir', type=str, required=True,
                        help="Path to folder containing trained models.")
    parser.add_argument('--backtest_output_dir', type=str, default='backtest_results',
                        help="Directory to save backtest results.")
    parser.add_argument('--target', type=str, default='Over2.5',
                        help="Target variable to backtest (Over2.5 or OverUnder10.5Corners).")
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
    """Prepare data for backtesting."""
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


def backtest_model(model, X, y, league_name: str, target: str = 'Over2.5', n_splits: int = 5):
    """Backtest model using time series cross-validation."""
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_test = X[test_idx]
        y_test = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_0 = precision_score(y_test, y_pred, labels=[0, 1], average=None)[0] if len(np.unique(y_pred)) > 1 else 0
        precision_1 = precision_score(y_test, y_pred, labels=[0, 1], average=None)[1] if len(np.unique(y_pred)) > 1 else 0
        recall_0 = recall_score(y_test, y_pred, labels=[0, 1], average=None)[0] if len(np.unique(y_pred)) > 1 else 0
        recall_1 = recall_score(y_test, y_pred, labels=[0, 1], average=None)[1] if len(np.unique(y_pred)) > 1 else 0
        
        fold_results.append({
            'fold': fold_idx + 1,
            'accuracy': accuracy,
            'precision_under': precision_0,
            'precision_over': precision_1,
            'recall_under': recall_0,
            'recall_over': recall_1,
            'test_size': len(test_idx)
        })
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        if y_proba is not None:
            all_y_proba.extend(y_proba)
    
    # Summary statistics
    summary = {
        'league': league_name,
        'target': target,
        'timestamp': datetime.now().isoformat(),
        'total_backtests': n_splits,
        'total_predictions': len(all_y_true),
        'overall_accuracy': float(np.mean([f['accuracy'] for f in fold_results])),
        'accuracy_std': float(np.std([f['accuracy'] for f in fold_results])),
        'average_precision_under': float(np.mean([f['precision_under'] for f in fold_results])),
        'average_precision_over': float(np.mean([f['precision_over'] for f in fold_results])),
        'average_recall_under': float(np.mean([f['recall_under'] for f in fold_results])),
        'average_recall_over': float(np.mean([f['recall_over'] for f in fold_results])),
        'fold_results': fold_results
    }
    
    return summary, all_y_true, all_y_pred, all_y_proba


def print_backtest_results(summary: dict):
    """Print backtest results."""
    target_name = "Goals (Over 2.5)" if summary['target'] == 'Over2.5' else "Corners (Over 10.5)"
    
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS: {summary['league']} - {target_name}")
    print(f"{'='*70}")
    print(f"Total Predictions: {summary['total_predictions']}")
    print(f"Test Folds: {summary['total_backtests']}")
    print(f"\n{'Metric':<30} {'Score':<20}")
    print("-" * 50)
    print(f"{'Overall Accuracy':<30} {summary['overall_accuracy']:<20.4f} (± {summary['accuracy_std']:.4f})")
    print(f"{'Precision - Under':<30} {summary['average_precision_under']:<20.4f}")
    print(f"{'Precision - Over':<30} {summary['average_precision_over']:<20.4f}")
    print(f"{'Recall - Under':<30} {summary['average_recall_under']:<20.4f}")
    print(f"{'Recall - Over':<30} {summary['average_recall_over']:<20.4f}")
    print(f"{'='*70}\n")


def save_backtest_results(summary: dict, output_dir: str, league_name: str, target: str):
    """Save backtest results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{league_name}_{target.replace('.', '_')}_backtest.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return filepath


def create_backtest_comparison(all_results: list, output_dir: str):
    """Create comparison report for all backtests."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df_results = []
    for result in all_results:
        df_results.append({
            'League': result['league'],
            'Target': result['target'],
            'Accuracy': result['overall_accuracy'],
            'Accuracy Std': result['accuracy_std'],
            'Precision Under': result['average_precision_under'],
            'Precision Over': result['average_precision_over'],
            'Recall Under': result['average_recall_under'],
            'Recall Over': result['average_recall_over'],
        })
    
    df = pd.DataFrame(df_results)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'backtest_comparison.csv')
    df.to_csv(csv_path, index=False)
    
    # Print summary
    print("\n" + "="*100)
    print("BACKTEST COMPARISON SUMMARY")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")
    
    return csv_path


def plot_backtest_results(all_results: list, output_dir: str):
    """Create visualizations of backtest results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate goals and corner results
    goals_results = [r for r in all_results if r['target'] == 'Over2.5']
    corners_results = [r for r in all_results if r['target'] == 'OverUnder10.5Corners']
    
    # Plot goals backtests
    if goals_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Goals Prediction Models - Backtest Results', fontsize=14, fontweight='bold')
        
        leagues = [r['league'] for r in goals_results]
        accuracies = [r['overall_accuracy'] for r in goals_results]
        accuracy_stds = [r['accuracy_std'] for r in goals_results]
        
        # Accuracy
        ax = axes[0, 0]
        ax.bar(leagues, accuracies, yerr=accuracy_stds, capsize=5, color='skyblue')
        ax.set_title('Overall Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Precision comparison
        ax = axes[0, 1]
        precisions_under = [r['average_precision_under'] for r in goals_results]
        precisions_over = [r['average_precision_over'] for r in goals_results]
        x = np.arange(len(leagues))
        width = 0.35
        ax.bar(x - width/2, precisions_under, width, label='Under 2.5', color='lightcoral')
        ax.bar(x + width/2, precisions_over, width, label='Over 2.5', color='lightgreen')
        ax.set_title('Precision by Prediction')
        ax.set_ylabel('Precision')
        ax.set_xticks(x)
        ax.set_xticklabels(leagues)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Recall comparison
        ax = axes[1, 0]
        recalls_under = [r['average_recall_under'] for r in goals_results]
        recalls_over = [r['average_recall_over'] for r in goals_results]
        ax.bar(x - width/2, recalls_under, width, label='Under 2.5', color='lightcoral')
        ax.bar(x + width/2, recalls_over, width, label='Over 2.5', color='lightgreen')
        ax.set_title('Recall by Prediction')
        ax.set_ylabel('Recall')
        ax.set_xticks(x)
        ax.set_xticklabels(leagues)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Average performance across leagues
        ax = axes[1, 1]
        metrics = ['Accuracy', 'Precision Under', 'Precision Over', 'Recall Under', 'Recall Over']
        avg_values = [
            np.mean(accuracies),
            np.mean(precisions_under),
            np.mean(precisions_over),
            np.mean(recalls_under),
            np.mean(recalls_over),
        ]
        ax.barh(metrics, avg_values, color='steelblue')
        ax.set_xlabel('Score')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'goals_backtest_results.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: goals_backtest_results.png")
        plt.close()
    
    # Plot corner backtests
    if corners_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Corner Prediction Models - Backtest Results', fontsize=14, fontweight='bold')
        
        leagues = [r['league'] for r in corners_results]
        accuracies = [r['overall_accuracy'] for r in corners_results]
        accuracy_stds = [r['accuracy_std'] for r in corners_results]
        
        # Accuracy
        ax = axes[0, 0]
        ax.bar(leagues, accuracies, yerr=accuracy_stds, capsize=5, color='lightgreen')
        ax.set_title('Overall Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Precision comparison
        ax = axes[0, 1]
        precisions_under = [r['average_precision_under'] for r in corners_results]
        precisions_over = [r['average_precision_over'] for r in corners_results]
        x = np.arange(len(leagues))
        width = 0.35
        ax.bar(x - width/2, precisions_under, width, label='Under 10.5', color='lightcoral')
        ax.bar(x + width/2, precisions_over, width, label='Over 10.5', color='lightyellow')
        ax.set_title('Precision by Prediction')
        ax.set_ylabel('Precision')
        ax.set_xticks(x)
        ax.set_xticklabels(leagues)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Recall comparison
        ax = axes[1, 0]
        recalls_under = [r['average_recall_under'] for r in corners_results]
        recalls_over = [r['average_recall_over'] for r in corners_results]
        ax.bar(x - width/2, recalls_under, width, label='Under 10.5', color='lightcoral')
        ax.bar(x + width/2, recalls_over, width, label='Over 10.5', color='lightyellow')
        ax.set_title('Recall by Prediction')
        ax.set_ylabel('Recall')
        ax.set_xticks(x)
        ax.set_xticklabels(leagues)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Average performance
        ax = axes[1, 1]
        metrics = ['Accuracy', 'Precision Under', 'Precision Over', 'Recall Under', 'Recall Over']
        avg_values = [
            np.mean(accuracies),
            np.mean(precisions_under),
            np.mean(precisions_over),
            np.mean(recalls_under),
            np.mean(recalls_over),
        ]
        ax.barh(metrics, avg_values, color='mediumseagreen')
        ax.set_xlabel('Score')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'corners_backtest_results.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: corners_backtest_results.png")
        plt.close()


def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.backtest_output_dir, exist_ok=True)
    
    # Load data
    print("Loading preprocessed data...")
    data = load_data(args.processed_data_input_dir)
    
    all_results = []
    
    # Backtest goals models
    print("\n" + "="*70)
    print("BACKTESTING GOALS PREDICTION MODELS (Over/Under 2.5)")
    print("="*70)
    
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
        
        # Backtest
        summary, y_true, y_pred, y_proba = backtest_model(model, X, y, league_name, target='Over2.5', n_splits=5)
        all_results.append(summary)
        print_backtest_results(summary)
        
        # Save results
        save_backtest_results(summary, args.backtest_output_dir, league_name, 'goals')
    
    # Backtest corner models
    print("\n" + "="*70)
    print("BACKTESTING CORNER PREDICTION MODELS (Over/Under 10.5)")
    print("="*70)
    
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
        
        # Backtest
        summary, y_true, y_pred, y_proba = backtest_model(model, X, y, league_name, target='OverUnder10.5Corners', n_splits=5)
        all_results.append(summary)
        print_backtest_results(summary)
        
        # Save results
        save_backtest_results(summary, args.backtest_output_dir, league_name, 'corners')
    
    # Create comparison and visualizations
    if all_results:
        print("\nGenerating comparison reports and visualizations...")
        create_backtest_comparison(all_results, args.backtest_output_dir)
        plot_backtest_results(all_results, args.backtest_output_dir)
        
        print(f"✓ All backtest results saved to: {args.backtest_output_dir}/")


if __name__ == "__main__":
    main()
