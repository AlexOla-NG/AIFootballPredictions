"""
Script to track model performance metrics over time.

This script maintains a historical record of model metrics to identify trends and improvements.
Metrics are tracked each time models are trained.

Usage:
------
python scripts/track_metrics.py --metrics_dir metrics --backtest_dir backtest_results --output_file metrics_history.json

"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Track model performance metrics over time.")
    parser.add_argument('--metrics_dir', type=str, default='metrics',
                        help="Directory containing evaluation metrics.")
    parser.add_argument('--backtest_dir', type=str, default='backtest_results',
                        help="Directory containing backtest results.")
    parser.add_argument('--output_file', type=str, default='metrics_history.json',
                        help="File to save historical metrics.")
    return parser.parse_args()


def load_metrics_from_dir(metrics_dir: str) -> list:
    """Load all metric files from metrics directory."""
    metrics_list = []
    
    if not os.path.exists(metrics_dir):
        return metrics_list
    
    for file in os.listdir(metrics_dir):
        if file.endswith('_metrics.json'):
            filepath = os.path.join(metrics_dir, file)
            with open(filepath, 'r') as f:
                metric = json.load(f)
                metrics_list.append(metric)
    
    return metrics_list


def load_backtest_from_dir(backtest_dir: str) -> list:
    """Load all backtest files from backtest directory."""
    backtest_list = []
    
    if not os.path.exists(backtest_dir):
        return backtest_list
    
    for file in os.listdir(backtest_dir):
        if file.endswith('_backtest.json'):
            filepath = os.path.join(backtest_dir, file)
            with open(filepath, 'r') as f:
                backtest = json.load(f)
                backtest_list.append(backtest)
    
    return backtest_list


def load_history(history_file: str) -> dict:
    """Load existing metrics history."""
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return {'evaluations': [], 'backtests': []}


def save_history(history: dict, history_file: str):
    """Save metrics history to file."""
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def create_comparison_report(history: dict, output_dir: str = '.'):
    """Create a comparison report showing metrics over time."""
    
    if not history['evaluations']:
        return
    
    # Create DataFrame for evaluations
    eval_records = []
    for eval_item in history['evaluations']:
        for league_target, data in eval_item.items():
            record = {
                'Timestamp': data.get('timestamp', ''),
                'League': data.get('league', ''),
                'Target': data.get('target', ''),
                'Accuracy': data.get('accuracy_test_mean', ''),
                'Precision': data.get('precision_weighted_test_mean', ''),
                'Recall': data.get('recall_weighted_test_mean', ''),
                'F1': data.get('f1_weighted_test_mean', ''),
                'ROC-AUC': data.get('roc_auc_test_mean', ''),
            }
            eval_records.append(record)
    
    if eval_records:
        df_eval = pd.DataFrame(eval_records)
        csv_path = os.path.join(output_dir, 'metrics_history.csv')
        df_eval.to_csv(csv_path, index=False)
        print(f"\n✓ Saved evaluation metrics history: {csv_path}")
    
    # Create DataFrame for backtests
    backtest_records = []
    for backtest_item in history['backtests']:
        for league_target, data in backtest_item.items():
            record = {
                'Timestamp': data.get('timestamp', ''),
                'League': data.get('league', ''),
                'Target': data.get('target', ''),
                'Accuracy': data.get('overall_accuracy', ''),
                'Accuracy Std': data.get('accuracy_std', ''),
                'Precision Under': data.get('average_precision_under', ''),
                'Precision Over': data.get('average_precision_over', ''),
                'Recall Under': data.get('average_recall_under', ''),
                'Recall Over': data.get('average_recall_over', ''),
            }
            backtest_records.append(record)
    
    if backtest_records:
        df_backtest = pd.DataFrame(backtest_records)
        csv_path = os.path.join(output_dir, 'backtest_history.csv')
        df_backtest.to_csv(csv_path, index=False)
        print(f"✓ Saved backtest metrics history: {csv_path}")


def print_metrics_summary(history: dict):
    """Print a summary of the latest metrics."""
    
    print("\n" + "="*80)
    print("METRICS HISTORY SUMMARY")
    print("="*80)
    
    if history['evaluations']:
        print("\nLatest Evaluation Metrics:")
        print("-" * 80)
        latest_eval = history['evaluations'][-1]
        for league_target, data in latest_eval.items():
            print(f"\n{league_target}:")
            print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
            print(f"  Accuracy:  {data.get('accuracy_test_mean', 'N/A'):.4f}")
            print(f"  Precision: {data.get('precision_weighted_test_mean', 'N/A'):.4f}")
            print(f"  Recall:    {data.get('recall_weighted_test_mean', 'N/A'):.4f}")
            print(f"  F1-Score:  {data.get('f1_weighted_test_mean', 'N/A'):.4f}")
            print(f"  ROC-AUC:   {data.get('roc_auc_test_mean', 'N/A'):.4f}")
    
    if history['backtests']:
        print("\n" + "-" * 80)
        print("\nLatest Backtest Results:")
        print("-" * 80)
        latest_backtest = history['backtests'][-1]
        for league_target, data in latest_backtest.items():
            print(f"\n{league_target}:")
            print(f"  Timestamp:        {data.get('timestamp', 'N/A')}")
            print(f"  Accuracy:         {data.get('overall_accuracy', 'N/A'):.4f}")
            print(f"  Precision (Under):{data.get('average_precision_under', 'N/A'):.4f}")
            print(f"  Precision (Over): {data.get('average_precision_over', 'N/A'):.4f}")
            print(f"  Recall (Under):   {data.get('average_recall_under', 'N/A'):.4f}")
            print(f"  Recall (Over):    {data.get('average_recall_over', 'N/A'):.4f}")
    
    print("\n" + "="*80 + "\n")


def main():
    args = parse_arguments()
    
    # Load current metrics and backtests
    print("Loading metrics and backtests...")
    current_metrics = load_metrics_from_dir(args.metrics_dir)
    current_backtests = load_backtest_from_dir(args.backtest_dir)
    
    # Load history
    history = load_history(args.output_file)
    
    # Group current metrics by league+target
    if current_metrics:
        metrics_grouped = {}
        for metric in current_metrics:
            key = f"{metric['league']}_{metric['target']}"
            metrics_grouped[key] = metric
        history['evaluations'].append(metrics_grouped)
    
    # Group current backtests by league+target
    if current_backtests:
        backtests_grouped = {}
        for backtest in current_backtests:
            key = f"{backtest['league']}_{backtest['target']}"
            backtests_grouped[key] = backtest
        history['backtests'].append(backtests_grouped)
    
    # Save updated history
    save_history(history, args.output_file)
    print(f"✓ Metrics history saved to: {args.output_file}")
    
    # Create comparison reports
    output_dir = os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.'
    create_comparison_report(history, output_dir)
    
    # Print summary
    print_metrics_summary(history)
    
    print(f"Total evaluation snapshots: {len(history['evaluations'])}")
    print(f"Total backtest snapshots: {len(history['backtests'])}")


if __name__ == "__main__":
    main()
