# Model Performance Tracking - Implementation Summary

## What Was Added

You now have a complete model evaluation and performance tracking system with three new scripts:

### 1. **evaluate_models.py** - Detailed Metrics Evaluation
Provides comprehensive metrics for all trained models:
- **Accuracy, Precision, Recall, F1-Score** per league and target
- **ROC-AUC scores** for model discrimination ability
- **Cross-validation results** (test vs training scores)
- **CSV comparison reports** across all leagues
- **Visualization plots** showing performance comparisons

**Usage:**
```bash
python scripts/evaluate_models.py --processed_data_input_dir data/processed --trained_models_output_dir models
```

**Output:**
- `metrics/` - Individual league metrics (JSON files)
- `model_comparison.csv` - Summary table
- PNG visualizations of performance

---

### 2. **backtest_predictions.py** - Historical Performance Testing
Tests models against actual historical data using time-series cross-validation:
- **Real-world accuracy** on held-out test periods
- **Separate precision/recall** for "Over" vs "Under" predictions
- **Per-league performance** rankings
- **Fold-by-fold breakdown** to identify variance

**Usage:**
```bash
python scripts/backtest_predictions.py --processed_data_input_dir data/processed --trained_models_output_dir models
```

**Output:**
- `backtest_results/` - Detailed results (JSON files)
- `backtest_comparison.csv` - Summary metrics
- PNG visualizations showing performance by league

---

### 3. **track_metrics.py** - Historical Metrics Tracking
Maintains a historical record across model training iterations:
- **Tracks snapshots** of metrics and backtest results over time
- **Identifies trends** - which metrics are improving/degrading
- **Creates timeline reports** for easy comparison
- **JSON history file** for programmatic access

**Usage:**
```bash
python scripts/track_metrics.py --metrics_dir metrics --backtest_dir backtest_results
```

**Output:**
- `metrics_history.json` - Complete historical record
- `metrics_history.csv` - Evaluation metrics timeline
- `backtest_history.csv` - Backtest results timeline

---

## How to Use These Together

### Full Automated Pipeline (Recommended)
```bash
./run_full_pipeline.sh
```

This now includes 9 steps:
1. Install dependencies
2. Acquire historical data
3. Preprocess data & engineer features
4. Train goals models
5. Train corner models
6. Acquire next matches
7. **Make predictions** ← Step 6
8. **Evaluate models** ← NEW Step 7
9. **Backtest predictions** ← NEW Step 8
10. **Track metrics over time** ← NEW Step 9

### Step-by-Step Manual Approach

After training models:

```bash
# 1. Evaluate current models
python scripts/evaluate_models.py \
    --processed_data_input_dir data/processed \
    --trained_models_output_dir models

# 2. Backtest on historical data
python scripts/backtest_predictions.py \
    --processed_data_input_dir data/processed \
    --trained_models_output_dir models

# 3. Track metrics over time
python scripts/track_metrics.py \
    --metrics_dir metrics \
    --backtest_dir backtest_results
```

---

## Understanding the Metrics

### Cross-Validation Metrics (from evaluate_models.py)
- **Accuracy**: Overall percentage of correct predictions
- **Precision**: Of your "Over" predictions, how many were correct?
- **Recall**: Of all actual "Over" matches, how many did you predict?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model's ability to distinguish between classes (0-1 scale, higher is better)

### Backtest Metrics (from backtest_predictions.py)
- **Accuracy**: % correct on held-out test data
- **Precision Under/Over**: How accurate are your specific predictions?
- **Recall Under/Over**: What % of actual outcomes did you catch?

### Tracking Trends (from track_metrics.py)
- Compare metrics across multiple training runs
- Identify which changes improved/hurt performance
- Track metrics over weeks/months to show progress

---

## Expected Output

### Example Evaluation Output
```
==============================================================
League: E0 | Target: Over2.5
Evaluation Time: 2025-12-07T15:30:00
==============================================================

📊 CROSS-VALIDATION RESULTS (5-Fold):
Metric               Test Mean       Test Std        Train Mean
-----------------------------------------------------------------
accuracy             0.8296          0.0555          0.8500
precision_weighted   0.8250          0.0600          0.8450
recall_weighted      0.8296          0.0555          0.8500
f1_weighted          0.8270          0.0580          0.8470
roc_auc              0.8950          0.0400          0.9100
```

### Example Backtest Output
```
======================================================================
BACKTEST RESULTS: E0 - Goals (Over 2.5)
======================================================================
Total Predictions: 1500
Test Folds: 5

Metric                         Score
--------------------------------------------------
Overall Accuracy              0.8200 (± 0.0350)
Precision - Under             0.8500
Precision - Over              0.8000
Recall - Under                0.8100
Recall - Over                 0.8300
======================================================================
```

---

## Files Generated

After running all evaluation steps, you'll have:

```
AIFootballPredictions/
├── metrics/
│   ├── E0_goals_metrics.json
│   ├── I1_goals_metrics.json
│   ├── ... (one for each league)
│   ├── model_comparison.csv
│   ├── goals_metrics_comparison.png
│   └── corners_metrics_comparison.png
├── backtest_results/
│   ├── E0_goals_backtest.json
│   ├── I1_goals_backtest.json
│   ├── ... (one for each league)
│   ├── backtest_comparison.csv
│   ├── goals_backtest_results.png
│   └── corners_backtest_results.png
├── metrics_history.json
├── metrics_history.csv
└── backtest_history.csv
```

---

## Key Improvements Made

1. **Feature Scaling** - StandardScaler added to both training scripts for better convergence
2. **Detailed Metrics** - Beyond accuracy: Precision, Recall, F1, ROC-AUC
3. **Real-World Testing** - Backtest on historical data with time-series validation
4. **Trend Analysis** - Track metrics across multiple training runs
5. **Visualizations** - PNG charts for easy interpretation
6. **Automated Pipeline** - All evaluation runs automatically as part of full pipeline

---

## Next Steps

1. Run the full pipeline: `./run_full_pipeline.sh`
2. Check `metrics/model_comparison.csv` for league performance
3. Review `backtest_results/backtest_comparison.csv` for real-world accuracy
4. View PNG visualizations for quick insights
5. Track `metrics_history.json` over time to see improvements

---

## Questions?

- Accuracy too low? Check `backtest_results/` to see where predictions fail
- Which league performs best? Check `model_comparison.csv`
- Are metrics improving? Check `metrics_history.csv` over time
- Need to debug? Individual JSON files have detailed per-fold metrics

