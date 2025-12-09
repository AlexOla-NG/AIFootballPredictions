#!/bin/bash

# AIFootballPredictions - Full Pipeline Execution Script
# This script runs all steps from data acquisition to predictions in sequence

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_PATH=".venv"
LEAGUES="E0 I1 SP1 F1 D1"
SEASONS="2526 2425 2324 2223"
NUM_FEATURES=20
CLUSTERING_THRESHOLD=0.5
N_SPLITS=10
VOTING="soft"
METRIC_CHOICE="accuracy"

# Directories
RAW_DATA_DIR="data/raw"
PROCESSED_DATA_DIR="data/processed"
MODELS_DIR="models"
NEXT_MATCHES_FILE="data/next_matches.json"
PREDICTIONS_FILE="final_predictions_with_corners.txt"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Virtual environment not found. Please create it first with:${NC}"
    echo "conda env create -f conda/aifootball_predictions.yaml"
    echo "conda activate aifootball_predictions"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}AIFootballPredictions Pipeline${NC}"
echo -e "${BLUE}================================${NC}\n"

# Step 0: Install missing dependencies
echo -e "${GREEN}Step 0: Installing dependencies from requirements.txt...${NC}"
.venv/bin/pip install -r requirements.txt -q

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}\n"
else
    echo -e "${YELLOW}✗ Failed to install dependencies${NC}"
    exit 1
fi

# Step 1: Data Acquisition
echo -e "${GREEN}Step 1: Acquiring historical match data...${NC}"
.venv/bin/python scripts/data_acquisition.py \
    --leagues $LEAGUES \
    --seasons $SEASONS \
    --raw_data_output_dir $RAW_DATA_DIR

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data acquisition completed${NC}\n"
else
    echo -e "${YELLOW}✗ Data acquisition failed${NC}"
    exit 1
fi

# Step 2: Data Preprocessing
echo -e "${GREEN}Step 2: Preprocessing data and engineering features...${NC}"
.venv/bin/python scripts/data_preprocessing.py \
    --raw_data_input_dir $RAW_DATA_DIR \
    --processed_data_output_dir $PROCESSED_DATA_DIR \
    --num_features $NUM_FEATURES \
    --clustering_threshold $CLUSTERING_THRESHOLD

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data preprocessing completed${NC}\n"
else
    echo -e "${YELLOW}✗ Data preprocessing failed${NC}"
    exit 1
fi

# Step 3: Train Goals Models
echo -e "${GREEN}Step 3: Training goals prediction models (Over/Under 2.5)...${NC}"
.venv/bin/python scripts/train_models.py \
    --processed_data_input_dir $PROCESSED_DATA_DIR \
    --trained_models_output_dir $MODELS_DIR \
    --metric_choice $METRIC_CHOICE \
    --n_splits $N_SPLITS \
    --voting $VOTING

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Goals model training completed${NC}\n"
else
    echo -e "${YELLOW}✗ Goals model training failed${NC}"
    exit 1
fi

# Step 4: Train Corner Models
echo -e "${GREEN}Step 4: Training corner prediction models (Over/Under 10.5)...${NC}"
.venv/bin/python scripts/train_corner_models.py \
    --processed_data_input_dir $PROCESSED_DATA_DIR \
    --trained_models_output_dir $MODELS_DIR \
    --metric_choice $METRIC_CHOICE \
    --n_splits $N_SPLITS \
    --voting $VOTING

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Corner model training completed${NC}\n"
else
    echo -e "${YELLOW}✗ Corner model training failed${NC}"
    exit 1
fi

# Step 5: Acquire Next Matches
echo -e "${GREEN}Step 5: Acquiring next matches data...${NC}"
.venv/bin/python scripts/acquire_next_matches.py \
    --get_teams_names_dir $PROCESSED_DATA_DIR \
    --next_matches_output_file $NEXT_MATCHES_FILE

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Next matches acquisition completed${NC}\n"
else
    echo -e "${YELLOW}✗ Next matches acquisition failed${NC}"
    exit 1
fi

# Step 6: Make Predictions
echo -e "${GREEN}Step 6: Making predictions (goals + corners)...${NC}"
.venv/bin/python scripts/make_predictions_enhanced.py \
    --input_leagues_models_dir $MODELS_DIR \
    --input_data_predict_dir $PROCESSED_DATA_DIR \
    --final_predictions_out_file $PREDICTIONS_FILE \
    --next_matches $NEXT_MATCHES_FILE \
    --predict_corners

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Predictions completed${NC}\n"
else
    echo -e "${YELLOW}✗ Predictions failed${NC}"
    exit 1
fi

# Step 7: Evaluate Models
echo -e "${GREEN}Step 7: Evaluating model performance...${NC}"
.venv/bin/python scripts/evaluate_models.py \
    --processed_data_input_dir $PROCESSED_DATA_DIR \
    --trained_models_output_dir $MODELS_DIR \
    --metrics_output_dir metrics

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Model evaluation completed${NC}\n"
else
    echo -e "${YELLOW}✗ Model evaluation failed${NC}"
    exit 1
fi

# Step 8: Backtest Models
echo -e "${GREEN}Step 8: Backtesting model predictions...${NC}"
.venv/bin/python scripts/backtest_predictions.py \
    --processed_data_input_dir $PROCESSED_DATA_DIR \
    --trained_models_output_dir $MODELS_DIR \
    --backtest_output_dir backtest_results

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Backtest completed${NC}\n"
else
    echo -e "${YELLOW}✗ Backtest failed${NC}"
    exit 1
fi

# Step 9: Track Metrics
echo -e "${GREEN}Step 9: Tracking metrics over time...${NC}"
.venv/bin/python scripts/track_metrics.py \
    --metrics_dir metrics \
    --backtest_dir backtest_results \
    --output_file metrics_history.json

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Metrics tracking completed${NC}\n"
else
    echo -e "${YELLOW}✗ Metrics tracking failed${NC}"
    exit 1
fi

# Summary
echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}✓ Full pipeline completed successfully!${NC}"
echo -e "${BLUE}================================${NC}\n"
echo -e "Results saved to:"
echo -e "  ${YELLOW}Predictions:${NC} $PREDICTIONS_FILE"
echo -e "  ${YELLOW}Models:${NC} $MODELS_DIR/"
echo -e "  ${YELLOW}Evaluation Metrics:${NC} metrics/"
echo -e "  ${YELLOW}Backtest Results:${NC} backtest_results/"
echo -e "  ${YELLOW}Metrics History:${NC} metrics_history.json\n"
