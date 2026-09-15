#!/bin/bash
# Quick run script for the Credit Risk Scoring dashboard

set -e
echo "Starting Credit Risk Scoring dashboard..."
echo ""

if [ ! -f "scripts/app.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    STREAMLIT=".venv/bin/streamlit"
else
    PYTHON="python3"
    STREAMLIT="streamlit"
fi

if ! "$PYTHON" -c "import streamlit, lightgbm, shap" 2>/dev/null; then
    echo "Installing dependencies..."
    "$PYTHON" -m pip install -r requirements.txt
    echo ""
fi

if [ ! -f "artifacts/models/lightgbm.joblib" ]; then
    echo "Training models (first run)..."
    "$PYTHON" scripts/train.py
    echo ""
fi

echo "Launching dashboard..."
"$STREAMLIT" run scripts/app.py
