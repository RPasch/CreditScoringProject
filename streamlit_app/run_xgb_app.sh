#!/bin/bash
# Quick launcher for XGBoost Credit Scoring Streamlit App

echo "🎯 Starting XGBoost Credit Scoring App..."
echo ""
echo "📋 Checking dependencies..."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ Starting XGBoost app on http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run xgb_app.py
