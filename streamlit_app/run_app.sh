#!/bin/bash
# Quick launcher for Credit Scoring Streamlit App

echo "🚀 Starting Credit Scoring App..."
echo ""
echo "📋 Checking dependencies..."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ Starting app on http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run app.py
