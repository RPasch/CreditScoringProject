#!/usr/bin/env python3
"""
Test script to verify setup before running Streamlit app
"""

import sys
import os

def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")
    try:
        import streamlit as st
        print("✅ streamlit")
    except ImportError:
        print("❌ streamlit - Run: pip install streamlit")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError:
        print("❌ pandas")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ plotly")
    except ImportError:
        print("❌ plotly")
        return False
    
    try:
        import joblib
        print("✅ joblib")
    except ImportError:
        print("❌ joblib")
        return False
    
    try:
        import requests
        print("✅ requests")
    except ImportError:
        print("❌ requests")
        return False
    
    return True


def test_files():
    """Test that required files exist"""
    print("\nTesting file structure...")
    
    # Check model file
    model_path = '../Notebooks/models/logreg_pipeline.pkl'
    if os.path.exists(model_path):
        print(f"✅ Model found: {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
        print("   Run logreg_model_notebook.py first to train the model")
        return False
    
    # Check test data
    data_path = '../data/wrangled_data/merged_test.csv'
    if os.path.exists(data_path):
        print(f"✅ Test data found: {data_path}")
    else:
        print(f"❌ Test data not found: {data_path}")
        return False
    
    return True


def test_model_loading():
    """Test that model can be loaded"""
    print("\nTesting model loading...")
    try:
        from logreg_utils import load_model_and_data
        model, features, test_df, X_test_processed = load_model_and_data()
        
        print(f"✅ Model loaded successfully")
        print(f"   Features: {len(features)}")
        print(f"   Test samples: {len(test_df)}")
        print(f"   Processed shape: {X_test_processed.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def main():
    print("="*60)
    print("CREDIT SCORING STREAMLIT APP - SETUP TEST")
    print("="*60)
    print()
    
    # Run tests
    imports_ok = test_imports()
    if not imports_ok:
        print("\n❌ Import test failed. Install missing packages:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    files_ok = test_files()
    if not files_ok:
        print("\n❌ File structure test failed. Check file paths.")
        sys.exit(1)
    
    model_ok = test_model_loading()
    if not model_ok:
        print("\n❌ Model loading test failed.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYou can now run the app:")
    print("  streamlit run app.py")
    print("  OR")
    print("  ./run_app.sh")
    print()


if __name__ == "__main__":
    main()
