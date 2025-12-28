#!/usr/bin/env python3
"""
Test script to verify XGBoost Streamlit app setup
"""

import sys
import os

def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")
    
    packages = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('xgboost', 'xgboost'),
        ('shap', 'shap'),
        ('plotly', 'plotly.graph_objects'),
        ('joblib', 'joblib'),
        ('dotenv', 'python-dotenv'),
        ('requests', 'requests'),
    ]
    
    all_ok = True
    for package_name, import_name in packages:
        try:
            __import__(import_name.split('.')[0])
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Run: pip install {package_name}")
            all_ok = False
    
    return all_ok


def test_files():
    """Test that required files exist"""
    print("\nTesting file structure...")
    
    files_to_check = [
        ('../models/xgboost_pipeline.pkl', 'XGBoost pipeline'),
        ('../models/xgboost_model.json', 'XGBoost model'),
        ('../data/wrangled_data/merged_test.csv', 'Test data'),
        ('../.env', 'Environment file (optional)'),
    ]
    
    all_ok = True
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {description}: {file_path}")
        else:
            if 'optional' in description.lower():
                print(f"⚠️  {description}: {file_path} (not found, but optional)")
            else:
                print(f"❌ {description}: {file_path}")
                all_ok = False
    
    return all_ok


def test_model_loading():
    """Test that XGBoost model can be loaded"""
    print("\nTesting XGBoost model loading...")
    try:
        import joblib
        import xgboost as xgb
        import pandas as pd
        
        # Load pipeline
        pipeline = joblib.load('../models/xgboost_pipeline.pkl')
        model = pipeline['model']
        top_features = pipeline['top_features']
        
        # Load model weights
        model.load_model('../models/xgboost_model.json')
        
        print(f"✅ Model loaded successfully")
        print(f"   Top features: {len(top_features)}")
        
        # Load test data
        test_df = pd.read_csv('../data/wrangled_data/merged_test.csv')
        print(f"✅ Test data loaded: {test_df.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def test_api_key():
    """Test if API key is available"""
    print("\nTesting API key configuration...")
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key and api_key != 'YOUR_OPENAI_API_KEY_HERE':
            print(f"✅ API key found: {api_key[:20]}...")
        else:
            print("⚠️  No valid API key found (AI explanations will be disabled)")
            print("   Set OPENAI_API_KEY in ../.env file")
        
        return True
    except Exception as e:
        print(f"⚠️  Could not check API key: {str(e)}")
        return True  # Non-critical


def main():
    print("="*70)
    print("XGBOOST STREAMLIT APP - SETUP TEST")
    print("="*70)
    print()
    
    # Run tests
    imports_ok = test_imports()
    if not imports_ok:
        print("\n❌ Import test failed. Install missing packages:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    files_ok = test_files()
    if not files_ok:
        print("\n❌ File structure test failed.")
        print("   Make sure you've trained the XGBoost model first.")
        sys.exit(1)
    
    model_ok = test_model_loading()
    if not model_ok:
        print("\n❌ Model loading test failed.")
        sys.exit(1)
    
    test_api_key()
    
    print("\n" + "="*70)
    print("✅ ALL CRITICAL TESTS PASSED!")
    print("="*70)
    print("\nYou can now run the XGBoost app:")
    print("  streamlit run xgb_app.py")
    print("  OR")
    print("  ./run_xgb_app.sh")
    print()
    print("💡 Tips:")
    print("  - Use threshold slider to adjust decision boundary")
    print("  - Enable AI explanations for natural language summaries")
    print("  - Compare predictions with true labels when available")
    print()


if __name__ == "__main__":
    main()
