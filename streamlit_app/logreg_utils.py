#!/usr/bin/env python3
"""
Utility functions for Streamlit app
Handles model loading and data processing
"""

import pandas as pd
import numpy as np
import joblib
import os

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
RECENCY_SENTINEL = -999


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for logistic regression model"""
    df = df.copy()
    eps = 1e-6
    
    if all(c in df.columns for c in ["AMT_ANNUITY", "AMT_INCOME_TOTAL"]):
        df["DTI_ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)
        
    if all(c in df.columns for c in ["AMT_CREDIT", "AMT_INCOME_TOTAL"]):
        df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)
        
    if all(c in df.columns for c in ["AMT_CREDIT", "AMT_GOODS_PRICE"]):
        df["LTV_CREDIT_TO_GOODS"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + eps)
    
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365.25
        
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_UNKNOWN"] = (df["DAYS_EMPLOYED"] >= 365000).astype(int)
        emp = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] < 365000, np.nan)
        df["EMPLOYED_YEARS"] = (-emp / 365.25).fillna(0)
        
    # External source combined score
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ext_present = [c for c in ext_cols if c in df.columns]
    if ext_present:
        df["EXT_SOURCE_MEAN"] = df[ext_present].mean(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_present].max(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_present].min(axis=1)
    
    return df


def load_model_and_data():
    """Load trained model and test data"""
    # Get base directory (works both locally and in deployment)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try multiple possible paths for models
    model_paths = [
        os.path.join(base_dir, 'Notebooks/models/logreg_pipeline.pkl'),
        os.path.join(base_dir, 'models/logreg_pipeline.pkl'),
        '../Notebooks/models/logreg_pipeline.pkl',
    ]
    
    pipeline_path = None
    for path in model_paths:
        if os.path.exists(path):
            pipeline_path = path
            break
    
    if not pipeline_path:
        raise FileNotFoundError(f"Could not find LogReg model. Tried: {model_paths}")
    
    # Load pipeline
    pipeline = joblib.load(pipeline_path)
    
    model = pipeline['model']
    scaler = pipeline['scaler']
    features = pipeline['features']
    categorical_features = pipeline.get('categorical_features', [])
    
    # Try multiple paths for test data
    data_paths = [
        os.path.join(base_dir, 'data/wrangled_data/merged_test.csv'),
        '../data/wrangled_data/merged_test.csv',
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if not data_path:
        raise FileNotFoundError(f"Could not find test data. Tried: {data_paths}")
    
    # Load test data
    test_df = pd.read_csv(data_path)
    
    # Apply feature engineering
    test_df = create_derived_features(test_df)
    
    # Save IDs
    test_ids = test_df[ID_COL].values
    
    # Remove ID and target if present
    feature_cols = [c for c in test_df.columns if c not in [ID_COL, TARGET_COL]]
    
    # Select only features that model expects
    available_features = [f for f in features if f in test_df.columns]
    missing_features = [f for f in features if f not in test_df.columns]
    
    if missing_features:
        print(f"Warning: {len(missing_features)} features missing from test data")
        # Add missing features with default values
        for feat in missing_features:
            if feat in categorical_features:
                test_df[feat] = 'UNKNOWN'
            else:
                test_df[feat] = 0
    
    # Select features in correct order
    X_test = test_df[features].copy()
    
    # Handle categorical encoding
    if categorical_features:
        for col in categorical_features:
            if col in X_test.columns:
                if X_test[col].dtype == 'object':
                    X_test[col] = X_test[col].astype('category').cat.codes
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    return model, features, test_df, X_test_scaled
