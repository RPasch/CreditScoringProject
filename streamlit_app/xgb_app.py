#!/usr/bin/env python3
"""
XGBoost Credit Scoring Streamlit App
Features: Sample selection, predictions, SHAP explanations, AI summaries
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import plotly.graph_objects as go
import requests
import json
import textwrap
import shap
from dotenv import load_dotenv

# Suppress pandas warnings
pd.options.mode.chained_assignment = None

# Load environment variables (for local development)
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="XGBoost Credit Scorer",
    page_icon="🎯",
    layout="wide"
)

# Configuration
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# API Key: Try Streamlit secrets first (production), then env vars (local)
try:
    API_KEY = st.secrets.get("OPENAI_API_KEY", "")
except:
    API_KEY = os.getenv('OPENAI_API_KEY', '')


def create_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for XGBoost model"""
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
    
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ext_present = [c for c in ext_cols if c in df.columns]
    if ext_present:
        df["EXT_SOURCE_MEAN"] = df[ext_present].mean(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_present].max(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_present].min(axis=1)
    
    return df


@st.cache_resource
def load_model_and_data():
    """Load XGBoost model and test data"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load pipeline from models directory (relative to this script)
        pipeline_path = os.path.join(script_dir, 'models/xgboost_pipeline.pkl')
        
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Model not found at: {pipeline_path}")
        
        # Load pipeline
        pipeline = joblib.load(pipeline_path)
        
        model = pipeline['model']
        features = pipeline['features']
        numeric_features = pipeline['numeric_features']
        categorical_features = pipeline['categorical_features']
        top_features = pipeline['top_features']
        
        # Load the XGBoost model JSON
        model_json_path = os.path.join(script_dir, 'models/xgboost_model.json')
        
        if not os.path.exists(model_json_path):
            raise FileNotFoundError(f"Model JSON not found at: {model_json_path}")
        
        model.load_model(model_json_path)
        
        # Load test data from repo root
        repo_root = os.path.dirname(script_dir)
        test_data_path = os.path.join(repo_root, 'data/wrangled_data/merged_test.csv')
        
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data not found at: {test_data_path}")
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        
        # Apply feature engineering
        test_df = create_xgb_features(test_df)
        
        # Prepare features
        X_test = test_df[features].copy()
        
        # Encode categoricals
        for col in categorical_features:
            if col in X_test.columns:
                X_test[col] = pd.Categorical(X_test[col]).codes
        
        # Select top features
        X_test_selected = X_test[top_features].copy()
        
        return model, top_features, test_df, X_test_selected
        
    except Exception as e:
        st.error("⚠️ Unable to load the credit scoring model. Please contact support.")
        return None, None, None, None


def explain_xgb_prediction(model, X_data, idx, top_features, top_n=15):
    """Generate SHAP explanation for prediction"""
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Get SHAP values for single sample
    sample = X_data[idx:idx+1]
    shap_values = explainer.shap_values(sample)
    
    # Get prediction
    dtest = xgb.DMatrix(sample, feature_names=top_features)
    pred_prob = model.predict(dtest)[0]
    
    # Create explanation dataframe
    explanation_data = []
    for i, feature in enumerate(top_features):
        explanation_data.append({
            'feature': feature,
            'value': sample[0, i],
            'shap_value': shap_values[0][i],
            'abs_shap': abs(shap_values[0][i])
        })
    
    explanation_df = pd.DataFrame(explanation_data).sort_values('abs_shap', ascending=False)
    
    return explanation_df, pred_prob, shap_values[0], explainer.expected_value


def create_shap_plot(explanation_df, pred_prob, top_n=15):
    """Create SHAP waterfall plot"""
    top_df = explanation_df.head(top_n)
    colors = ['#2ECC40' if x > 0 else '#FF4136' for x in top_df['shap_value']]
    
    fig = go.Figure(go.Bar(
        x=top_df['shap_value'].values,
        y=[f"{row['feature'][:40]}" for _, row in top_df.iterrows()],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.4f}" for v in top_df['shap_value'].values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>'
    ))
    
    decision = "DEFAULT" if pred_prob >= 0.06 else "APPROVED"
    
    fig.update_layout(
        title=f"SHAP Feature Contributions - {decision} ({pred_prob:.1%})",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis_title="Feature",
        height=max(400, top_n * 30),
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='lightgray'),
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def get_ai_explanation(explanation_df, pred_prob, api_key):
    """Get AI-generated explanation"""
    decision = "DEFAULT" if pred_prob >= 0.06 else "APPROVED"
    
    top_positive = explanation_df[explanation_df['shap_value'] > 0].head(3)
    top_negative = explanation_df[explanation_df['shap_value'] < 0].head(3)
    
    prompt = f"""As a credit analyst, provide a brief 2-3 sentence explanation for this loan decision.

Decision: {decision} (Default Probability: {pred_prob:.2%})

Key factors increasing default risk:
{top_positive[['feature', 'value', 'shap_value']].to_string() if not top_positive.empty else 'None'}

Key factors decreasing default risk:
{top_negative[['feature', 'value', 'shap_value']].to_string() if not top_negative.empty else 'None'}

Write a concise explanation focusing on the main reasons for the {decision.lower()} decision."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a credit risk analyst providing brief, clear explanations."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception:
        return "⚠️ AI explanation is temporarily unavailable. Please check your API key or try again later."


# Main App
def main():
    st.title("🎯 XGBoost Credit Scoring System")
    st.markdown("---")
    
    # Load model and data
    with st.spinner("Loading XGBoost model and data..."):
        model, top_features, test_df, X_test_selected = load_model_and_data()
    
    if model is None:
        st.error("Failed to load model. Please check configuration.")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    use_ai = st.sidebar.checkbox("Enable AI Explanations", value=False)
    
    if use_ai:
        api_key_input = st.sidebar.text_input(
            "OpenAI API Key", 
            value=API_KEY,
            type="password"
        )
    
    top_n_features = st.sidebar.slider("Top Features to Display", 5, 25, 15)
    threshold = st.sidebar.slider("Decision Threshold", 0.01, 0.50, 0.06, 0.01)
    
    # Main layout
    st.subheader("📋 Select Application")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Get available IDs
        available_ids = test_df[ID_COL].tolist()
        
        selected_id = st.selectbox(
            "Application ID",
            options=available_ids,
            index=0
        )
        
        # Get index
        idx = test_df[test_df[ID_COL] == selected_id].index[0]
        
        st.markdown(f"**Selected Index:** {idx}")
        st.markdown(f"**Total Applications:** {len(available_ids)}")
        
        # Show true label if available
        if TARGET_COL in test_df.columns:
            true_label = test_df.loc[idx, TARGET_COL]
            label_text = "DEFAULT ❌" if true_label == 1 else "APPROVED ✅"
            st.markdown(f"**True Label:** {label_text}")
        
        # Predict button
        predict_button = st.button("🎯 Run Prediction", type="primary", width="stretch")
    
    with col2:
        st.markdown("### 📊 Sample Data Preview")
        
        # Show sample row
        sample_data = test_df.iloc[idx:idx+1].T
        sample_data.columns = ['Value']
        
        st.dataframe(
            sample_data.head(20),
            width="stretch",
            height=400
        )
        
        with st.expander(f"View all {len(test_df.columns)} columns"):
            st.dataframe(sample_data, width="stretch")
    
    # Prediction section
    if predict_button:
        try:
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            with st.spinner("Running XGBoost inference..."):
                # Get explanation
                explanation_df, pred_prob, shap_values, base_value = explain_xgb_prediction(
                    model, X_test_selected.values, idx, top_features, top_n=top_n_features
                )
                
                decision = "DEFAULT ❌" if pred_prob >= threshold else "APPROVED ✅"
                decision_class = pred_prob >= threshold
        except Exception:
            st.error("⚠️ Unable to generate prediction. Please try selecting a different application.")
            st.stop()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Decision", 
                decision,
                delta=f"{(pred_prob - threshold) * 100:.1f}% from threshold"
            )
        
        with col2:
            st.metric(
                "Default Probability",
                f"{pred_prob:.1%}"
            )
        
        with col3:
            confidence = max(pred_prob, 1-pred_prob)
            st.metric(
                "Confidence",
                f"{confidence:.1%}"
            )
        
        with col4:
            st.metric(
                "Threshold",
                f"{threshold:.1%}"
            )
        
        # Check if prediction matches true label
        if TARGET_COL in test_df.columns:
            true_label = test_df.loc[idx, TARGET_COL]
            correct = (decision_class and true_label == 1) or (not decision_class and true_label == 0)
            
            if correct:
                st.success(f"✅ Prediction matches true label!")
            else:
                if true_label == 1 and not decision_class:
                    st.error("❌ Missed Default (False Negative)")
                else:
                    st.warning("⚠️ False Positive (Rejected good applicant)")
        
        # Top factors
        st.subheader("🔍 Top Contributing Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Factors Increasing Default Risk** ⬆️")
            positive_factors = explanation_df[explanation_df['shap_value'] > 0].head(5)
            for _, row in positive_factors.iterrows():
                st.markdown(f"↑ **{row['feature'][:35]}**: {row['shap_value']:.4f}")
        
        with col2:
            st.markdown("**Factors Decreasing Default Risk** ⬇️")
            negative_factors = explanation_df[explanation_df['shap_value'] < 0].head(5)
            for _, row in negative_factors.iterrows():
                st.markdown(f"↓ **{row['feature'][:35]}**: {row['shap_value']:.4f}")
        
        # SHAP Plot
        st.subheader("📊 SHAP Feature Importance Analysis")
        fig = create_shap_plot(explanation_df, pred_prob, top_n=top_n_features)
        st.plotly_chart(fig, width="stretch")
        
        # Model interpretation
        with st.expander("ℹ️ Understanding SHAP Values"):
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** shows how each feature contributes to the prediction:
            
            - **Positive SHAP values** (green): Increase default probability
            - **Negative SHAP values** (red): Decrease default probability
            - **Larger absolute values**: Stronger impact on this specific prediction
            - **Base value**: Average model prediction across all data
            
            The sum of all SHAP values + base value = final prediction
            """)
        
        # AI Explanation
        if use_ai and api_key_input and api_key_input != "YOUR_OPENAI_API_KEY_HERE":
            st.subheader("🤖 AI-Generated Explanation")
            
            with st.spinner("Generating AI explanation..."):
                ai_explanation = get_ai_explanation(
                    explanation_df, pred_prob, api_key_input
                )
            
            st.info(ai_explanation)
        elif use_ai:
            st.warning("⚠️ Please enter a valid OpenAI API key in the sidebar to enable AI explanations.")
        
        # Detailed breakdown
        with st.expander("View All Features and SHAP Values"):
            display_df = explanation_df[['feature', 'value', 'shap_value', 'abs_shap']].copy()
            
            # Create color coding for SHAP values
            colors = ['rgba(255, 100, 100, 0.3)' if x > 0 else 'rgba(100, 255, 100, 0.3)' 
                      for x in display_df['shap_value']]
            
            # Create Plotly table
            fig_table = go.Figure(data=[go.Table(
                header=dict(
                    values=['Feature', 'Value', 'SHAP Value', 'Impact Direction'],
                    fill_color='lightgray',
                    align='left',
                    font=dict(size=12, color='black')
                ),
                cells=dict(
                    values=[
                        display_df['feature'],
                        display_df['value'].round(4),
                        display_df['shap_value'].round(4),
                        ['Increases Risk' if x > 0 else 'Decreases Risk' for x in display_df['shap_value']]
                    ],
                    fill_color=[['white']*len(display_df), 
                                ['white']*len(display_df),
                                colors,
                                ['white']*len(display_df)],
                    align='left',
                    font=dict(size=11)
                )
            )])
            
            fig_table.update_layout(
                height=min(400, 30 * len(display_df) + 50),
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig_table, width="stretch")
        
        # Feature values for this sample
        with st.expander("🔢 View Feature Values for This Sample"):
            feature_values = pd.DataFrame({
                'Feature': top_features,
                'Value': X_test_selected.iloc[idx].values
            })
            st.dataframe(feature_values, width="stretch", height=400)


if __name__ == "__main__":
    main()
