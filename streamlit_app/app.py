#!/usr/bin/env python3
"""
Lightweight Streamlit Frontend for Credit Scoring Model
Uses Logistic Regression model for predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
import requests
import json
import textwrap
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Credit Scoring Predictor",
    page_icon="💳",
    layout="wide"
)

# Configuration
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
RECENCY_SENTINEL = -999
API_KEY = os.getenv('OPENAI_API_KEY', '')  # Load from .env file

# Import helper functions
from logreg_utils import create_derived_features, load_model_and_data


@st.cache_resource
def initialize_model():
    """Load model and test data once"""
    try:
        model, features, test_df, X_test_processed = load_model_and_data()
        return model, features, test_df, X_test_processed
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None


def explain_prediction_streamlit(model, X_processed, idx, features):
    """Generate explanation for a single prediction"""
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    
    sample = X_processed[idx:idx+1]
    prob = model.predict_proba(sample)[0, 1]
    
    # Calculate contributions
    contributions = sample[0] * coef
    
    feature_info = []
    for i, feature in enumerate(features):
        feature_info.append({
            'feature': feature,
            'value': sample[0, i],
            'coefficient': coef[i],
            'contribution': contributions[i],
            'abs_contribution': abs(contributions[i])
        })
    
    explanation_df = pd.DataFrame(feature_info).sort_values('abs_contribution', ascending=False)
    
    return explanation_df, prob


def create_contribution_plot(explanation_df, prob, top_n=15):
    """Create contribution bar chart"""
    top_df = explanation_df.head(top_n)
    colors = ['#2ECC40' if x > 0 else '#FF4136' for x in top_df['contribution']]
    
    fig = go.Figure(go.Bar(
        x=top_df['contribution'].values,
        y=[f"{row['feature'][:30]}" for _, row in top_df.iterrows()],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in top_df['contribution'].values],
        textposition='outside'
    ))
    
    decision = "APPROVED" if prob >= 0.5 else "REJECTED"
    
    fig.update_layout(
        title=f"Feature Contributions - {decision} ({prob:.1%})",
        xaxis_title="Contribution to log-odds",
        yaxis_title="Feature",
        height=max(400, top_n * 30),
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='lightgray'),
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def get_ai_explanation_streamlit(explanation_df, prob, api_key):
    """Get AI explanation using OpenAI API"""
    decision = "APPROVED" if prob >= 0.5 else "REJECTED"
    
    # Get top factors
    top_positive = explanation_df[explanation_df['contribution'] > 0].head(3)
    top_negative = explanation_df[explanation_df['contribution'] < 0].head(3)
    
    prompt = f"""As a credit analyst, provide a brief 2-3 sentence explanation for this loan decision.

Decision: {decision} (Probability: {prob:.2%})

Key positive factors:
{top_positive[['feature', 'value', 'contribution']].to_string() if not top_positive.empty else 'None'}

Key negative factors:
{top_negative[['feature', 'value', 'contribution']].to_string() if not top_negative.empty else 'None'}

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
    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"


# Main App
def main():
    st.title("💳 Credit Scoring Predictor")
    st.markdown("---")
    
    # Load model and data
    with st.spinner("Loading model and data..."):
        model, features, test_df, X_test_processed = initialize_model()
    
    if model is None:
        st.error("Failed to load model. Please check configuration.")
        return
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    use_ai = st.sidebar.checkbox("Enable AI Explanations", value=False)
    if use_ai:
        api_key_input = st.sidebar.text_input(
            "OpenAI API Key", 
            value=API_KEY,
            type="password"
        )
    
    top_n_features = st.sidebar.slider("Features to Display", 5, 20, 10)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Select Application")
        
        # Get available IDs
        available_ids = test_df[ID_COL].tolist()
        
        selected_id = st.selectbox(
            "Application ID",
            options=available_ids,
            index=0
        )
        
        # Get index of selected ID
        idx = test_df[test_df[ID_COL] == selected_id].index[0]
        
        st.markdown(f"**Selected Index:** {idx}")
        st.markdown(f"**Total Applications:** {len(available_ids)}")
        
        # Prediction button
        predict_button = st.button("🎯 Make Prediction", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Application Features")
        
        # Display feature values for selected application
        feature_data = test_df.iloc[idx][features].to_frame()
        feature_data.columns = ['Value']
        
        # Format display
        st.dataframe(
            feature_data.head(20),
            use_container_width=True,
            height=400
        )
        
        if len(features) > 20:
            with st.expander(f"View all {len(features)} features"):
                st.dataframe(feature_data, use_container_width=True)
    
    # Prediction section
    if predict_button:
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        with st.spinner("Generating prediction..."):
            # Get explanation and prediction
            explanation_df, prob = explain_prediction_streamlit(
                model, X_test_processed, idx, features
            )
            
            decision = "APPROVED ✅" if prob >= 0.5 else "REJECTED ❌"
            decision_class = prob >= 0.5
        
        # Display prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Decision", 
                decision,
                delta=f"{(prob - 0.5) * 100:.1f}% from threshold"
            )
        
        with col2:
            st.metric(
                "Default Probability",
                f"{prob:.1%}"
            )
        
        with col3:
            confidence = max(prob, 1-prob)
            st.metric(
                "Confidence",
                f"{confidence:.1%}"
            )
        
        # Top contributing features
        st.subheader("🔍 Top Contributing Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Factors (Reduce Risk)**")
            positive_factors = explanation_df[explanation_df['contribution'] > 0].head(5)
            for _, row in positive_factors.iterrows():
                st.markdown(f"↑ **{row['feature'][:30]}**: {row['contribution']:.3f}")
        
        with col2:
            st.markdown("**Negative Factors (Increase Risk)**")
            negative_factors = explanation_df[explanation_df['contribution'] < 0].head(5)
            for _, row in negative_factors.iterrows():
                st.markdown(f"↓ **{row['feature'][:30]}**: {row['contribution']:.3f}")
        
        # Contribution plot
        st.subheader("📊 Feature Contribution Analysis")
        fig = create_contribution_plot(explanation_df, prob, top_n=top_n_features)
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Explanation
        if use_ai and api_key_input and api_key_input != "YOUR_OPENAI_API_KEY_HERE":
            st.subheader("🤖 AI-Generated Explanation")
            
            with st.spinner("Generating AI explanation..."):
                ai_explanation = get_ai_explanation_streamlit(
                    explanation_df, prob, api_key_input
                )
            
            st.info(ai_explanation)
        elif use_ai:
            st.warning("⚠️ Please enter a valid OpenAI API key in the sidebar to enable AI explanations.")
        
        # Detailed breakdown
        with st.expander("📋 View Detailed Feature Breakdown"):
            st.dataframe(
                explanation_df[['feature', 'value', 'coefficient', 'contribution']],
                use_container_width=True
            )


if __name__ == "__main__":
    main()
