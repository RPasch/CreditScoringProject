# Credit Scoring Streamlit Apps

Two lightweight web interfaces for credit scoring predictions.

## 🎯 Available Apps

### 1. **XGBoost Credit Scorer** (`xgb_app.py`) - RECOMMENDED
Advanced gradient boosting model with SHAP explanations

**Features:**
- 📋 Application selection from test dataset
- 📊 Sample data preview with all features
- 🎯 XGBoost model predictions
- 📈 SHAP feature importance analysis
- 🤖 AI-generated explanations (optional)
- ✅ True label comparison (when available)

### 2. **Logistic Regression Scorer** (`app.py`)
Interpretable linear model with coefficient-based explanations

**Features:**
- 📋 Application selection via dropdown
- 📊 Feature display and values
- 🎯 Logistic regression predictions
- 📈 Feature contribution charts
- 🤖 AI explanations (optional)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run an App

**Option A: XGBoost App (Recommended)**
```bash
streamlit run xgb_app.py
# OR
./run_xgb_app.sh
```

**Option B: Logistic Regression App**
```bash
streamlit run app.py
# OR
./run_app.sh
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Select an Application**: Choose an ID from the dropdown menu
2. **View Features**: Examine the applicant's feature values
3. **Make Prediction**: Click the "Make Prediction" button
4. **Review Results**:
   - Decision (Approved/Rejected)
   - Probability score
   - Top contributing features
   - Interactive contribution chart
   - AI explanation (if enabled)

## Configuration

### Enable AI Explanations

1. Check "Enable AI Explanations" in the sidebar
2. Enter your OpenAI API key
3. AI explanations will appear with predictions

### Adjust Display

- Use the sidebar slider to control number of features shown in charts
- Expand sections to view detailed breakdowns

## File Structure

```
streamlit_app/
├── xgb_app.py          # XGBoost Streamlit application ⭐
├── app.py              # Logistic Regression application
├── logreg_utils.py     # LogReg model utilities
├── run_xgb_app.sh      # XGBoost app launcher
├── run_app.sh          # LogReg app launcher
├── requirements.txt    # Python dependencies
├── test_setup.py       # Setup verification script
└── README.md           # This file
```

## Requirements

- Python 3.8+
- Trained models in `../models/` directory:
  - `xgboost_pipeline.pkl` and `xgboost_model.json`
  - `logreg_pipeline.pkl`
- Test data in `../data/wrangled_data/merged_test.csv`
- OpenAI API key (optional, for AI explanations)

## Notes

- Model and data are cached for performance
- First load may take a few seconds
- AI explanations require valid OpenAI API key
- All predictions are based on pre-loaded test data

## Troubleshooting

**Model not found**: Ensure you've run `logreg_model_notebook.py` first to train and save the model.

**Data not found**: Check that test data exists in `../data/wrangled_data/merged_test.csv`

**API errors**: Verify your OpenAI API key is valid and has available credits.
