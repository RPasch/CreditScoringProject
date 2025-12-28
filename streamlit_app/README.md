# Credit Scoring Streamlit App

A lightweight web interface for credit scoring predictions using Logistic Regression.

## Features

- 📋 **Application Selection**: Choose from test dataset via dropdown
- 📊 **Feature Display**: View all features for selected application
- 🎯 **Real-time Predictions**: Get instant credit decisions
- 📈 **Visual Explanations**: Interactive feature contribution charts
- 🤖 **AI Explanations**: Natural language decision summaries (optional)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
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
├── app.py              # Main Streamlit application
├── logreg_utils.py     # Model loading and data processing
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Requirements

- Python 3.8+
- Trained logistic regression model in `../Notebooks/models/`
- Test data in `../data/wrangled_data/merged_test.csv`

## Notes

- Model and data are cached for performance
- First load may take a few seconds
- AI explanations require valid OpenAI API key
- All predictions are based on pre-loaded test data

## Troubleshooting

**Model not found**: Ensure you've run `logreg_model_notebook.py` first to train and save the model.

**Data not found**: Check that test data exists in `../data/wrangled_data/merged_test.csv`

**API errors**: Verify your OpenAI API key is valid and has available credits.
