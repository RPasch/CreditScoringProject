# XGBoost vs Logistic Regression Apps

## Quick Comparison

| Feature | XGBoost App ⭐ | LogReg App |
|---------|---------------|------------|
| **Model Type** | Gradient Boosting (XGBoost) | Logistic Regression |
| **Explanation Method** | SHAP values | Coefficient × Feature |
| **Interpretability** | Complex (non-linear) | Simple (linear) |
| **Accuracy** | Higher | Lower |
| **Speed** | Slower | Faster |
| **Features Used** | 80+ features | 50 features |
| **Best For** | Production predictions | Understanding feature impact |

## When to Use Each App

### 🎯 Use XGBoost App (`xgb_app.py`) When:
- ✅ You need the **most accurate predictions**
- ✅ You want to see **complex feature interactions**
- ✅ You're making **real credit decisions**
- ✅ You need **SHAP explanations** for regulatory compliance
- ✅ You want to compare predictions with **true labels**

### 📊 Use LogReg App (`app.py`) When:
- ✅ You want **simple, linear interpretations**
- ✅ You need **faster predictions**
- ✅ You're **explaining to non-technical stakeholders**
- ✅ You want to understand **direct feature coefficients**
- ✅ You're doing **exploratory analysis**

## Key Differences

### 1. Explanation Method

**XGBoost (SHAP)**
```
Feature "EXT_SOURCE_MEAN" has SHAP value of -0.45
→ This specific value of 0.67 DECREASES default probability
→ Captures non-linear effects and interactions
```

**Logistic Regression (Coefficients)**
```
Feature "EXT_SOURCE_MEAN" has coefficient of -2.3
→ Higher values ALWAYS decrease default probability linearly
→ Simple multiplication: value × coefficient = contribution
```

### 2. Feature Selection

**XGBoost**
- Uses 80+ features
- Can handle redundant/correlated features
- Feature importance via SHAP

**Logistic Regression**
- Uses 50 carefully selected features
- Removes highly correlated features (>0.95)
- Feature importance via coefficients

### 3. Decision Threshold

**XGBoost**
- Default: 0.06 (optimized for recall)
- Catches more potential defaults
- Higher false positive rate

**Logistic Regression**
- Default: 0.5 (balanced)
- Conservative approach
- Lower false positive rate

## Performance Comparison

Based on validation data:

| Metric | XGBoost | LogReg |
|--------|---------|--------|
| **ROC-AUC** | ~0.76 | ~0.73 |
| **Precision @ 0.5** | Lower | Higher |
| **Recall @ 0.5** | Higher | Lower |
| **Interpretability** | Medium | High |
| **Training Time** | Hours | Minutes |
| **Inference Speed** | Slower | Faster |

## Recommendation

### 🏆 For Production Use
**Use XGBoost App** - Better performance, comprehensive SHAP explanations

### 📚 For Learning/Teaching
**Use LogReg App** - Easier to understand, direct feature impact

### 🔬 For Analysis
**Use Both** - Compare predictions and understand model differences

## Running Both Apps

You can run both apps simultaneously on different ports:

```bash
# Terminal 1: XGBoost on port 8501
streamlit run xgb_app.py

# Terminal 2: LogReg on port 8502
streamlit run app.py --server.port 8502
```

Then compare predictions side-by-side!

## Example Use Cases

### Use Case 1: Production Credit Decision
**App:** XGBoost
**Reason:** Highest accuracy, SHAP for compliance

### Use Case 2: Explaining to Loan Officer
**App:** Logistic Regression
**Reason:** Simple coefficient-based explanations

### Use Case 3: Model Comparison Research
**App:** Both
**Reason:** Understand prediction differences

### Use Case 4: Regulatory Audit
**App:** XGBoost
**Reason:** SHAP provides detailed, defensible explanations

### Use Case 5: Quick Screening
**App:** Logistic Regression
**Reason:** Faster predictions, good enough accuracy
