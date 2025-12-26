# 📉 Customer Churn Prediction using Machine Learning

## 📌 Problem Statement
Customer churn is a major concern for subscription-based businesses.  
This project aims to **predict whether a customer will churn** based on demographic information, service usage, and billing behavior, enabling proactive retention strategies.

---

## 📊 Dataset
- **Source:** Telco Customer Churn Dataset (IBM / Kaggle)
- **Samples:** ~7,000 customers
- **Target Variable:** `Churn` (Yes / No)

### Feature Types
- **Numerical:** tenure, MonthlyCharges, TotalCharges  
- **Categorical:** Contract, InternetService, PaymentMethod, TechSupport, etc.

> Note: The `customerID` feature was removed as it is a unique identifier and does not carry predictive information.

---

## 🛠️ Methodology

### 1. Data Cleaning
- Converted `TotalCharges` from string to numeric
- Handled missing values logically (new customers → `TotalCharges = 0`)
- Removed identifier leakage (`customerID`)

### 2. Preprocessing
- Numerical features scaled using `StandardScaler`
- Categorical features encoded using `OneHotEncoder`
- All preprocessing handled using `ColumnTransformer`
- Prevented data leakage using `Pipeline`

### 3. Models Evaluated
- Logistic Regression (baseline, interpretable)
- Random Forest Classifier (non-linear comparison)

---

## 📈 Evaluation Strategy
- Train–test split (80–20) with stratification
- 5-fold **Stratified Cross-Validation**
- Primary metric: **ROC–AUC**
- Additional metrics:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 📊 Results

### Logistic Regression (Final Model)
- **Test ROC–AUC:** ~0.84
- **Cross-validation mean ROC–AUC:** ~0.845
- **Cross-validation standard deviation:** ~0.013

#### Threshold Tuning
- Default threshold (0.5): Recall ≈ 0.56
- Tuned threshold (0.4): Recall ≈ 0.67

### Random Forest (Comparison)
- **Cross-validation mean ROC–AUC:** ~0.829
- No improvement over Logistic Regression
- Higher complexity without performance gain

---

## 🔍 Key Insights (Feature Importance)

Feature importance analysis using Logistic Regression coefficients revealed:

### Lower churn risk
- Longer customer tenure
- Two-year contracts
- DSL or no-internet plans

### Higher churn risk
- Month-to-month contracts
- Fiber optic internet service
- Higher accumulated charges

These insights align strongly with real-world customer behavior patterns.

---

## ✅ Final Model Selection
**Logistic Regression** was selected due to:
- Better generalization performance
- Lower variance across folds
- High interpretability
- Business-aligned decision making

---

## 📁 Project Structure

```text
customer_churn_ml/
│
├── data/
│   └── raw/
│       └── telco_customer_churn.csv
│
├── notebooks/
│   └── customer_churn.ipynb
│
├── models/
│   └── customer_churn_logistic_pipeline.pkl
│
└── README.md
```
## 🚀 Skills Demonstrated
- Classification modeling
- Handling class imbalance
- ML pipelines & preprocessing
- Threshold tuning
- Cross-validation
- Model comparison
- Feature interpretation
- Business-driven ML decisions

---

## 🧰 Tools & Libraries
- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib / Seaborn
- Git & GitHub

---

## 📌 Future Improvements
- Deploy model using Streamlit
- Hyperparameter tuning
- Cost-sensitive learning
- Real-time churn scoring API
