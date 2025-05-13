# 💼 BankIQ: Loan Default Risk Prediction & Explainability

A complete end-to-end data science project simulating a bank's customer base, engineering financial and behavioral features, predicting loan default risk using interpretable ML techniques like SHAP, and visualizing predictions through an interactive Streamlit app.

---

## 🚀 Overview

**Goal:** Build a comprehensive pipeline to simulate, analyze, and predict loan default risk — then explain the predictions using SHAP and deploy insights through an interactive UI for stakeholders.

---

## 📊 Dataset (Synthetic)

Simulated using `Faker`, covering:
- Customer demographics (`Age`, `Income`, `CreditScore`)
- Product and loan behavior (`LoanAmount`, `EMI`, `Support interactions`)
- Campaign responses and engagement

---

## 🧠 Key Features Engineered

| Feature | Description |
|--------|-------------|
| `ProductEngagementScore` | Ratio of active products to total |
| `LoanBurdenScore`        | AvgLoanAmount × AvgEMItoIncomeRatio |
| `AvgEMItoIncomeRatio`    | EMI divided by income |
| `SupportFrequency`       | Number of support tickets |
| `HighRiskLoan` (Target)  | 1 if loan interest > 10% |

---

## ⚙️ Modeling

| Model | Description |
|-------|-------------|
| `Logistic Regression (Balanced)` | Tuned threshold for higher recall |
| `XGBoost`                        | Boosted tree model with `scale_pos_weight` |

---

## 📈 Model Performance (XGBoost)

- **Recall (High Risk):** 21%
- **Precision (High Risk):** 25%
- **ROC AUC:** 0.87

> With SHAP-based explainability, we interpret *why* the model flags someone as risky.

---

## 🔍 SHAP Insights

### Top Predictors:
1. `AvgEMItoIncomeRatio`
2. `AvgLoanAmount`
3. `Income`
4. `LoanBurdenScore`
5. `CreditScore`

### Local Force Plot:
Explains individual prediction breakdown via Streamlit app.

---

## 🌐 Interactive App (Streamlit)

### Preview:
- Select a customer and get real-time risk prediction
- Visualize SHAP force plot for explainability
- View summary plots globally

### Run Locally:
```bash
streamlit run script/streamlit_app.py
```

---

## 📁 Repository Structure

```
BankIQ/
├── data/
│   └── processed/
│       └── enhanced_customers.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_XGBoost.ipynb
│   └── 04_Explainability_SHAP.ipynb
├── scripts/
│   ├── 01_data_generation.py
│   ├── 02_feature_engineering.py
│   └── data_cleaning.py
|   └── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## 🧪 How to Run

```bash
git clone https://github.com/<your-username>/BankIQ.git
cd BankIQ

# Install dependencies
pip install -r requirements.txt

# Run feature engineering
python scripts/02_feature_engineering.py

# Explore via Jupyter or launch Streamlit app
jupyter notebook
streamlit run scripts/streamlit_app.py
```

---

## 🎯 Use Case Alignment

✅ Credit Risk  
✅ Loan Approval Scoring  
✅ Customer Engagement Segmentation  
✅ Explainable AI (SHAP)  
✅ Data Storytelling through Dashboards

---

## 🙌 Author

**Devanshu Dixit**  
Data & ML Practitioner | Fintech Enthusiast  
[LinkedIn](https://linkedin.com/in/devanshudixit) • [GitHub](https://github.com/devanshdixit)