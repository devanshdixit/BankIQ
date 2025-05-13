
# 💼 BankIQ: Loan Default Risk Prediction & Explainability

A complete end-to-end data science project simulating a bank's customer base, engineering financial and behavioral features, and predicting loan default risk using interpretable ML techniques like SHAP.

---

## 🚀 Overview

**Goal:** Build a comprehensive pipeline to simulate, analyze, and predict loan default risk — then explain the predictions using SHAP for interview and business stakeholder use.

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

### Top Predictors (from SHAP bar plot):
1. `AvgEMItoIncomeRatio`
2. `AvgLoanAmount`
3. `Income`
4. `LoanBurdenScore`
5. `CreditScore`

### Sample Force Plot:
Explains individual prediction breakdown (see `notebooks/3_SHAP_Explainability.ipynb`).

---

## 📁 Repository Structure

```
BankIQ/
├── data/
│   └── processed/ (sample .csv)
├── notebooks/
│   ├── 1_Data_Preparation.ipynb
│   ├── 2_Modeling.ipynb
│   └── 3_SHAP_Explainability.ipynb
├── scripts/
│   └── feature_engineering.py
├── README.md
```

---

## 🧪 How to Run

```bash
git clone https://github.com/<your-username>/BankIQ.git
cd BankIQ

# Install requirements
pip install -r requirements.txt

# Run feature engineering
python scripts/feature_engineering.py

# Launch Jupyter for notebooks
jupyter notebook
```

---

## 🎯 Use Case Alignment

✅ Credit Risk  
✅ Loan Approval Scoring  
✅ Customer Engagement Segmentation  
✅ Explainable AI (SHAP)

---

## 🙌 Author

Devanshu Dixit  
Data & ML Practitioner | Fintech Enthusiast  
[LinkedIn](https://linkedin.com/in/devanshu-dixit) | [GitHub](https://github.com/devanshudixit)
