import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components

# Page config
st.set_page_config(page_title="Loan Default Risk Predictor", layout="wide")

st.title("💼 BankIQ: Loan Default Risk Predictor")
st.markdown("This app uses an XGBoost model to predict high-risk loans and explain the decisions using SHAP.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/devanshudixit/Desktop/projects/BankIQ/data/processed/enhanced_customers.csv")
    df = df.dropna(subset=["HighRiskLoan"])
    return df

df = load_data()

# Feature list
features = [
    "LoanBurdenScore", "AvgLoanAmount", "AvgEMItoIncomeRatio",
    "CreditScore", "Income", "Age"
]
target = "HighRiskLoan"

X = df[features]
y = df[target]

# Train model
@st.cache_resource
def train_model(X, y):
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return model, explainer, shap_values

model, explainer, shap_values = train_model(X, y)

# User selection
st.sidebar.header("🔍 Select a customer to explain")
index = st.sidebar.slider("Customer Index", 0, len(X) - 1, 0)

# Show customer input features
st.subheader("📌 Selected Customer Details")
st.dataframe(X.iloc[[index]])

# Show prediction
pred_prob = model.predict_proba(X.iloc[[index]])[0, 1]
st.metric("Risk Score (Probability of Default)", f"{pred_prob:.2%}")

st.subheader("🧠 SHAP Force Plot (Local Explanation)")
shap.initjs()
# Create the force plot
force_plot = shap.force_plot(
    base_value=explainer.expected_value,
    shap_values=explainer(X.iloc[[index]])[0].values,
    features=X.iloc[[index]],
    matplotlib=False
)

# Save as HTML and load the content
shap_html_path = "/tmp/shap_force_plot.html"
shap.save_html(shap_html_path, force_plot)

with open(shap_html_path, "r") as f:
    html_content = f.read()

# Display in Streamlit
components.html(html_content, height=300)

# SHAP summary plot (optional)
if st.checkbox("Show SHAP Summary Plot (Global Explanation)"):
    st.subheader("📊 SHAP Summary Plot")
    fig_summary, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=6, show=False)
    st.pyplot(fig_summary)