import pandas as pd
import numpy as np
import os

# Create a directory for processed data
os.makedirs("data/processed", exist_ok=True)

# Load cleaned data
customers = pd.read_csv("data/cleaned/cleaned_customers.csv")
products = pd.read_csv("data/cleaned/cleaned_products.csv")
transactions = pd.read_csv("data/cleaned/cleaned_transactions.csv")
loans = pd.read_csv("data/cleaned/cleaned_loans.csv")
campaign_responses = pd.read_csv("data/cleaned/cleaned_campaign_responses.csv")
support_interactions = pd.read_csv("data/cleaned/cleaned_support_interactions.csv")

print("Cleaned data loaded successfully!")

def feature_engineering():
    # ---------------------
    # Customer Profile Features
    # ---------------------
    print("Generating customer profile features...")
    bins = [0, 25, 35, 50, 65, 100]
    labels = ["Youth", "Young Adult", "Middle Aged", "Senior", "Elderly"]
    customers["AgeGroup"] = pd.cut(customers["Age"], bins=bins, labels=labels)

    bins = [0, 30000, 70000, 120000, 200000]
    labels = ["Low", "Medium", "High", "Very High"]
    customers["IncomeBracket"] = pd.cut(customers["Income"], bins=bins, labels=labels)

    customers["RiskScore"] = np.where(customers["CreditScore"] >= 700, "Low", 
                                      np.where(customers["CreditScore"] >= 500, "Medium", "High"))

    # ---------------------
    # Product Usage Features
    # ---------------------
    print("Generating product features...")
    product_count = products.groupby("CustomerID")["ProductType"].count().reset_index()
    product_count.columns = ["CustomerID", "ProductCount"]

    active_count = products[products["ActiveStatus"] == "Active"].groupby("CustomerID")["ProductType"].count().reset_index()
    active_count.columns = ["CustomerID", "ActiveProductCount"]

    product_features = product_count.merge(active_count, on="CustomerID", how="left")
    product_features["ActiveProductCount"] = product_features["ActiveProductCount"].fillna(0)

    # New Feature: Product Engagement Score
    product_features["ProductEngagementScore"] = np.where(
        product_features["ProductCount"] > 0,
        product_features["ActiveProductCount"] / product_features["ProductCount"],
        0
    )

    # ---------------------
    # Financial Features
    # ---------------------
    print("Generating financial features...")
    avg_transaction = transactions.groupby("CustomerID")["Amount"].mean().reset_index()
    avg_transaction.columns = ["CustomerID", "AvgTransactionAmount"]

    trans_count = transactions.groupby("CustomerID")["TransactionID"].count().reset_index()
    trans_count.columns = ["CustomerID", "TransactionFrequency"]

    financial_features = avg_transaction.merge(trans_count, on="CustomerID", how="left")

    # ---------------------
    # Loan Features
    # ---------------------
    print("Generating loan features...")
    loans["EMItoIncomeRatio"] = np.where(
        (loans["Amount"].isnull()) | (loans["Amount"] == 0), 
        np.nan, 
        loans["EMI"] / loans["Amount"]
    )
    loans["EMItoIncomeRatio"].replace([np.inf, -np.inf], np.nan, inplace=True)

    loans["HighRiskLoan"] = np.where(loans["InterestRate"] > 10, 1, 0)

    avg_loan = loans.groupby("CustomerID").agg(
        AvgLoanAmount=("Amount", "mean"),
        AvgEMItoIncomeRatio=("EMItoIncomeRatio", "mean"),
        HighRiskLoan=("HighRiskLoan", "max")
    ).reset_index()

    # New Feature: Loan Burden Score
    avg_loan["LoanBurdenScore"] = avg_loan["AvgLoanAmount"] * avg_loan["AvgEMItoIncomeRatio"]

    # ---------------------
    # Engagement Features
    # ---------------------
    print("Generating engagement features...")
    interaction_count = support_interactions.groupby("CustomerID")["InteractionID"].count().reset_index()
    interaction_count.columns = ["CustomerID", "SupportFrequency"]

    support_interactions["NPSBucket"] = pd.cut(support_interactions["NPSScore"], bins=[0, 3, 7, 10], labels=["Low", "Medium", "High"])

    # ---------------------
    # Merging All Features
    # ---------------------
    print("Merging all features...")
    enhanced_customers = customers.merge(product_features, on="CustomerID", how="left")
    enhanced_customers = enhanced_customers.merge(financial_features, on="CustomerID", how="left")
    enhanced_customers = enhanced_customers.merge(avg_loan, on="CustomerID", how="left")
    enhanced_customers = enhanced_customers.merge(interaction_count, on="CustomerID", how="left")

    numeric_cols = enhanced_customers.select_dtypes(include=[np.number]).columns
    categorical_cols = enhanced_customers.select_dtypes(include=['category', 'object']).columns

    enhanced_customers[numeric_cols] = enhanced_customers[numeric_cols].fillna(0)

    for col in categorical_cols:
        if isinstance(enhanced_customers[col].dtype, pd.CategoricalDtype):
            if "Unknown" not in enhanced_customers[col].cat.categories:
                enhanced_customers[col] = enhanced_customers[col].cat.add_categories(["Unknown"])
        enhanced_customers[col] = enhanced_customers[col].fillna("Unknown")

    filepath = "data/processed/enhanced_customers.csv"
    enhanced_customers.to_csv(filepath, index=False)
    print(f"Processed data saved to {filepath}")

def main():
    print("Starting feature engineering...")
    feature_engineering()
    print("Feature engineering completed successfully!")

if __name__ == "__main__":
    main()