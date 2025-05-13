import pandas as pd
import numpy as np
import os

# Create a directory for cleaned data
os.makedirs("data/cleaned", exist_ok=True)

def clean_customers(df):
    # Drop duplicates based on CustomerID
    df = df.drop_duplicates(subset=["CustomerID"])
    
    # Handle missing values
    df["Name"] = df["Name"].fillna("Unknown")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Gender"] = df["Gender"].fillna("Other")
    df["Income"] = df["Income"].fillna(df["Income"].mean())
    df["CreditScore"] = df["CreditScore"].fillna(df["CreditScore"].mean())
    df["RiskProfile"] = df["RiskProfile"].fillna("Medium")
    df["RelationshipLength"] = df["RelationshipLength"].fillna(df["RelationshipLength"].median())
    df["MaritalStatus"] = df["MaritalStatus"].fillna("Single")
    
    # Standardize Gender values
    df["Gender"] = df["Gender"].replace({"M": "Male", "F": "Female", "O": "Other"})
    
    return df

def clean_products(df):
    # Drop duplicates based on CustomerID and ProductType
    df = df.drop_duplicates(subset=["CustomerID", "ProductType"])
    
    # Handle missing balances with the median balance
    df["Balance"] = df["Balance"].fillna(df["Balance"].median())
    
    # Remove negative balances
    df = df[df["Balance"] >= 0]

    # Fill missing credit limits with the median of existing credit card limits
    median_limit = df.loc[df["ProductType"] == "Credit Card", "CreditLimit"].median()
    df["CreditLimit"] = df["CreditLimit"].fillna(median_limit)
    
    return df

def clean_transactions(df):
    # Drop duplicates based on TransactionID
    df = df.drop_duplicates(subset=["TransactionID"])

    # Handle missing values in transaction amount
    df["Amount"] = df["Amount"].fillna(df["Amount"].mean())

    # Remove transactions with negative or zero amounts
    df = df[df["Amount"] > 0]

    # Standardize transaction types
    valid_types = ["Deposit", "Withdrawal", "Payment", "Transfer", "Fee"]
    df = df[df["TransactionType"].isin(valid_types)]
    
    return df

def clean_loans(df):
    # Drop duplicates based on LoanID
    df = df.drop_duplicates(subset=["LoanID"])
    
    # Fill missing interest rates and EMIs with the median
    df["InterestRate"] = df["InterestRate"].fillna(df["InterestRate"].median())
    df["EMI"] = df["EMI"].fillna(df["EMI"].median())
    
    # Remove loans with unrealistic negative values
    df = df[(df["Amount"] > 0) & (df["InterestRate"] >= 0)]
    
    return df

def clean_campaign_responses(df):
    # Drop duplicates based on CampaignID
    df = df.drop_duplicates(subset=["CampaignID"])

    # Fill missing responses with "No Response"
    df["Response"] = df["Response"].fillna("No Response")

    # Standardize response types
    valid_responses = ["Positive", "Negative", "No Response"]
    df = df[df["Response"].isin(valid_responses)]
    
    return df

def clean_support_interactions(df):
    # Drop duplicates based on InteractionID
    df = df.drop_duplicates(subset=["InteractionID"])

    # Handle missing NPS scores with median
    df["NPSScore"] = df["NPSScore"].fillna(df["NPSScore"].median())

    # Standardize resolution statuses
    valid_resolutions = ["Resolved", "Unresolved", "Pending"]
    df = df[df["ResolutionStatus"].isin(valid_resolutions)]
    
    return df

def save_cleaned_data(df, filename):
    filepath = os.path.join("data/cleaned", filename)
    df.to_csv(filepath, index=False)
    print(f"Cleaned data saved to {filepath}")

def main():
    # Load and clean each dataset
    print("Cleaning customer data...")
    customers = pd.read_csv("data/customers.csv")
    customers = clean_customers(customers)
    save_cleaned_data(customers, "cleaned_customers.csv")

    print("Cleaning product data...")
    products = pd.read_csv("data/products.csv")
    products = clean_products(products)
    save_cleaned_data(products, "cleaned_products.csv")

    print("Cleaning transaction data...")
    transactions = pd.read_csv("data/transactions.csv")
    transactions = clean_transactions(transactions)
    save_cleaned_data(transactions, "cleaned_transactions.csv")

    print("Cleaning loan data...")
    loans = pd.read_csv("data/loans.csv")
    loans = clean_loans(loans)
    save_cleaned_data(loans, "cleaned_loans.csv")

    print("Cleaning campaign response data...")
    responses = pd.read_csv("data/campaign_responses.csv")
    responses = clean_campaign_responses(responses)
    save_cleaned_data(responses, "cleaned_campaign_responses.csv")

    print("Cleaning support interaction data...")
    interactions = pd.read_csv("data/support_interactions.csv")
    interactions = clean_support_interactions(interactions)
    save_cleaned_data(interactions, "cleaned_support_interactions.csv")

    print("Data cleaning completed successfully!")

if __name__ == "__main__":
    main()
