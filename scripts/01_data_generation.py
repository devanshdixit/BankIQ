import pandas as pd
import numpy as np
from faker import Faker
import random
import os
from datetime import datetime, timedelta

# Initialize Faker for generating realistic data
fake = Faker()

# Number of customers and transactions to simulate
NUM_CUSTOMERS = 1000
NUM_TRANSACTIONS = 10000
NUM_LOANS = 500
NUM_CAMPAIGNS = 50
NUM_SUPPORT = 2000

# Transaction categories
TRANSACTION_TYPES = ["Deposit", "Withdrawal", "Payment", "Transfer", "Fee"]
TRANSACTION_CATEGORIES = ["Shopping", "Bills", "Salary", "EMI", "Entertainment", "Groceries", "Insurance", "Miscellaneous"]

# Loan types and statuses
LOAN_TYPES = ["Home Loan", "Car Loan", "Personal Loan", "Education Loan"]
LOAN_STATUS = ["Active", "Closed", "Default"]

# Support interaction types
INTERACTION_TYPES = ["Call", "Email", "Chat", "Branch Visit"]
ISSUE_TYPES = ["General Inquiry", "Complaint", "Technical Issue", "Billing Issue"]
RESOLUTION_STATUS = ["Resolved", "Unresolved", "Pending"]

# Campaign channels
CHANNELS = ["Email", "SMS", "Push Notification"]
RESPONSE_TYPES = ["Positive", "Negative", "No Response"]

# Load existing customer data
customers = pd.read_csv("data/customers.csv")

# Generate transaction data
def generate_transactions(num_transactions):
    transactions = []
    for _ in range(num_transactions):
        customer_id = random.choice(customers["CustomerID"])
        transaction_id = fake.uuid4()
        date = fake.date_between(start_date="-2y", end_date="today")
        amount = round(random.uniform(10, 5000), 2)
        transaction_type = random.choice(TRANSACTION_TYPES)
        category = random.choice(TRANSACTION_CATEGORIES)
        channel = random.choice(["App", "Branch", "ATM", "Online"])
        
        transaction = {
            "TransactionID": transaction_id,
            "CustomerID": customer_id,
            "Date": date,
            "Amount": amount,
            "TransactionType": transaction_type,
            "Category": category,
            "Channel": channel,
        }
        transactions.append(transaction)
    return pd.DataFrame(transactions)

# Generate loan data
def generate_loans(num_loans):
    loans = []
    for _ in range(num_loans):
        customer_id = random.choice(customers["CustomerID"])
        loan_id = fake.uuid4()
        loan_type = random.choice(LOAN_TYPES)
        amount = random.randint(10000, 1000000)
        interest_rate = round(random.uniform(2, 12), 2)
        term_years = random.randint(1, 30)
        # Corrected date conversion to string
        start_date = fake.date_between(start_date="-10y", end_date="-1y").strftime("%Y-%m-%d")
        end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=term_years * 365)).strftime("%Y-%m-%d")
        status = random.choice(LOAN_STATUS)
        emi = round(amount / (term_years * 12), 2)
        
        loan = {
            "LoanID": loan_id,
            "CustomerID": customer_id,
            "LoanType": loan_type,
            "Amount": amount,
            "InterestRate": interest_rate,
            "TermYears": term_years,
            "EMI": emi,
            "StartDate": start_date,
            "EndDate": end_date,
            "Status": status,
        }
        loans.append(loan)
    return pd.DataFrame(loans)

# Generate campaign response data
def generate_campaign_responses(num_customers, num_campaigns):
    responses = []
    for _ in range(num_campaigns):
        customer_id = random.choice(customers["CustomerID"])
        campaign_id = fake.uuid4()
        date = fake.date_between(start_date="-1y", end_date="today")
        channel = random.choice(CHANNELS)
        response = random.choice(RESPONSE_TYPES)
        
        response_data = {
            "CustomerID": customer_id,
            "CampaignID": campaign_id,
            "Date": date,
            "Channel": channel,
            "Response": response,
        }
        responses.append(response_data)
    return pd.DataFrame(responses)

# Generate support interaction data
def generate_support_interactions(num_support):
    interactions = []
    for _ in range(num_support):
        customer_id = random.choice(customers["CustomerID"])
        interaction_id = fake.uuid4()
        date = fake.date_between(start_date="-1y", end_date="today")
        interaction_type = random.choice(INTERACTION_TYPES)
        issue_type = random.choice(ISSUE_TYPES)
        resolution = random.choice(RESOLUTION_STATUS)
        nps_score = random.randint(1, 10)
        
        interaction = {
            "InteractionID": interaction_id,
            "CustomerID": customer_id,
            "Date": date,
            "InteractionType": interaction_type,
            "IssueType": issue_type,
            "ResolutionStatus": resolution,
            "NPSScore": nps_score,
        }
        interactions.append(interaction)
    return pd.DataFrame(interactions)

# Save DataFrames to CSV
def save_to_csv(df, filename):
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

# Main function to generate and save data
def main():
    print("Generating transaction data...")
    transactions = generate_transactions(NUM_TRANSACTIONS)
    save_to_csv(transactions, "transactions.csv")
    
    print("Generating loan data...")
    loans = generate_loans(NUM_LOANS)
    save_to_csv(loans, "loans.csv")
    
    print("Generating campaign response data...")
    responses = generate_campaign_responses(NUM_CUSTOMERS, NUM_CAMPAIGNS)
    save_to_csv(responses, "campaign_responses.csv")
    
    print("Generating support interaction data...")
    interactions = generate_support_interactions(NUM_SUPPORT)
    save_to_csv(interactions, "support_interactions.csv")
    
    print("Data generation completed successfully!")

if __name__ == "__main__":
    main()
