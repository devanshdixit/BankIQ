import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Initialize Faker for generating realistic data
fake = Faker()

# Number of customers to simulate
NUM_CUSTOMERS = 1000

# Define product types
PRODUCTS = ["Savings Account", "Credit Card", "Mortgage", "Investment Account", "Fixed Deposit"]

# Helper function to generate a random risk profile
def risk_profile():
    return random.choice(["Low", "Medium", "High"])

# Generate synthetic customer data
def generate_customers(num_customers):
    customers = []
    for _ in range(num_customers):
        customer_id = fake.uuid4()
        name = fake.name()
        age = random.randint(18, 70)
        gender = random.choice(["Male", "Female", "Other"])
        income = random.randint(20000, 200000)
        location = fake.city()
        credit_score = random.randint(300, 850)
        risk = risk_profile()
        relationship_length = random.randint(1, 20)
        marital_status = random.choice(["Single", "Married", "Divorced"])
        
        customer = {
            "CustomerID": customer_id,
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Income": income,
            "Location": location,
            "CreditScore": credit_score,
            "RiskProfile": risk,
            "RelationshipLength": relationship_length,
            "MaritalStatus": marital_status,
        }
        customers.append(customer)
    return pd.DataFrame(customers)

# Generate product ownership data
def generate_products(customers):
    products = []
    for customer_id in customers["CustomerID"]:
        num_products = random.randint(1, len(PRODUCTS))
        selected_products = random.sample(PRODUCTS, num_products)
        
        for product in selected_products:
            open_date = fake.date_between(start_date='-20y', end_date='today')
            active_status = random.choice(["Active", "Inactive"])
            credit_limit = random.randint(1000, 50000) if product == "Credit Card" else np.nan
            balance = round(random.uniform(1000, 50000), 2)
            usage_score = round(random.uniform(0, 1), 2)
            
            product_record = {
                "CustomerID": customer_id,
                "ProductType": product,
                "OpenDate": open_date,
                "ActiveStatus": active_status,
                "CreditLimit": credit_limit,
                "Balance": balance,
                "UsageScore": usage_score,
            }
            products.append(product_record)
    return pd.DataFrame(products)

# Save DataFrames to CSV
def save_to_csv(df, filename):
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

# Main function to generate and save data
def main():
    print("Generating customer data...")
    customers = generate_customers(NUM_CUSTOMERS)
    print("Generating product ownership data...")
    products = generate_products(customers)

    # Save data to CSV files
    save_to_csv(customers, "customers.csv")
    save_to_csv(products, "products.csv")
    print("Data generation completed successfully!")

if __name__ == "__main__":
    main()
