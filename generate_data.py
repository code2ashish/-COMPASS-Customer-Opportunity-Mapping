import pandas as pd
import numpy as np
from faker import Faker
import random
import time

# --- CONFIGURATION ---
# Define constants for the script to make it easy to modify.
NUM_CUSTOMERS = 500000
OUTPUT_FILE = 'customer_data.csv'
PRODUCT_LIST = [
    "Savings Account", "Checking Account", "Credit Card", "Mortgage", 
    "Personal Loan", "Auto Loan", "Investment Account"
]

def generate_customer_data(num_customers):
    """
    Generates a synthetic dataset of banking customers.

    This function creates a list of customer records, where each record is a 
    dictionary. This approach is readable and mimics building data record-by-record.
    """
    
    # Initialize Faker for generating realistic placeholder data
    fake = Faker()
    
    records = []
    
    # Start timer to measure generation time
    start_time = time.time()

    print(f"Starting data generation for {num_customers} customers...")

    for i in range(num_customers):
        # --- Create individual data points for a single customer ---
        
        # Basic demographics
        age = random.randint(18, 80)
        income = random.randint(25000, 250000)
        
        # Financial profile. We'll make debt somewhat related to income later.
        credit_score = random.randint(300, 850)
        account_balance = round(random.uniform(500, 150000), 2)
        
        # To make the data more realistic, we tie debt to income. A person with a
        # higher income is more likely to have a higher debt load.
        debt_to_income_ratio = random.uniform(0.1, 1.5)
        total_debt = round(income * debt_to_income_ratio, 2)
        
        # Customer's existing relationship with the bank
        num_products = random.randint(1, 4)
        existing_products = ",".join(random.sample(PRODUCT_LIST, k=num_products))
        
        # Behavioral data
        payment_history = np.random.choice(['On-time', 'Late', 'Mixed'], p=[0.7, 0.1, 0.2])
        
        # --- Assemble the record ---
        customer_record = {
            'customer_id': i + 1,
            'age': age,
            'income': income,
            'city': fake.city(),
            'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Student'], p=[0.6, 0.2, 0.1, 0.1]),
            'credit_score': credit_score,
            'existing_products': existing_products,
            'account_balance': account_balance,
            'total_debt': total_debt,
            'number_of_open_accounts': random.randint(1, 15),
            'payment_history': payment_history,
            'app_logins_per_month': random.randint(0, 50),
            'customer_service_calls': random.randint(0, 10),
            'website_visits_per_month': random.randint(0, 30)
        }
        
        records.append(customer_record)

        # Provide a progress update to the user for long-running generation
        if (i + 1) % 100000 == 0:
            print(f"  ... {i + 1}/{num_customers} records generated.")

    end_time = time.time()
    print(f"Data generation finished in {end_time - start_time:.2f} seconds.")
    
    return records

def main():
    """
    Main function to run the data generation and save to a CSV file.
    """
    # 1. Generate the raw data
    customer_records = generate_customer_data(NUM_CUSTOMERS)
    
    # 2. Convert the list of dictionaries into a Pandas DataFrame
    # This is an efficient way to create a DataFrame from generated records.
    print("\nConverting records to a Pandas DataFrame...")
    customer_df = pd.DataFrame(customer_records)
    
    # 3. Save the DataFrame to a CSV file
    try:
        print(f"Saving DataFrame to '{OUTPUT_FILE}'...")
        customer_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccessfully created '{OUTPUT_FILE}'!")
        
        # 4. Display a sample of the data
        print("\n--- Data Sample (First 5 Rows) ---")
        print(customer_df.head())
        
    except Exception as e:
        print(f"\nError: Could not save the file. Reason: {e}")

# This is a standard Python convention to make sure the 'main' function
# is called only when the script is executed directly.
if __name__ == "__main__":
    main()