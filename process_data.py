import pandas as pd
from sqlalchemy import create_engine
import time

# --- CONFIGURATION ---
RAW_DATA_PATH = 'customer_data.csv'
PROCESSED_DATA_PATH = 'processed_customer_data.csv'

def data_processing_pipeline(raw_df):
    """
    Processes the raw customer data using an in-memory SQL database.
    
    This function demonstrates an SQL-based data pipeline by:
    1. Loading a DataFrame into an in-memory SQLite database.
    2. Running SQL queries to perform feature engineering.
    3. Returning the processed data as a new DataFrame.
    """
    
    print("Initializing in-memory SQL engine...")
    # 'sqlite:///:memory:' creates a temporary database in RAM. 
    # It's fast and perfect for processing without a real database server.
    engine = create_engine('sqlite:///:memory:')

    # Load the raw data into the SQL database as a table named 'customers'
    print(f"Loading {len(raw_df)} records into the 'customers' SQL table...")
    raw_df.to_sql('customers', engine, index=False, if_exists='replace')

    # --- Feature Engineering with SQL ---
    # This is where we create new, valuable features from the existing data.
    # These features will help the recommendation model make better decisions.
    
    print("Running SQL queries for feature engineering...")
    
    query = """
    SELECT
        *,
        -- Feature 1: Debt-to-Income Ratio (DTI)
        -- A key indicator of financial health. Lower is generally better.
        CAST(total_debt AS REAL) / income AS debt_to_income_ratio,
        
        -- Feature 2: Customer Engagement Score
        -- A simple score to quantify how digitally active a customer is.
        (app_logins_per_month + website_visits_per_month) AS engagement_score,
        
        -- Feature 3: Product Count
        -- Directly measures product penetration, a key project KPI.
        -- We count the number of commas and add 1 to get the number of products.
        LENGTH(existing_products) - LENGTH(REPLACE(existing_products, ',', '')) + 1 AS product_count
        
    FROM
        customers
    """

    # Execute the query and load the results back into a Pandas DataFrame
    processed_df = pd.read_sql(query, engine)
    
    print("Feature engineering complete.")
    
    return processed_df

def main():
    """
    Main function to execute the data processing workflow.
    """
    start_time = time.time()
    
    try:
        # 1. Load the raw data from the CSV file
        print(f"Loading raw data from '{RAW_DATA_PATH}'...")
        raw_customer_df = pd.read_csv(RAW_DATA_PATH)
        
        # 2. Run the data processing pipeline
        processed_customer_df = data_processing_pipeline(raw_customer_df)
        
        # 3. Save the processed data to a new CSV file
        print(f"Saving processed data to '{PROCESSED_DATA_PATH}'...")
        processed_customer_df.to_csv(PROCESSED_DATA_PATH, index=False)
        
        end_time = time.time()
        
        print(f"\nSuccessfully processed data in {end_time - start_time:.2f} seconds.")
        print(f"Output saved to '{PROCESSED_DATA_PATH}'")

        # 4. Display a sample of the new, processed data to verify
        print("\n--- Processed Data Sample (First 5 Rows) ---")
        print(processed_customer_df.head())
        
    except FileNotFoundError:
        print(f"\nError: The file '{RAW_DATA_PATH}' was not found.")
        print("Please make sure you have run 'generate_data.py' first.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()