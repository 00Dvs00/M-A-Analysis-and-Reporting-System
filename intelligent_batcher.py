import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging

# --- Import our project's functions ---
# We need the function to get the S&P 500 list
from get_sp500_tickers import get_sp500_info
# We need the main ETL function to run the process
from etl_pipeline import run_etl_pipeline

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def find_missing_companies():
    """
    Connects to the database, compares the companies present against a master
    list of S&P 500 companies, and returns a DataFrame of the missing ones.
    """
    # --- Step 1: Connect to the Database ---
    load_dotenv()
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        logging.error("Database credentials not found in .env file. Aborting.")
        return None

    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        engine = create_engine(DATABASE_URL)
        logging.info("Successfully connected to the database.")
    except Exception as e:
        logging.error(f"Failed to connect to the database: {e}")
        return None

    # --- Step 2: Get the Master List of S&P 500 Companies ---
    master_df = get_sp500_info()
    if master_df is None:
        logging.error("Could not fetch the master S&P 500 list. Aborting.")
        return None
    
    # --- Step 3: Get the List of Companies Already in Our Database ---
    try:
        db_df = pd.read_sql("SELECT ticker, cik FROM companies", engine)
        logging.info(f"Found {len(db_df)} companies already in the database.")
    except Exception as e:
        logging.error(f"Failed to query companies from the database: {e}")
        # If the table doesn't exist or is empty, create an empty DataFrame
        db_df = pd.DataFrame(columns=['ticker', 'cik'])
        
    # --- Step 4: Compare and Find What's Missing ---
    # We identify companies as missing if their CIK is not in our database.
    # This correctly handles cases like GOOG/GOOGL.
    if not db_df.empty:
        missing_df = master_df[~master_df['cik'].isin(db_df['cik'])]
    else:
        # If our database is empty, then all companies are "missing"
        missing_df = master_df

    return missing_df


if __name__ == "__main__":
    print("\n=======================================================")
    print("=== Running Intelligent Batcher to find and process ===")
    print("===          missing S&P 500 companies            ===")
    print("=======================================================\n")
    
    missing_companies_df = find_missing_companies()
    
    # --- Step 5: Execute the ETL for the Missing Companies ---
    if missing_companies_df is not None:
        if missing_companies_df.empty:
            logging.info("SUCCESS: The database is already up-to-date with all S&P 500 companies.")
        else:
            num_missing = len(missing_companies_df)
            logging.warning(f"Found {num_missing} missing companies. Preparing a targeted ETL run.")
            
            # Display the first few missing tickers for the user's information
            print("-------------------------------------------------------")
            print(f"Sample of missing tickers: {', '.join(missing_companies_df['ticker'].head().tolist())}")
            print("-------------------------------------------------------")

            # Run the ETL pipeline with only the DataFrame of missing companies
            # This is where the magic happens!
            run_etl_pipeline(tickers_to_process=missing_companies_df, num_filings_per_form=4)

            logging.info(f"Intelligent batch run for {num_missing} companies is complete.")